import time
import random
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from nets.models import GraphNIModel
from env.MCEnv import MCEnv
from env.MISEnv import MISEnv
from utils.utils import load_test_data, print_epoch_results, print_epoch_header, generate_word, set_random_seeds
from args.train_ni_args import get_args

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    def __init__(self, params):
        self.params = params
        self.verbose = self.params['verbose']

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.params['device'] = device
        if self.verbose:
            print(f"Using: {device}")
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        if self.params['seed'] is not None:
            set_random_seeds(self.params['seed'])

        if self.params['problem'] == 'mc':
            Env = MCEnv
        elif self.params['problem'] == 'mis':
            Env = MISEnv
        else:
            raise ValueError(f"Invalid problem: {self.params['problem']}")

        self.ni_model = GraphNIModel(**self.params).to(device)
        if self.params['compile']:
            self.ni_model = torch.compile(self.ni_model)

        if self.verbose:
            print(f'Number of parameters: {sum(p.numel() for p in self.ni_model.parameters() if p.requires_grad)}')

        self.env = Env(self.params, device, testing=False)
        self.test_env = Env(self.params, device, testing=True)

        self.optimizer = AdamW(
            self.ni_model.parameters(),
            lr=self.params['lr'],
            betas=self.params['betas'],
            weight_decay=self.params['weight_decay'],
        )
        self.scaler = GradScaler(enabled=self.params.get('amp', False))

        if self.params['ni_model_load_path'] != '':
            checkpoint = torch.load(self.params['ni_model_load_path'], map_location=self.device, weights_only=False)
            self.ni_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.params['lr']

        self.test_graphs = load_test_data(
            self.params['problem'], self.params['eval_graph_types'], self.params['num_eval_graphs']
        )
        self.replay_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = (
            Path("runs/train")
            / f"{self.params['problem']}_ni"
            / f"{self.params['train_graph_type'].lower()}"
            / f"{timestamp}"
        )
        self.run_path = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        params_file = run_dir / "params.json"
        with params_file.open("w") as fp:
            json.dump(self.params, fp, indent=4)

        if self.params['wandb']:
            if wandb is None:
                raise ImportError("wandb is not installed. Install wandb or run with --no-wandb.")
            project_name = f'PB_NCO_NI_{self.params["problem"]}'
            wandb.login()
            wandb.init(project=project_name, name=self.params['run_id'] or None, config=self.params)

    # ------------------------------------------------------------------
    # Advantage estimation
    # ------------------------------------------------------------------

    def _compute_discounted_advantages(self, rewards_list, gamma, batch_size, pop_size, steps):
        """Standard REINFORCE: γ-discounted returns minus leave-one-out population baseline."""
        discounted_sum = torch.zeros(batch_size, pop_size, device='cpu')
        ep_returns = []
        for r in reversed(rewards_list):
            discounted_sum = r + gamma * discounted_sum
            ep_returns.insert(0, discounted_sum.clone())

        returns_stack = torch.stack(ep_returns, dim=0)  # [steps, batch, pop]
        advantages_list = []
        for t in range(steps):
            returns_t = returns_stack[t]
            if pop_size > 1:
                sum_r = returns_t.sum(dim=1, keepdim=True)
                baseline_t = (sum_r - returns_t) / (pop_size - 1)  # leave-one-out
                adv_t = returns_t - baseline_t
            else:
                adv_t = returns_t
            advantages_list.append(adv_t)
        return advantages_list

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        start_time = time.time()
        for epoch in range(1, self.params['n_epochs'] + 1):
            if self.verbose:
                print_epoch_header(epoch, self.params['n_epochs'], start_time)

            epoch_results = {
                'train/best_reward_values_mean': [],
                'train/best_reward_values_max': [],
                'train/avg_reward_values': [],
                'train/loss_values': [],
                'train/entropy_values': [],
                'train/revisited': [],
                'train/avg_similarity': [],
                'train/max_similarity': [],
                'train/self_mem_percentage': [],
                'train/total_steps': [],
                'train/train_sampled_n': [],
                'train/batch_size': [self.params['batch_size']],
                'train/pop_size': [self.params['pop_size']],
                'train/train_max_steps_multiplier': [self.params['train_max_steps_multiplier']],
                'train/kl_early_stopped': [],
            }
            self.replay_buffer = []

            # -------- EPISODE COLLECTION --------
            for episode in range(self.params['n_episodes']):
                episode_data, episode_logs, used_size = self.run_episode()
                self.replay_buffer.append(episode_data)
                epoch_results['train/best_reward_values_mean'].append(episode_logs['best_reward_values_mean'])
                epoch_results['train/best_reward_values_max'].append(episode_logs['best_reward_values_max'])
                epoch_results['train/avg_reward_values'].append(np.mean(episode_logs['avg_reward_values']))
                epoch_results['train/revisited'].append(np.mean(episode_logs['revisited']))
                epoch_results['train/avg_similarity'].append(np.mean(episode_logs['avg_similarity']))
                epoch_results['train/max_similarity'].append(np.mean(episode_logs['max_similarity']))
                epoch_results['train/self_mem_percentage'].append(np.mean(episode_logs['self_mem_percentage']))
                epoch_results['train/total_steps'].append(episode_logs['total_steps'])
                epoch_results['train/train_sampled_n'].append(used_size)

            if self.verbose:
                print_epoch_results(epoch_results)

            # -------- PPO UPDATE --------
            avg_loss, avg_entropy, kl_stopped = self.ppo_update_from_buffer(epoch)
            self.replay_buffer = []
            epoch_results['train/loss_values'].append(avg_loss)
            epoch_results['train/entropy_values'].append(avg_entropy)
            epoch_results['train/kl_early_stopped'].append(float(kl_stopped))

            # -------- LOGGING --------
            if self.params['wandb']:
                wandb.log({k: np.mean(v) for k, v in epoch_results.items()})

            # -------- EVAL / CKPT --------
            if ((epoch - 1) % self.params['eval_freq'] == 0) or (epoch == self.params['n_epochs']):
                test_results, wandb_results = self.run_test()
                if self.params['wandb']:
                    wandb.log(wandb_results)
                if self.params['save_models']:
                    self.save_model(epoch)

            self.env.cur_epoch += 1

        self.save_model(self.params['n_epochs'])

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_episode(self):
        episode_logs = {
            'problem_size': self.env.problem_size,
            'best_reward_values': None,
            'avg_reward_values': [],
            'loss_values': [],
            'revisited': [],
            'avg_similarity': [],
            'max_similarity': [],
            'self_mem_percentage': [],
            'total_steps': 0,
        }

        self.ni_model.train()

        states_list = []
        actions_list = []
        old_log_probs_list = []
        rewards_list = []

        state, done = self.env.reset(self.params['batch_size'], self.params['pop_size'])

        steps = 0
        while not done:
            state_cpu = state.cpu()
            states_list.append(state_cpu)

            with autocast('cuda', enabled=self.params.get('amp', False)):
                logits = self.ni_model(state)

            log_p = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    f"NaN/Inf in model output at step {steps}. "
                    f"logits: min={logits.min().item():.3f}, max={logits.max().item():.3f}, "
                    f"nan_count={torch.isnan(logits).sum().item()}. "
                    "Model weights likely corrupted by a prior NaN gradient update."
                )
            actions = probs.multinomial(num_samples=1).squeeze(-1)
            chosen_log_p = log_p.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)

            next_state, R_dict, done = self.env.step(actions)
            steps += 1

            rewards = R_dict['Reward'].reshape(
                self.params['batch_size'], self.params['pop_size']
            ).detach().cpu()

            actions_list.append(actions.cpu())
            old_log_probs_list.append(chosen_log_p.detach().cpu())
            rewards_list.append(rewards)

            if episode_logs['best_reward_values'] is None:
                episode_logs['best_reward_values'] = R_dict['Objective Value Reward']
            else:
                episode_logs['best_reward_values'] += R_dict['Objective Value Reward']
            episode_logs['avg_reward_values'].append(rewards.mean().item())
            episode_logs['revisited'].append(R_dict['Re-Visited'].mean().item())
            episode_logs['avg_similarity'].append(
                (R_dict['Avg similarity'] / self.params['batch_size']).mean().item()
            )
            episode_logs['max_similarity'].append(
                (R_dict['Max similarity'] / self.params['batch_size']).mean().item()
            )
            episode_logs['self_mem_percentage'].append(R_dict['Self memory percentage'].mean().item())
            episode_logs['total_steps'] += 1

            state = next_state
            if done:
                break

        batch_size = self.params['batch_size']
        pop_size = self.params['pop_size']
        gamma = self.params['gamma']

        # ---- Advantage estimation ----
        advantages_list = self._compute_discounted_advantages(
            rewards_list, gamma, batch_size, pop_size, steps
        )

        episode_logs['best_reward_values_mean'] = episode_logs['best_reward_values'].mean().item()
        episode_logs['best_reward_values_max'] = episode_logs['best_reward_values'].max().item()

        # ---- Transition selection ----
        n_to_store = min(self.params['n_stored_states'], steps)
        if n_to_store >= 2:
            step_size = (steps - 1) / (n_to_store - 1)
            indices = [int(round(i * step_size)) for i in range(n_to_store)]
        elif n_to_store == 1:
            indices = [0]
        else:
            raise ValueError(f"n_stored_states must be >= 1, got {self.params['n_stored_states']}")
        stored_states = [states_list[i] for i in indices]
        stored_actions = [actions_list[i] for i in indices]
        stored_old_log_probs = [old_log_probs_list[i] for i in indices]
        stored_advantages = [advantages_list[i] for i in indices]

        if self.params['normalize_advantages']:
            # Normalize over the full episode (before any subsampling above) for a stable scale
            adv_tensor = torch.cat([adv.reshape(-1) for adv in advantages_list])
            adv_mean = adv_tensor.mean()
            adv_std = adv_tensor.std(unbiased=False) + 1e-8
            stored_advantages = [(adv - adv_mean) / adv_std for adv in stored_advantages]

        episode_data = (stored_states, stored_actions, stored_old_log_probs, stored_advantages)
        return episode_data, episode_logs, self.env.problem_size

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update_from_buffer(self, epoch: int):
        """
        Merge all episodes in the replay buffer and run PPO for ppo_epochs passes.

        Fixes vs v1:
          - Loss is divided by batch_count before backward so gradient magnitude is
            independent of ppo_update_batch_count (was incorrectly summed in v1).

        Optional improvements (controlled by params):
          - KL early stopping: breaks out of PPO epochs when mean approx-KL > kl_target.

        Returns (avg_loss, avg_entropy, kl_early_stopped).
        """
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        for (states_list, actions_list, old_log_probs_list, advantages_list) in self.replay_buffer:
            all_states.extend(states_list)
            all_actions.extend(actions_list)
            all_old_log_probs.extend(old_log_probs_list)
            all_advantages.extend(advantages_list)

        T_total = len(all_states)
        entropy_coef = self.params['entropy_coef']
        kl_target = self.params.get('kl_target', 0.0) or 0.0
        kl_stopped = False

        total_loss_accum = 0.0
        total_entropy_accum = 0.0
        n_updates = 0
        idxs = list(range(T_total))

        for ppo_epoch in range(self.params['ppo_epochs']):
            if kl_stopped:
                break

            random.shuffle(idxs)
            accumulated_loss = 0.0
            accumulated_entropy = 0.0
            batch_count = 0
            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for t in range(T_total):
                cur_idx = idxs[t]
                state_t = all_states[cur_idx].to(self.device)
                actions_t = all_actions[cur_idx].to(self.device)
                old_logp_t = all_old_log_probs[cur_idx].to(self.device)
                adv_t = all_advantages[cur_idx].reshape(-1).to(self.device)

                with autocast('cuda', enabled=self.params.get('amp', False)):
                    new_logits = self.ni_model(state_t)

                new_logp = F.log_softmax(new_logits, dim=-1)
                new_logp_actions = new_logp.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

                if entropy_coef == 0:
                    with torch.no_grad():
                        probs = F.softmax(new_logits, dim=-1)
                else:
                    probs = F.softmax(new_logits, dim=-1)

                safe_log_p = torch.nan_to_num(new_logp, nan=0.0, neginf=0.0, posinf=0.0)
                entropy = -(safe_log_p * probs).sum(dim=-1).mean()

                ratio = torch.exp(new_logp_actions - old_logp_t)
                surrogate = torch.min(
                    ratio * adv_t,
                    torch.clamp(ratio, 1 - self.params['ppo_clip'], 1 + self.params['ppo_clip']) * adv_t,
                ).mean()

                loss = -surrogate - entropy_coef * entropy

                accumulated_loss += loss
                accumulated_entropy += entropy.item()

                # Approximate KL: E[log π_old - log π_new]
                with torch.no_grad():
                    approx_kl = (old_logp_t - new_logp_actions).mean().item()
                epoch_kl_sum += approx_kl
                epoch_kl_count += 1

                batch_count += 1
                last_step = (t == T_total - 1)
                if (batch_count >= self.params['ppo_update_batch_count']) or last_step:
                    update_loss = accumulated_loss / batch_count
                    if torch.isnan(update_loss) or torch.isinf(update_loss):
                        print(
                            f"[WARNING] NaN/Inf loss detected (loss={update_loss.item():.4f}), "
                            f"skipping gradient update to avoid corrupting model weights."
                        )
                        accumulated_loss = 0.0
                        accumulated_entropy = 0.0
                        batch_count = 0
                        continue
                    self.optimizer.zero_grad()
                    self.scaler.scale(update_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.ni_model.parameters(), self.params['max_grad_norm'])
                    if torch.isnan(grad_norm):
                        print("[WARNING] NaN gradient norm detected, skipping optimizer step.")
                        self.optimizer.zero_grad()
                        self.scaler.update()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    torch.cuda.empty_cache()

                    total_loss_accum += (accumulated_loss / batch_count).item()
                    total_entropy_accum += accumulated_entropy / batch_count
                    accumulated_loss = 0.0
                    accumulated_entropy = 0.0
                    batch_count = 0
                    n_updates += 1

            # KL early stopping: check at the end of each PPO epoch
            if kl_target > 0 and epoch_kl_count > 0:
                mean_kl = epoch_kl_sum / epoch_kl_count
                if mean_kl > kl_target:
                    kl_stopped = True
                    if self.verbose:
                        print(
                            f"  [KL early stop] mean KL {mean_kl:.4f} > target {kl_target:.4f} "
                            f"after PPO epoch {ppo_epoch + 1}/{self.params['ppo_epochs']}"
                        )

        avg_loss = total_loss_accum / n_updates if n_updates > 0 else 0.0
        avg_entropy = total_entropy_accum / n_updates if n_updates > 0 else 0.0
        return avg_loss, avg_entropy, kl_stopped

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, epoch):
        path = self.run_path / f"epoch{epoch}.pth"
        torch.save({
            'model_state_dict': self.ni_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params,
        }, path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_test(self,):
        self.ni_model.eval()
        test_results = {}
        wandb_results = {}

        for i, (cur_test_graph_list, graph_type) in enumerate(zip(self.test_graphs, self.params['eval_graph_types'])):
            sum_best_obj = 0.0
            sum_avg_obj = 0.0
            sum_revisited = 0.0
            sum_total_revisited = 0.0
            sum_self_mem_perc = 0.0
            sum_elapsed_times = 0.0
            wandb_results_i = {}

            for j, cur_test_graph in enumerate(cur_test_graph_list):
                if isinstance(cur_test_graph, tuple):
                    test_batch_size, problem_size, _ = cur_test_graph[0].size()
                elif len(cur_test_graph.size()) == 3:
                    test_batch_size, problem_size, _ = cur_test_graph.size()
                else:
                    problem_size, _ = cur_test_graph.size()
                    test_batch_size = 1
                    cur_test_graph = cur_test_graph.unsqueeze(0)

                test_pop_size = self.params['test_pop_size']
                self.test_env.problem_size = problem_size
                self.ni_model.edge_embeddings_computed = False
                start_time = time.time()
                state, done = self.test_env.reset(
                    test_batch_size, test_pop_size, test_graph=cur_test_graph, seed=self.params['seed']
                )

                steps = 0
                self_mem_perc = []
                revisited = []
                total_revisited = []
                elapsed_time = [0.0]
                while not done:
                    with autocast('cuda', enabled=self.params.get('amp', False)):
                        logits = self.ni_model(state)

                    topk_actions = torch.topk(logits, self.params['topk'], dim=1).indices

                    for k in range(self.params['topk']):
                        steps += 1
                        actions = topk_actions[:, k]

                        if k > 0 and state.mask is not None:
                            exit_loop = False
                            for s in range(test_pop_size):
                                if state.mask[s, actions[s]] == 1:
                                    exit_loop = True
                            if exit_loop:
                                break

                        state, R_dict, done = self.test_env.step(actions)

                        self_mem_perc.append(R_dict['Self memory percentage'].cpu().numpy())
                        revisited.append(R_dict['Re-Visited'].reshape(test_batch_size, test_pop_size).cpu().numpy())
                        total_revisited.append(R_dict['Total Re-Visited'].reshape(test_batch_size, test_pop_size).cpu().numpy())
                        elapsed_time.append(time.time() - start_time)

                best_objective_values = self.test_env.best_objective_values.reshape(test_batch_size, test_pop_size)
                all_best_obj_values = best_objective_values.max(dim=1).values.mean().item()
                all_avg_obj_values = best_objective_values.mean(dim=1).mean().item()

                if self.verbose:
                    print(
                        f"=== Testing with {test_batch_size} graphs of size {problem_size}, "
                        f"pop size: {test_pop_size}, max steps: {self.test_env.max_iterations}, "
                        f"patience: {self.test_env.patience} ==="
                    )
                    total_time = elapsed_time[-1]
                    last_revisited = np.array(revisited).mean()
                    print(
                        f"Best obj: {all_best_obj_values:.3f}. Average obj: {all_avg_obj_values:.3f}. "
                        f"Revisited: {last_revisited:.5f} Avg steps: {steps:.2f}. "
                        f"Tot time: {total_time:.2f}s ({total_time / test_batch_size:.2f}s/instance)."
                    )

                sum_best_obj += all_best_obj_values
                sum_avg_obj += all_avg_obj_values
                sum_revisited += np.array(revisited).mean()
                sum_total_revisited += np.array(total_revisited).mean()
                sum_self_mem_perc += np.array(self_mem_perc).mean()
                sum_elapsed_times += elapsed_time[-1]

            n_graph_batches = len(cur_test_graph_list)
            wandb_results_i[f'{graph_type}/Best Obj'] = sum_best_obj / n_graph_batches
            wandb_results_i[f'{graph_type}/Avg Obj'] = sum_avg_obj / n_graph_batches
            wandb_results_i[f'{graph_type}/Revisited'] = sum_revisited / n_graph_batches
            wandb_results_i[f'{graph_type}/Total Revisited'] = sum_total_revisited / n_graph_batches
            wandb_results_i[f'{graph_type}/Self Memory'] = sum_self_mem_perc / n_graph_batches
            wandb_results_i[f'{graph_type}/Elapsed Time'] = sum_elapsed_times / n_graph_batches

            wandb_results.update(wandb_results_i)

        return test_results, wandb_results


def main():
    params = get_args()
    if params['debug']:
        params['verbose'] = True
        params['wandb'] = False
        params['save_models'] = False
        params['min_problem_size'] = 20
        params['max_problem_size'] = 20
        params['n_epochs'] = 10
        params['n_episodes'] = 10
        params['eval_graph_types'] = ['ER100']
        params['num_eval_graphs'] = [100]

    if params['distance_metric'] == 'default':
        if params['problem'] == 'mc':
            params['distance_metric'] = 'edge_hamming'
        elif params['problem'] == 'mis':
            params['distance_metric'] = 'node_hamming'

    execution_name = generate_word(6)
    params['execution_name'] = execution_name

    if params['verbose'] and params.get('world_size', 1) == 1:
        print(f'Execution name: {execution_name}')
        print(params)

    trainer = Trainer(params)
    trainer.train()


if __name__ == "__main__":
    main()
