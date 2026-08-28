import time
import json
import copy
import re
from dataclasses import replace
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast
from nets.models import MCNCModel, MISNCModel
from env.MCEnv import MCEnv
from env.MISEnv import MISEnv
from utils.utils import print_epoch_header, load_test_data, generate_word, load_checkpoint_forgiving, set_random_seeds
from utils.env_utils import node_hamming_distance, edge_hamming_distance
from args.train_nc_args import get_args

try:
    import wandb
except ImportError:
    wandb = None

    
class Trainer:
    def __init__(self, params, run_path=None):
        # Set parameters
        self.params = params
        self.verbose = self.params['verbose']

        # Set device and dtype
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.params['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_dtype(torch.float32)

        if self.params['seed'] is not None:
            set_random_seeds(self.params['seed'])

        # Set environment and Model
        if self.params['problem'] == 'mc':
            Env = MCEnv
            self.nc_model = MCNCModel(**self.params).to(device)
        elif self.params['problem'] == 'mis':
            Env = MISEnv
            self.nc_model = MISNCModel(**self.params).to(device)
        else:
            raise ValueError(f"Invalid problem: {self.params['problem']}")

        # Training environment
        self.env = Env(self.params, device, testing=False)

        # Testing environment
        self.test_env = Env(self.params, device, testing=True)

        if self.verbose:
            print(f'Number of parameters in each model: {sum(p.numel() for p in self.nc_model.parameters() if p.requires_grad)}')

        # Optimizer
        self.optimizer = AdamW(self.nc_model.parameters(), lr=self.params['lr'], betas=self.params['betas'],
                               weight_decay=self.params['weight_decay'])

        # Restore model weights
        if self.params['nc_model_load_path'] != '':
            checkpoint = torch.load(self.params['nc_model_load_path'], map_location=self.device, weights_only=False)
            load_checkpoint_forgiving(self.nc_model, checkpoint['model_state_dict'], allow_partial=False)

            # Restore optimizer state if training
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                pass
            # Update the learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.params['lr']

        self.archive_model = None
        if self.params['cnc_archive_model_update_freq'] > 0:
            self._refresh_archive_model()

        # Compile
        if self.params['compile']:
            self.compiled_nc_model = torch.compile(self.nc_model)

        # Load test data
        self.test_graphs = load_test_data(self.params['problem'], self.params['eval_graph_types'], self.params['num_eval_graphs'])

        # Set exploration weights
        self.cNC_weights = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

        # Set exploration weight
        if self.params['nc_train_mode'] == 'exploitation':
            self.exploration_weight = 0.0
        elif self.params['nc_train_mode'] == 'exploration':
            self.exploration_weight = 1.0
        elif self.params['nc_train_mode'] == 'conditioned_network':
            self.exploration_weight = "-conditioned"
        elif self.params['nc_train_mode'] == 'exploration_exploitation':
            self.exploration_weight = self.params['exploration_weight']

        # Build the run-directory path
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if run_path is None:
            run_path = 'runs/train'
        run_name = timestamp
        if self.params.get('run_id'):
            safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.params['run_id']).strip("_")
            run_name = f"{timestamp}__{safe_run_id}" if safe_run_id else timestamp
        run_dir = Path(f"{run_path}") / f"{self.params['problem']}_nc" / f"{self.params['nc_train_mode']}" / f"{self.params['train_graph_type']}" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = run_dir
        # Dump params to JSON for reproducibility
        params_file = run_dir / "params.json"
        with params_file.open("w") as fp:
            json.dump(self.params, fp, indent=4)
        if self.verbose:
            print(f"Run directory: {self.run_path}")

        # Set wandb
        if self.params['wandb']:
            if wandb is None:
                raise ImportError("wandb is not installed. Install wandb or run with --no-wandb.")
            project_name = f'PB_NCO_NC_{self.params["problem"]}'
            wandb.login()
            wandb.init(
                project=project_name,
                name=self.params['run_id'] or None,
                config=self.params,
            )

    def train(self):
        start_time = time.time()
        for epoch in range(1, self.params['n_epochs'] + 1):
            # Containers for per-batch statistics
            batch_losses = []
            batch_avg_obj = []
            batch_avg_obj_R = []
            batch_avg_bl = []
            batch_avg_div = []
            batch_presample_obj = []
            batch_archive_obj = []
            batch_presample_loss = []
            batch_conditioned_loss = []
            batch_extra_metrics = []

            if self.verbose:
                print_epoch_header(epoch, self.params['n_epochs'], start_time)

            self.nc_model.train()

            for episode in range(1, self.params['n_episodes']+1):
                loss, obj_values, obj_rewards, diversity_rewards, baseline, extra_metrics = self.run_train_episode()

                # ==== collect per-batch stats ====
                batch_losses.append(loss)
                batch_avg_obj.append(obj_values)
                batch_avg_obj_R.append(obj_rewards)
                batch_avg_bl.append(baseline)
                batch_avg_div.append(diversity_rewards)
                if extra_metrics['presample_obj'] is not None:
                    batch_presample_obj.append(extra_metrics['presample_obj'])
                if extra_metrics['archive_obj'] is not None:
                    batch_archive_obj.append(extra_metrics['archive_obj'])
                if extra_metrics['presample_loss'] is not None:
                    batch_presample_loss.append(extra_metrics['presample_loss'])
                if extra_metrics['conditioned_loss'] is not None:
                    batch_conditioned_loss.append(extra_metrics['conditioned_loss'])
                batch_extra_metrics.append(extra_metrics)

            # ==== end of epoch: summarize ====
            self.env.cur_epoch += 1
            loss_mean = np.mean(batch_losses)
            loss_std = np.std(batch_losses)
            obj_mean = np.mean(batch_avg_obj)
            obj_R_mean = np.mean(batch_avg_obj_R)
            bl_mean = np.mean(batch_avg_bl)
            div_mean = np.mean(batch_avg_div)
            presample_obj_mean = np.mean(batch_presample_obj) if batch_presample_obj else None
            archive_obj_mean = np.mean(batch_archive_obj) if batch_archive_obj else None
            presample_loss_mean = np.mean(batch_presample_loss) if batch_presample_loss else None
            conditioned_loss_mean = np.mean(batch_conditioned_loss) if batch_conditioned_loss else None
            diagnostic_metrics = self._aggregate_metric_dicts(batch_extra_metrics)

            print(f"Loss:  mean={loss_mean:.6f}  std={loss_std:.4f} Obj Value:  mean={obj_mean:.3f}  Baseline: mean={bl_mean:.3f} "
                  f"Obj R: mean={obj_R_mean:.3f} Diversity R: mean={div_mean:.3f}\n")
            if presample_obj_mean is not None:
                print(f"Pre-sample Obj: mean={presample_obj_mean:.3f} Archive Obj: mean={archive_obj_mean:.3f} "
                      f"Pre-sample Loss: mean={presample_loss_mean if presample_loss_mean is not None else 0.0:.6f} "
                      f"Conditioned Loss: mean={conditioned_loss_mean if conditioned_loss_mean is not None else 0.0:.6f}\n")
            if diagnostic_metrics:
                print(
                    "Diagnostics: "
                    f"obj[p10/p50/p90]={diagnostic_metrics.get('objective/p10', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('objective/p50', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('objective/p90', 0.0):.3f} "
                    f"objR[p10/p50/p90]={diagnostic_metrics.get('obj_reward/p10', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('obj_reward/p50', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('obj_reward/p90', 0.0):.3f} "
                    f"div[p10/p50/p90]={diagnostic_metrics.get('diversity/p10', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('diversity/p50', 0.0):.3f}/"
                    f"{diagnostic_metrics.get('diversity/p90', 0.0):.3f} "
                    f"removals_mean={diagnostic_metrics.get('decoder/removals/mean', 0.0):.3f} "
                    f"conflicts_mean={diagnostic_metrics.get('decoder/conflicts/mean', 0.0):.3f}\n"
                )

            # Log test results to wandb
            if self.params['wandb']:
                metrics = {
                    "loss/mean": loss_mean,
                    "loss/std": loss_std,
                    "objective/mean": obj_mean,
                    "obj_reward/mean": obj_R_mean,
                    "baseline/mean": bl_mean,
                    "diversity/mean": div_mean,
                }
                if presample_obj_mean is not None:
                    metrics.update({
                        "presample/objective_mean": presample_obj_mean,
                        "presample/archive_objective_mean": archive_obj_mean,
                    })
                if presample_loss_mean is not None:
                    metrics["loss/presample_mean"] = presample_loss_mean
                if conditioned_loss_mean is not None:
                    metrics["loss/conditioned_mean"] = conditioned_loss_mean
                metrics.update(diagnostic_metrics)
                # log everything in one shot
                wandb.log(metrics, step=epoch)

            should_eval = ((epoch - 1) % self.params['eval_freq'] == 0) or (epoch == self.params['n_epochs'])
            if should_eval:
                if self.params['nc_train_mode'] == 'conditioned_network':
                    # Run conditioned network test
                    self.run_conditioned_network_test(epoch)

                self.run_test(epoch)

            # Save checkpoints
            if epoch % self.params['save_model_freq'] == 0 and self.params['save_model']:
                self.save_model(epoch)

            if self.params['cnc_archive_model_update_freq'] > 0 and epoch % self.params['cnc_archive_model_update_freq'] == 0:
                self._refresh_archive_model()

        # Save final model
        if self.params['save_model']:
            self.save_model(self.params['n_epochs'])

    @staticmethod
    def _append_tensor_metric(store, name, value, mode='full'):
        if value is None:
            return
        if mode == 'none':
            return
        if mode == 'summary':
            value = value.detach().float()
            stats = store.setdefault(name, {'sum': 0.0, 'count': 0})
            stats['sum'] += value.sum().item()
            stats['count'] += value.numel()
            return
        flat = value.detach().float().reshape(-1).cpu()
        if flat.numel() == 0:
            return
        store.setdefault(name, []).append(flat)

    @staticmethod
    def _summarize_tensor_metrics(store, mode='full'):
        metrics = {}
        if mode == 'none':
            return metrics
        for name, chunks in store.items():
            if mode == 'summary':
                count = chunks.get('count', 0)
                if count:
                    metrics[f'{name}/mean'] = chunks['sum'] / count
            else:
                if not chunks:
                    continue
                values = torch.cat(chunks)
                metrics[f'{name}/mean'] = values.mean().item()
                metrics[f'{name}/p10'] = torch.quantile(values, 0.10).item()
                metrics[f'{name}/p50'] = torch.quantile(values, 0.50).item()
                metrics[f'{name}/p90'] = torch.quantile(values, 0.90).item()
        return metrics

    @staticmethod
    def _aggregate_metric_dicts(metric_dicts):
        keys = sorted({k for d in metric_dicts for k, v in d.items() if isinstance(v, (int, float, np.floating))})
        return {
            key: float(np.mean([d[key] for d in metric_dicts if key in d and d[key] is not None]))
            for key in keys
        }

    def run_train_episode(self):
        """Run one training episode using PPO with leave-one-out baseline.

        This version collects several "restarts" of the environment to build a
        larger batch of sampled solutions. All gathered rollouts share the same
        policy parameters and are reused across multiple PPO epochs.
        """

        n_restarts = self.params.get('ppo_restarts', 10)
        ppo_epochs = self.params.get('ppo_epochs', 3)
        clip_eps = self.params.get('ppo_clip', 0.2)
        entropy_coef = self.params.get('entropy_coef', 0.0)
        kl_target = self.params.get('kl_target', 0.0) or 0.0

        if self.params['problem'] == 'mis' and self.params['feasibility_decoder'] != 'heatmap':
            raise NotImplementedError(
                "MIS PPO log-prob recomputation currently supports only the heatmap decoder. "
                "Sequential MIS decoding needs a trajectory-level action trace."
            )
        if (
            self.params['problem'] == 'mis'
            and self.params.get('mis_heatmap_post_add', False)
            and self.params.get('mis_heatmap_post_add_mode', 'greedy') != 'greedy'
        ):
            raise NotImplementedError(
                "MIS heatmap post-add training currently supports only greedy mode. "
                "Sampled post-add needs an additional PPO action trace."
            )

        ppo_records = []
        train_diagnostics = self.params.get('train_diagnostics', 'full')

        obj_value_sum = 0.0
        obj_reward_sum = 0.0
        diversity_sum = 0.0
        baseline_sum = 0.0
        presample_obj_sum = 0.0
        archive_obj_sum = 0.0
        presample_count = 0
        presample_loss_records = 0
        diagnostic_tensors = {}
        for _ in range(n_restarts):
            # Generate new batch for this restart
            state, _ = self.env.reset(self.params['batch_size'], 1, train_nc=True)

            with torch.no_grad():
                original_exploration_weight = state.exploration_weight
                if self.params['nc_train_mode'] != 'conditioned_network':
                    pass
                elif self.params['cnc_archive_source'] == 'model_presample':
                    presample_state = self._make_presample_state(state)
                    with autocast('cuda', enabled=self.params.get('amp', False)):
                        presample_logits = self._archive_model_forward(presample_state)

                    presample_solutions, presample_obj_values, presample_obj_reward, presample_diversity_rewards, presample_log_probs, presample_R_dict = self.env.heatmap_inference(
                        presample_logits,
                        feasibility_decoder=self.params['feasibility_decoder'],
                        visited_solutions=presample_state.visited_solutions,
                        n_rollouts=self.params['cnc_presample_rollouts'],
                        training=True,
                        greedy=False,
                        compute_diversity=False,
                    )
                    presample_diversity_rewards = self._compute_diversity_rewards(
                        presample_solutions,
                        presample_state.visited_solutions,
                        state.graph,
                        self.params['cnc_diversity_metric_train'],
                        rollout_info=presample_R_dict,
                        archive_probs=getattr(presample_state, 'archive_probs', None),
                    )
                    visited_solutions, archive_obj_mean, archive_probs = self._select_archive_from_presamples(
                        presample_solutions,
                        presample_obj_values,
                        state.graph,
                        presample_R_dict,
                    )
                    state = replace(
                        state,
                        visited_solutions=visited_solutions,
                        archive_probs=archive_probs,
                        exploration_weight=original_exploration_weight,
                    )
                    self.env.state = state

                    presample_obj_sum += presample_obj_values.mean().item()
                    archive_obj_sum += archive_obj_mean
                    presample_count += 1
                    if self.params['cnc_presample_loss_mode'] != 'none':
                        presample_reward = self._compute_presample_reward(
                            presample_obj_reward,
                            presample_diversity_rewards,
                            original_exploration_weight,
                        )
                        presample_advantage, presample_baseline = self._loo_advantage(presample_reward)
                        ppo_records.append({
                            'state': presample_state,
                            'solutions': presample_solutions,
                            'ppo_actions': presample_R_dict.get('ppo_actions', presample_solutions),
                            'old_log_probs': presample_log_probs.detach(),
                            'advantages': presample_advantage.detach(),
                            'n_rollouts': self.params['cnc_presample_rollouts'],
                            'role': 'presample',
                        })
                        baseline_sum += presample_baseline.mean().item()
                        presample_loss_records += 1
                        self._append_tensor_metric(diagnostic_tensors, 'presample_obj_reward', presample_obj_reward, train_diagnostics)
                elif self.params['cnc_archive_source'] == 'random':
                    pass
                else:
                    raise ValueError(f"Invalid cnc_archive_source: {self.params['cnc_archive_source']}")

                # Compute logits under the current policy
                with autocast('cuda', enabled=self.params.get('amp', False)):
                    logits = self._policy_forward(state)

                # Sample rollouts and obtain log-probs
                decode_kind = self.params['feasibility_decoder']
                solutions, obj_values, obj_reward, diversity_rewards, log_probs_mean, R_dict = self.env.heatmap_inference(
                    logits, feasibility_decoder=decode_kind, visited_solutions=state.visited_solutions, n_rollouts=self.params['n_rollouts'], training=True, greedy=False, compute_diversity=False,
                )
                diversity_rewards = self._compute_diversity_rewards(
                    solutions,
                    state.visited_solutions,
                    state.graph,
                    self.params['cnc_diversity_metric_train'],
                    rollout_info=R_dict,
                    archive_probs=getattr(state, 'archive_probs', None),
                )

                # Compute rewards and LOO baselines
                if self.params['nc_train_mode'] == 'exploitation':
                    reward = obj_reward
                elif self.params['nc_train_mode'] == 'exploration_exploitation':
                    reward = self.params['exploration_weight'] * diversity_rewards + (1 - self.params['exploration_weight']) * obj_reward
                elif self.params['nc_train_mode'] == 'exploration':
                    reward = diversity_rewards
                elif self.params['nc_train_mode'] == 'conditioned_network':
                    reward = state.exploration_weight.unsqueeze(-1) * diversity_rewards + (1 - state.exploration_weight.unsqueeze(-1)) * obj_reward
                else:
                    raise ValueError(f"Invalid nc_train_mode: {self.params['nc_train_mode']}")

                # Leave-One-out Baseline Computation
                advantage, baselines_loo = self._loo_advantage(reward)

                norm_adv = False
                if norm_adv:
                    # advantage shape: [B, K]
                    adv_mean = advantage.mean(dim=1, keepdim=True)
                    adv_std = advantage.std(dim=1, keepdim=True).clamp_min(1e-6)
                    advantage = (advantage - adv_mean) / adv_std

                ppo_records.append({
                    'state': state,
                    'solutions': solutions,
                    'ppo_actions': R_dict.get('ppo_actions', solutions),
                    'old_log_probs': log_probs_mean.detach(),
                    'advantages': advantage.detach(),
                    'n_rollouts': self.params['n_rollouts'],
                    'role': 'conditioned',
                })

                self._append_tensor_metric(diagnostic_tensors, 'objective', obj_values, train_diagnostics)
                self._append_tensor_metric(diagnostic_tensors, 'obj_reward', obj_reward, train_diagnostics)
                self._append_tensor_metric(diagnostic_tensors, 'diversity', diversity_rewards, train_diagnostics)
                self._append_tensor_metric(diagnostic_tensors, 'advantage', advantage, train_diagnostics)
                if 'all_num_of_removals' in R_dict:
                    self._append_tensor_metric(diagnostic_tensors, 'decoder/removals', R_dict['all_num_of_removals'], train_diagnostics)
                if 'all_num_of_conflicts' in R_dict:
                    self._append_tensor_metric(diagnostic_tensors, 'decoder/conflicts', R_dict['all_num_of_conflicts'], train_diagnostics)
                if 'ppo_actions' in R_dict:
                    self._append_tensor_metric(diagnostic_tensors, 'decoder/raw_selected', R_dict['ppo_actions'].sum(dim=-1).float(), train_diagnostics)
                if 'repaired_solutions' in R_dict:
                    self._append_tensor_metric(diagnostic_tensors, 'decoder/repaired_selected', R_dict['repaired_solutions'].sum(dim=-1).float(), train_diagnostics)
                if 'post_added' in R_dict:
                    self._append_tensor_metric(diagnostic_tensors, 'decoder/post_added', R_dict['post_added'], train_diagnostics)
                self._append_tensor_metric(diagnostic_tensors, 'decoder/final_selected', solutions.sum(dim=-1).float(), train_diagnostics)

            obj_value_sum += obj_values.mean().item() if obj_values is not None else 0.0
            obj_reward_sum += obj_reward.mean().item() if obj_reward is not None else 0.0
            diversity_sum += diversity_rewards.mean().item() if diversity_rewards is not None else 0.0
            baseline_sum += baselines_loo.mean().item() if baselines_loo is not None else 0.0

        total_loss = 0.0
        presample_loss_sum = 0.0
        presample_loss_count = 0
        conditioned_loss_sum = 0.0
        conditioned_loss_count = 0
        n_updates = 0
        kl_stopped = False
        kl_stop_ppo_epoch = 0
        mean_kl_last = 0.0
        for ppo_epoch in range(ppo_epochs):
            if kl_stopped:
                break

            epoch_loss = 0.0
            batch_count = 0
            epoch_kl_sum = 0.0
            epoch_kl_count = 0
            for r, record in enumerate(ppo_records):
                batch_count += 1
                state = record['state']
                with autocast('cuda', enabled=self.params.get('amp', False)):
                    new_logits = self._policy_forward(state)

                new_log_p = F.log_softmax(new_logits, dim=-1)
                expanded_log_p = new_log_p.unsqueeze(1).expand(-1, record['n_rollouts'], -1, -1)

                if self.params['problem'] == 'mc':
                    gathered = expanded_log_p[:, :, 1:].gather(
                        3,
                        record['solutions'][:, :, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    gathered = expanded_log_p.gather(
                        3,
                        record['ppo_actions'].unsqueeze(-1)
                    ).squeeze(-1)
                if self.params.get('ppo_logprob_reduction', 'mean') == 'sum':
                    new_log_probs = gathered.sum(dim=2)
                else:
                    new_log_probs = gathered.mean(dim=2)

                with torch.no_grad():
                    approx_kl = (record['old_log_probs'] - new_log_probs).mean().item()
                epoch_kl_sum += approx_kl
                epoch_kl_count += 1

                ratio = torch.exp(new_log_probs - record['old_log_probs'])
                surrogate1 = ratio * record['advantages']
                surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * record['advantages']
                loss_r = -torch.min(surrogate1, surrogate2).mean()

                probs = new_log_p.exp()
                entropy = -(new_log_p * probs).sum(dim=-1).mean()
                loss_r = loss_r - entropy_coef * entropy
                if record['role'] == 'presample':
                    loss_r = self.params['cnc_presample_loss_coef'] * loss_r
                    presample_loss_sum += loss_r.detach().item()
                    presample_loss_count += 1
                else:
                    conditioned_loss_sum += loss_r.detach().item()
                    conditioned_loss_count += 1

                epoch_loss += loss_r

                if (batch_count >= self.params['ppo_update_batch_count']) or (r == len(ppo_records) - 1):
                    epoch_loss = epoch_loss / batch_count
                    self.optimizer.zero_grad()
                    epoch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nc_model.parameters(), self.params['max_grad_norm'])
                    self.optimizer.step()

                    total_loss += epoch_loss.item()
                    batch_count = 0
                    epoch_loss = 0.0
                    n_updates += 1

            if epoch_kl_count > 0:
                mean_kl_last = epoch_kl_sum / epoch_kl_count
            if kl_target > 0 and epoch_kl_count > 0:
                if mean_kl_last > kl_target:
                    kl_stopped = True
                    kl_stop_ppo_epoch = ppo_epoch + 1
                    if self.verbose:
                        print(
                            f"  [KL early stop] mean KL {mean_kl_last:.4f} > target {kl_target:.4f} "
                            f"after PPO epoch {ppo_epoch + 1}/{ppo_epochs}"
                        )

        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0

        avg_obj_values = obj_value_sum / n_restarts
        avg_obj_reward = obj_reward_sum / n_restarts
        avg_diversity = diversity_sum / n_restarts
        avg_baseline = baseline_sum / (n_restarts + presample_loss_records)
        extra_metrics = {
            'presample_obj': presample_obj_sum / presample_count if presample_count else None,
            'archive_obj': archive_obj_sum / presample_count if presample_count else None,
            'presample_loss': presample_loss_sum / presample_loss_count if presample_loss_count else None,
            'conditioned_loss': conditioned_loss_sum / conditioned_loss_count if conditioned_loss_count else None,
            'kl/mean': mean_kl_last,
            'kl_early_stopped': float(kl_stopped),
            'kl/stop_ppo_epoch': float(kl_stop_ppo_epoch),
        }
        for i in range(1, ppo_epochs + 1):
            extra_metrics[f'kl/stopped_after_ppo_epoch_{i}'] = float(kl_stop_ppo_epoch == i)
        extra_metrics.update(self._summarize_tensor_metrics(diagnostic_tensors, train_diagnostics))

        return avg_loss, avg_obj_values, avg_obj_reward, avg_diversity, avg_baseline, extra_metrics

    def _refresh_archive_model(self):
        self.archive_model = copy.deepcopy(self.nc_model)
        self.archive_model.eval()
        for p in self.archive_model.parameters():
            p.requires_grad_(False)

    def _policy_forward(self, state):
        return self.compiled_nc_model(state) if self.params['compile'] else self.nc_model(state)

    def _archive_model_forward(self, state):
        if self.archive_model is not None:
            return self.archive_model(state)
        return self._policy_forward(state)

    def _make_presample_state(self, state):
        if self.params['cnc_presample_context'] == 'zero_archive':
            visited_solutions = torch.zeros(
                state.batch_size,
                state.problem_size,
                self.params['n_visited_solutions'],
                dtype=torch.float32,
                device=self.device,
            )
        elif self.params['cnc_presample_context'] == 'random_archive':
            visited_solutions = torch.randint(
                0,
                2,
                (state.batch_size, state.problem_size, self.params['n_visited_solutions']),
                device=self.device,
            ).float()
        else:
            raise ValueError(f"Invalid cnc_presample_context: {self.params['cnc_presample_context']}")

        exploration_weight = torch.zeros(state.batch_size, dtype=torch.float32, device=self.device)
        greedy_inference_hint = None
        if self.params.get('cnc_presample_greedy_feature', False):
            greedy_inference_hint = torch.ones(state.batch_size, dtype=torch.float32, device=self.device)
        return replace(
            state,
            visited_solutions=visited_solutions,
            exploration_weight=exploration_weight,
            greedy_inference_hint=greedy_inference_hint,
        )

    def _select_archive_from_presamples(self, solutions, obj_values, graph, rollout_info=None):
        k = self.params['n_visited_solutions']
        idx = self._select_archive_indices(solutions, obj_values, graph, k, self.params['cnc_archive_select'])
        archive_obj_mean = obj_values.gather(1, idx).mean().item()

        if self.params.get('cnc_archive_solution_source', 'repaired') == 'raw':
            if rollout_info is None or 'ppo_actions' not in rollout_info:
                raise ValueError("Raw archive source requires heatmap ppo_actions.")
            archive_candidates = rollout_info['ppo_actions'].float()
        else:
            archive_candidates = solutions.float()

        archive = self._gather_archive_candidates(archive_candidates, idx)
        archive_probs = None
        if rollout_info is not None and 'select_probs' in rollout_info:
            archive_probs = self._gather_archive_candidates(rollout_info['select_probs'].float(), idx)
        return archive, archive_obj_mean, archive_probs

    def _select_archive_from_candidates(self, solutions, obj_values, graph, k, select_mode):
        idx = self._select_archive_indices(solutions, obj_values, graph, k, select_mode)
        archive_obj_mean = obj_values.gather(1, idx).mean().item()
        selected = self._gather_archive_candidates(solutions.float(), idx)
        return selected, archive_obj_mean

    def _select_archive_indices(self, solutions, obj_values, graph, k, select_mode):
        if solutions.size(1) < k:
            repeats = (k + solutions.size(1) - 1) // solutions.size(1)
            solutions = solutions.repeat(1, repeats, 1)
            obj_values = obj_values.repeat(1, repeats)

        if select_mode == 'best_objective':
            idx = obj_values.topk(k, dim=1).indices
        elif select_mode == 'random_samples':
            rand_scores = torch.rand_like(obj_values.float())
            idx = rand_scores.topk(k, dim=1).indices
        elif select_mode == 'diverse_best':
            idx = self._select_diverse_best_indices(solutions, obj_values, graph, k)
        else:
            raise ValueError(f"Invalid archive selection mode: {select_mode}")

        return idx

    @staticmethod
    def _gather_archive_candidates(candidates, idx):
        if candidates.size(1) < idx.size(1):
            repeats = (idx.size(1) + candidates.size(1) - 1) // candidates.size(1)
            candidates = candidates.repeat(1, repeats, 1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, candidates.size(-1))
        selected = candidates.gather(1, gather_idx).float()
        return selected.permute(0, 2, 1).contiguous()

    def _select_diverse_best_indices(self, solutions, obj_values, graph, k):
        batch_size, n_candidates, _ = solutions.size()
        selected = torch.empty(batch_size, k, dtype=torch.long, device=self.device)
        sorted_idx = obj_values.argsort(dim=1, descending=True)

        for b in range(batch_size):
            chosen = [int(sorted_idx[b, 0].item())]
            while len(chosen) < k:
                best_score = None
                best_idx = None
                chosen_archive = solutions[b, chosen].T.unsqueeze(0).float()
                for candidate in sorted_idx[b]:
                    candidate_idx = int(candidate.item())
                    if candidate_idx in chosen:
                        continue
                    candidate_solution = solutions[b, candidate_idx].unsqueeze(0)
                    diversity = self.env.distance_fn(candidate_solution, chosen_archive, graph[b].unsqueeze(0))[0]
                    score = obj_values[b, candidate_idx].float() + diversity
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = candidate_idx
                chosen.append(best_idx)
            selected[b] = torch.tensor(chosen, dtype=torch.long, device=self.device)

        return selected

    def _compute_diversity_rewards(self, solutions, visited_solutions, graph, metric, rollout_info=None, archive_probs=None):
        if visited_solutions is None:
            return None
        if metric == 'configured':
            metric = self.params['distance_metric']

        source = self.params.get('cnc_diversity_solution_source', 'repaired')
        if source == 'raw':
            if rollout_info is None or 'ppo_actions' not in rollout_info:
                raise ValueError("Raw diversity source requires heatmap ppo_actions.")
            diversity_solutions = rollout_info['ppo_actions']
        else:
            diversity_solutions = solutions

        if source == 'prob':
            if rollout_info is None or 'ppo_actions' not in rollout_info:
                raise ValueError("Probability diversity source requires heatmap ppo_actions.")
            prob_archive = archive_probs if archive_probs is not None else visited_solutions.float()
            rewards = []
            for r in range(rollout_info['ppo_actions'].size(1)):
                cur_raw = rollout_info['ppo_actions'][:, r, :]
                prob_node_reward = self._prob_node_hamming_distance(cur_raw, prob_archive)
                if metric == 'mixed_node_edge':
                    alpha = self.params['cnc_mixed_diversity_alpha']
                    edge_reward = edge_hamming_distance(cur_raw, visited_solutions, graph)
                    cur_reward = alpha * prob_node_reward + (1.0 - alpha) * edge_reward
                else:
                    cur_reward = prob_node_reward
                rewards.append(cur_reward)
            return torch.stack(rewards, dim=1)

        rewards = []
        for r in range(diversity_solutions.size(1)):
            cur_solution = diversity_solutions[:, r, :]
            if metric == 'node_hamming':
                cur_reward = node_hamming_distance(cur_solution, visited_solutions, graph)
            elif metric == 'edge_hamming':
                cur_reward = edge_hamming_distance(cur_solution, visited_solutions, graph)
            elif metric == 'mixed_node_edge':
                alpha = self.params['cnc_mixed_diversity_alpha']
                node_reward = node_hamming_distance(cur_solution, visited_solutions, graph)
                edge_reward = edge_hamming_distance(cur_solution, visited_solutions, graph)
                cur_reward = alpha * node_reward + (1.0 - alpha) * edge_reward
            else:
                raise ValueError(f"Invalid diversity metric: {metric}")
            rewards.append(cur_reward)
        return torch.stack(rewards, dim=1)

    @staticmethod
    def _prob_node_hamming_distance(solution, archive_probs):
        s01 = (solution > 0).float()
        probs = archive_probs.float().clamp(0.0, 1.0).permute(0, 2, 1).contiguous()
        expected_distance = torch.where(
            s01.unsqueeze(1) > 0.5,
            1.0 - probs,
            probs,
        )
        return expected_distance.mean(dim=(-1, -2))

    def _compute_presample_reward(self, obj_reward, diversity_rewards, exploration_weight):
        if self.params['cnc_presample_loss_mode'] == 'objective_ppo':
            return obj_reward
        if self.params['cnc_presample_loss_mode'] == 'mixed_reward':
            if diversity_rewards is None:
                return obj_reward
            return exploration_weight.unsqueeze(-1) * diversity_rewards + (1 - exploration_weight.unsqueeze(-1)) * obj_reward
        raise ValueError(f"Invalid cnc_presample_loss_mode: {self.params['cnc_presample_loss_mode']}")

    def _loo_advantage(self, reward):
        n_rollouts = reward.size(1)
        if n_rollouts > 1:
            sum_r = reward.sum(dim=1, keepdim=True)
            baselines_loo = (sum_r - reward) / (n_rollouts - 1)
            advantage = reward - baselines_loo
        else:
            advantage = reward
            baselines_loo = reward
        return advantage, baselines_loo

    def _make_eval_archive(self, state, graph_type, graph_idx, source):
        k = self.params['n_visited_solutions']
        if source == 'fixed_random':
            return self._make_deterministic_random_archive(state, graph_type, graph_idx, k)
        if source == 'best_of_random':
            n_pool = self.params['cnc_eval_presample_rollouts']
            generator = self._make_eval_generator(graph_type, graph_idx)
            candidates = torch.randint(
                0,
                2,
                (state.batch_size, n_pool, state.problem_size),
                device=self.device,
                generator=generator,
            )
            if self.params['problem'] == 'mc':
                candidates[:, :, 0] = 1
            obj_values = self._compute_constructive_obj_values(candidates, state.graph)
            archive, _ = self._select_archive_from_candidates(candidates, obj_values, state.graph, k, 'best_objective')
            return archive
        if source == 'self_presample':
            presample_state = replace(
                state,
                visited_solutions=torch.zeros(state.batch_size, state.problem_size, k, dtype=torch.float32, device=self.device),
                exploration_weight=torch.zeros(state.batch_size, dtype=torch.float32, device=self.device),
            )
            logits = self._policy_forward(presample_state)
            candidates, obj_values, _, _, _, _ = self.test_env.heatmap_inference(
                logits,
                feasibility_decoder=self.params['feasibility_decoder'],
                visited_solutions=presample_state.visited_solutions,
                n_rollouts=self.params['cnc_eval_presample_rollouts'],
                training=True,
                greedy=False,
            )
            archive, _ = self._select_archive_from_candidates(candidates, obj_values, state.graph, k, 'best_objective')
            return archive
        raise ValueError(f"Invalid eval archive source: {source}")

    def _make_deterministic_random_archive(self, state, graph_type, graph_idx, k):
        generator = self._make_eval_generator(graph_type, graph_idx)
        archive = torch.randint(
            0,
            2,
            (state.batch_size, state.problem_size, k),
            device=self.device,
            generator=generator,
        ).float()
        if self.params['problem'] == 'mc':
            archive[:, 0, :] = 1.0
        return archive

    def _make_eval_generator(self, graph_type, graph_idx):
        seed = self.params['cnc_eval_archive_seed'] + 1009 * graph_idx + 9176 * sum(ord(c) for c in graph_type)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def _compute_mc_obj_values(self, solutions, graph):
        ising_solutions = 2 * solutions - 1
        outer_solutions = ising_solutions.unsqueeze(-1) * ising_solutions.unsqueeze(-2)
        diff_matrix = 1.0 - outer_solutions
        product = diff_matrix * graph.unsqueeze(1)
        return 0.25 * product.sum(dim=(-1, -2))

    def _compute_constructive_obj_values(self, solutions, graph):
        if self.params['problem'] == 'mc':
            return self._compute_mc_obj_values(solutions, graph)
        if self.params['problem'] == 'mis':
            selected = solutions.float()
            conflicts = 0.5 * (selected @ graph.float() * selected).sum(dim=(-1, -2))
            return selected.sum(dim=-1) - self.params['punish_w'] * conflicts
        raise ValueError(f"Invalid problem: {self.params['problem']}")

    @staticmethod
    def _mean_item(value):
        if value is None:
            return 0.0
        if torch.is_tensor(value):
            return value.float().mean().item()
        return float(value)

    @staticmethod
    def _mean_instance_max(value):
        if value is None:
            return 0.0
        value = value.float()
        if value.dim() >= 2:
            return value.max(dim=1).values.mean().item()
        return value.mean().item()

    def _mis_eval_summary(self, obj_values, diversity_rewards, R_dict, certainty, total_time):
        raw_solutions = R_dict.get('raw_solutions', R_dict.get('ppo_actions'))
        repaired_solutions = R_dict.get('repaired_solutions')
        final_solutions = R_dict.get('final_solutions')

        if final_solutions is None:
            final_selected = obj_values.float()
        else:
            final_selected = final_solutions.sum(dim=-1).float()
        if repaired_solutions is None:
            repaired_selected = final_selected
        else:
            repaired_selected = repaired_solutions.sum(dim=-1).float()
        if raw_solutions is None:
            raw_selected = repaired_selected
        else:
            raw_selected = raw_solutions.sum(dim=-1).float()

        conflicts = R_dict.get('all_num_of_conflicts')
        removals = R_dict.get('all_num_of_removals')
        post_added = R_dict.get('post_added')
        if post_added is None:
            post_added = final_selected - repaired_selected

        return {
            "best_max_obj_value": self._mean_instance_max(obj_values),
            "best_avg_obj_value": self._mean_item(obj_values.mean(dim=1) if obj_values.dim() >= 2 else obj_values),
            "diversity": self._mean_item(diversity_rewards),
            "certainty": self._mean_item(certainty),
            "time": float(total_time),
            "raw_obj_mean": self._mean_item(raw_selected),
            "raw_obj_max": self._mean_instance_max(raw_selected),
            "repaired_obj_mean": self._mean_item(repaired_selected),
            "repaired_obj_max": self._mean_instance_max(repaired_selected),
            "final_obj_mean": self._mean_item(final_selected),
            "final_obj_max": self._mean_instance_max(final_selected),
            "raw_selected_mean": self._mean_item(raw_selected),
            "repaired_selected_mean": self._mean_item(repaired_selected),
            "final_selected_mean": self._mean_item(final_selected),
            "conflicts_mean": self._mean_item(conflicts),
            "removals_mean": self._mean_item(removals),
            "post_added_mean": self._mean_item(post_added),
        }

    @staticmethod
    def _add_metric_totals(totals, metrics):
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)

    @staticmethod
    def _average_metric_totals(totals, count):
        if count <= 0:
            return {}
        return {key: value / count for key, value in totals.items()}

    def save_model(self, epoch):
        path = self.run_path / f"epoch{epoch}.pth"
        torch.save({
            'model_state_dict': self.nc_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.params,
        }, path)

    @torch.no_grad()
    def run_test(self, epoch):
        # Set model to evaluation mode
        self.nc_model.eval()
        print(f"=== Starting test epoch {epoch} ===")
        # For each instance type
        for i, (cur_test_graph_list, graph_type) in enumerate(zip(self.test_graphs, self.params['eval_graph_types'])):
            mean_max_vals = 0
            mean_avg_vals = 0
            diversity_vals = 0
            certainty_vals = 0
            all_times = 0
            R_dict_all = {}
            metric_totals = {}
            for j, cur_test_graph in enumerate(cur_test_graph_list):
                if len(cur_test_graph.size()) == 2:  # Different sizes
                    problem_size, _ = cur_test_graph.size()
                    test_batch_size = 1
                    cur_test_graph = cur_test_graph.unsqueeze(0)  # batch size = 1
                else:
                    test_batch_size, problem_size, _ = cur_test_graph.size()

                # Reset environment
                state, done = self.test_env.reset(test_batch_size, 1, test_graph=cur_test_graph.to(self.device), train_nc=True)  # Generate new batch
                state.exploration_weight = torch.zeros(test_batch_size, device=self.device)
                state = replace(
                    state,
                    visited_solutions=self._make_eval_archive(state, graph_type, j, self.params['cnc_eval_archive_source']),
                )
                self.test_env.state = state

                # Compute logits
                start_time = time.time()
                with autocast('cuda', enabled=self.params.get('amp', False)):
                    logits = self._policy_forward(state)

                # Perform the rollouts
                if self.params['problem'] == 'mis':
                    _, obj_values, _, diversity_rewards, _, R_dict = self.test_env.heatmap_inference(
                        logits,
                        feasibility_decoder=self.params['feasibility_decoder'],
                        visited_solutions=state.visited_solutions,
                        n_rollouts=self.params['n_rollouts'],
                        training=True,
                        greedy=False,
                    )
                else:
                    _, obj_values, _, diversity_rewards, _, R_dict = self.test_env.heatmap_inference(
                        logits, feasibility_decoder='sequential', visited_solutions=state.visited_solutions, n_rollouts=1,
                        training=True, greedy=True
                    )

                probs = F.softmax(logits, dim=-1)
                node_entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [B, N]
                entropy = node_entropy.mean(dim=1)  # [B]
                # certainty C_t = 1 - H_t / log(2)
                certainty = 1 - entropy / torch.log(torch.tensor(2.0, device=self.device))

                total_time = time.time() - start_time

                if self.params['problem'] == 'mis':
                    graph_metrics = self._mis_eval_summary(obj_values, diversity_rewards, R_dict, certainty, total_time)
                    self._add_metric_totals(metric_totals, graph_metrics)
                    print(
                        f"[Graph #{j + 1} | size={problem_size}] "
                        f"Final avg/max: {graph_metrics['final_obj_mean']:.3f}/{graph_metrics['final_obj_max']:.3f}, "
                        f"Raw avg: {graph_metrics['raw_obj_mean']:.3f}, "
                        f"Repaired avg: {graph_metrics['repaired_obj_mean']:.3f}, "
                        f"Conflicts: {graph_metrics['conflicts_mean']:.3f}, "
                        f"Removals: {graph_metrics['removals_mean']:.3f}, "
                        f"Post-add: {graph_metrics['post_added_mean']:.3f}, "
                        f"Diversity: {graph_metrics['diversity']:.3f}, "
                        f"Certainty: {graph_metrics['certainty']:.3f}, "
                        f"Time: {total_time:.2f}s"
                    )
                else:
                    # Compute the rewards and gaps
                    avg_obj_value = obj_values.mean(dim=1)
                    max_obj_value = obj_values.max(dim=1).values
                    mean_max = max_obj_value.mean().item()
                    mean_avg = avg_obj_value.mean().item()
                    certainty = certainty.mean().item()
                    diversity = diversity_rewards.mean().item() if diversity_rewards is not None else 0.0

                    # record
                    mean_max_vals += mean_max
                    mean_avg_vals += mean_avg
                    diversity_vals += diversity
                    certainty_vals += certainty
                    all_times += total_time
                    # Additional records
                    for key, value in R_dict.items():
                        mean_val = value.mean().item()
                        R_dict_all[key] = R_dict_all.get(key, 0.0) + mean_val

                    print(
                        f"[Graph #{i + 1} | size={problem_size}] "
                        f"Diversity: {diversity:.3f}, "
                        f"Certainty: {certainty:.3f}, "
                        f"Time: {total_time:.2f}s"
                    )

            # Log test results to wandb
            if self.params['wandb']:
                n_div = len(cur_test_graph_list)
                if self.params['problem'] == 'mis':
                    avg_metrics = self._average_metric_totals(metric_totals, n_div)
                    wandb.log({f"test_{graph_type}/{key}": value for key, value in avg_metrics.items()}, step=epoch)
                    continue
                mean_max_vals /= n_div
                mean_avg_vals /= n_div
                diversity_vals /= n_div
                certainty_vals /= n_div
                all_times /= n_div

                avg_R = {k: v / n_div for k, v in R_dict_all.items()}

                base_metrics = {
                    f"test_{graph_type}/best_max_obj_value": mean_max_vals,
                    f"test_{graph_type}/best_avg_obj_value": mean_avg_vals,
                    f"test_{graph_type}/diversity": diversity_vals,
                    f"test_{graph_type}/certainty": certainty_vals,
                    f"test_{graph_type}/time": all_times,
                }
                R_metrics = {
                    f"test_{graph_type}/{key}": value
                    for key, value in avg_R.items()
                }

                # merge and log once
                metrics = {**base_metrics, **R_metrics}
                # call wandb.log once per graph
                wandb.log(metrics, step=epoch)

        print(f"=== Finished test epoch {epoch} ===\n")

    @torch.no_grad()
    def run_conditioned_network_test(self, epoch):
        # Set model to evaluation mode
        self.nc_model.eval()
        print(f"=== Starting conditioned-network test epoch {epoch} ===")

        n_weights = len(self.cNC_weights)
        # For each instance type
        for type_idx, (cur_test_graph_list, graph_type) in enumerate(zip(self.test_graphs, self.params['eval_graph_types'])):
            mean_max_vals = np.zeros(n_weights)
            mean_avg_vals = np.zeros(n_weights)
            diversity_vals = np.zeros(n_weights)
            certainty_vals = np.zeros(n_weights)
            metric_totals_by_weight = [dict() for _ in range(n_weights)]
            for i, cur_test_graph in enumerate(cur_test_graph_list):
                if len(cur_test_graph.size()) == 2:  # Different sizes
                    problem_size, _ = cur_test_graph.size()
                    test_batch_size = 1
                    cur_test_graph = cur_test_graph.unsqueeze(0)  # batch size = 1
                else:
                    test_batch_size, problem_size, _ = cur_test_graph.size()

                # Reset environment
                state, done = self.test_env.reset(test_batch_size, 1, test_graph=cur_test_graph.to(self.device), train_nc=True)  # Generate new batch
                state = replace(
                    state,
                    visited_solutions=self._make_eval_archive(state, graph_type, i, self.params['cnc_eval_archive_source']),
                )
                self.test_env.state = state

                for w_idx, w in enumerate(self.cNC_weights):
                    start_time = time.time()

                    # Set exploration weight for conditioned network
                    state.exploration_weight = torch.ones(test_batch_size, device=self.device) * w

                    # Compute logits
                    with autocast('cuda', enabled=self.params.get('amp', False)):
                        logits = self._policy_forward(state)

                    # Perform the rollouts
                    if self.params['problem'] == 'mis':
                        _, obj_values, _, diversity_rewards, _, R_dict = self.test_env.heatmap_inference(
                            logits,
                            feasibility_decoder=self.params['feasibility_decoder'],
                            visited_solutions=state.visited_solutions,
                            n_rollouts=self.params['n_rollouts'],
                            training=True,
                            greedy=False,
                        )
                    else:
                        _, obj_values, _, diversity_rewards, _, R_dict = self.test_env.heatmap_inference(
                            logits, feasibility_decoder='sequential', visited_solutions=state.visited_solutions, n_rollouts=1,
                            training=True, greedy=True
                        )

                    probs = F.softmax(logits, dim=-1)
                    node_entropy = -(probs * torch.log(probs)).sum(dim=-1)  # [B, N]
                    entropy = node_entropy.mean(dim=1)  # [B]
                    # certainty C_t = 1 - H_t / log(2)
                    certainty = 1 - entropy / torch.log(torch.tensor(2.0, device=self.device))
                    total_time = time.time() - start_time

                    if self.params['problem'] == 'mis':
                        graph_metrics = self._mis_eval_summary(obj_values, diversity_rewards, R_dict, certainty, total_time)
                        self._add_metric_totals(metric_totals_by_weight[w_idx], graph_metrics)
                        print(
                            f"[Graph {i} | {graph_type} | w={w}] "
                            f"Final avg/max: {graph_metrics['final_obj_mean']:.3f}/{graph_metrics['final_obj_max']:.3f}, "
                            f"Raw avg: {graph_metrics['raw_obj_mean']:.3f}, "
                            f"Repaired avg: {graph_metrics['repaired_obj_mean']:.3f}, "
                            f"Conflicts: {graph_metrics['conflicts_mean']:.3f}, "
                            f"Removals: {graph_metrics['removals_mean']:.3f}, "
                            f"Post-add: {graph_metrics['post_added_mean']:.3f}, "
                            f"Diversity: {graph_metrics['diversity']:.3f}, "
                            f"Certainty: {graph_metrics['certainty']:.3f}, "
                            f"Time: {total_time:.2f}s"
                        )
                    else:
                        # Compute metrics
                        avg_obj_value = obj_values.mean(dim=1)
                        max_obj_value = obj_values.max(dim=1).values
                        mean_max = max_obj_value.mean().item()
                        mean_avg = avg_obj_value.mean().item()
                        diversity = diversity_rewards.mean().item() if diversity_rewards is not None else 0.0
                        certainty_scalar = certainty.mean().item()

                        # record
                        mean_max_vals[w_idx] += mean_max
                        mean_avg_vals[w_idx] += mean_avg
                        diversity_vals[w_idx] += diversity
                        certainty_vals[w_idx] += certainty_scalar

                        print(
                            f"[Graph {i} | {graph_type} | w={w}] "
                            f"Avg. val.: {mean_avg:.3f}, "
                            f"Max val.: {mean_max:.3f}, "
                            f"Diversity: {diversity:.3f}, "
                            f"Certainty: {certainty_scalar:.3f}, "
                            f"Time: {total_time:.2f}s"
                        )
                print("")

            if self.params['wandb'] and self.params['problem'] == 'mis':
                rows = []
                scalar_metrics = {}
                for w_idx, w in enumerate(self.cNC_weights):
                    avg_metrics = self._average_metric_totals(metric_totals_by_weight[w_idx], len(cur_test_graph_list))
                    weight_key = f"{float(w):.3g}"
                    rows.append({
                        "epoch": epoch,
                        "graph_type": graph_type,
                        "weight": float(w),
                        **avg_metrics,
                    })
                    for key, value in avg_metrics.items():
                        scalar_metrics[f"conditioned_{graph_type}/w_{weight_key}/{key}"] = value

                if rows:
                    table = wandb.Table(columns=list(rows[0].keys()))
                    for row in rows:
                        table.add_data(*[row[col] for col in rows[0].keys()])
                    scalar_metrics[f"conditioned_{graph_type}/table"] = table
                wandb.log(scalar_metrics, step=epoch)

        print(f"=== Finished conditioned-network test epoch {epoch} ===\n")


def main():
    # Get args
    params = get_args()

    if params['debug']:
        params['verbose'] = True
        params['wandb'] = False
        params['save_model'] = False
        params['n_epochs'] = 5
        params['n_episodes'] = 5
        params['min_problem_size'] = 20
        params['max_problem_size'] = 100
        params['eval_graph_types'] = ['ER20']
        params['num_eval_graphs'] = [100]

    if params['distance_metric'] == 'default':
        if params['problem'] == 'mc':
            params['distance_metric'] = 'edge_hamming'
        elif params['problem'] == 'mis':
            params['distance_metric'] = 'node_hamming'

    execution_name = generate_word(6)
    params['execution_name'] = execution_name

    if params['verbose']:
        print(f'Execution name: {execution_name}')
        print(params)

    trainer = Trainer(params)
    trainer.train()


if __name__ == "__main__":
    main()
