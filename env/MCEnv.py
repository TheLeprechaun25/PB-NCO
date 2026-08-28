import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from env.memory import select_memory
from env.generators import ERGraphGenerator, RBGraphGenerator
from utils.env_utils import State, distance_fn


class MCEnv:
    def __init__(self, params, device, testing, compute_metrics=True, initializer=None):
        self.params = params
        self.initializer = initializer

        # Env Params
        self.testing = testing
        self.compute_metrics = compute_metrics
        self.problem_size = None
        self.batch_size = None
        self.pop_size = None
        self.batch_pop_range = None
        self.patience = params.get('test_patience' if testing else 'train_patience', None)
        self.max_iterations = None
        self.initialization = params.get('initialization', 'random')
        self.distance_fn = distance_fn(params['distance_metric'])

        self.device = device
        self.float_dtype = torch.float32
        self.int_dtype = torch.int32

        self.iteration = 0
        self.cur_epoch = 0
        self.max_epochs = params.get('n_epochs')

        # Env Modules
        if not testing:
            train_graph_type = params.get('train_graph_type')
            if train_graph_type == 'ER':
                conn_low = params.get('conn_low', 0.15)
                conn_high = params.get('conn_high', 0.15)
                self.generator = ERGraphGenerator(min_n=params['min_problem_size'], max_n=params['max_problem_size'], cl=self.params.get('curriculum_learning', False),
                                                  max_size_exec_percent=params.get('max_size_exec_percent', 0.8), p_connection_low=conn_low, p_connection_high=conn_high, max_epochs=self.max_epochs)

            elif train_graph_type == 'RB':
                if params['min_problem_size'] == 200 and params['max_problem_size'] == 300:
                    graph_sizes = 'small'
                elif params['min_problem_size'] == 800 and params['max_problem_size'] == 1200:
                    graph_sizes = 'large'
                elif params['min_problem_size'] == 200 and params['max_problem_size'] == 1200:
                    graph_sizes = 'mixed'
                else:
                    raise NotImplementedError

                self.generator = RBGraphGenerator(graph_sizes=graph_sizes)
            else:
                raise NotImplementedError
        else:
            self.generator = None

        self.memory = None
        self.memory_type = params.get('memory_type', 'none')
        self.state = None

        # Env Records
        self.objective_values = None
        self.reward = None
        self.best_objective_values = None
        self.best_solutions = None
        self.cur_best_objective_values = None
        self.cur_best_solutions = None
        self.non_improving_steps = None
        self.last_ms_initializations = None

        self.baseline = None  # For normalizing the NC reward
        self.boost = None  # For normalizing the NC reward

    def reset(self, batch_size, pop_size, problem_size=None, test_graph=None, train_nc=False, seed=None):
        """
        :param batch_size: (int) Num of instances in each batch (for training)
        :param pop_size: (int) Num of threads for each instance
        :param problem_size: (int) Problem size (number of nodes) (used to fix size - in gpu_utils to check max size)
        :param test_graph: (torch.Tensor) Graph to be used for testing. Shape: (n, n)
        :param train_nc: (bool) Whether to train the neural constructive model
        :param seed: (int) Random seed
        """
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.batch_pop_range = torch.arange(batch_size * pop_size).to(self.device)

        # Generate a batch of graphs if test_graph is None (training)
        if test_graph is None:  # Training, generate batch of graphs
            # pull current bounds (these may have been shrunk after an OOM)
            if hasattr(self.generator, "set_bounds"):
                self.generator.set_bounds(self.params['min_problem_size'], self.params['max_problem_size'])

            adj_matrix = self.generator.generate_graphs(batch_size, problem_size, cur_epoch=self.cur_epoch)
            adj_matrix = torch.from_numpy(adj_matrix).int().to(self.device)

        else:  # Testing, use the given test graph
            assert self.testing
            adj_matrix = test_graph.clone().int().to(self.device)

        self.problem_size = adj_matrix.shape[1]  # Assuming adj_matrix is of shape (batch_size, n, n)

        # Compute the norm factors (baseline and upper bound) given by the expected obj value of each graph.
        n_edges = torch.triu(adj_matrix, diagonal=1).sum(dim=(1, 2))
        self.boost = torch.sqrt(n_edges * self.problem_size * math.log(2) / 2)
        self.baseline = 0.5 * n_edges  # baseline = n_edges/2

        if not train_nc:
            # Set max_iterations
            self.max_iterations = int(self.params[('test_max_steps_multiplier' if self.testing else 'train_max_steps_multiplier')] * self.problem_size)

            # Initialize memory
            self.memory = select_memory(memory_type=self.memory_type,
                                        mem_aggr=self.params['mem_aggr'] if 'mem_aggr' in self.params else 'linear',
                                        value_type=self.params['mem_value_type'],
                                        state_dim=self.problem_size,
                                        distance_metric=self.params['distance_metric'],
                                        adj_matrix=adj_matrix,
                                        batch_size=self.batch_size,
                                        pop_size=self.pop_size,
                                        problem=self.params['problem'],
                                        memory_limit=self.params.get('memory_size', 100000),
                                        device=self.device)

            # Initialize solutions
            solutions = self.generate_batch_of_solutions(adj_matrix, seed)
            ising_solutions = 2 * solutions - 1

            # Initialize mask: mask action 0 (always to 1)
            mask = torch.zeros((batch_size*pop_size, self.problem_size), dtype=torch.bool, device=self.device)
            mask[:, 0] = True

            # Initialize memory info
            if self.params['mem_value_type'] == 'actions':
                n_features = 2
            elif self.params['mem_value_type'] == 'combined':
                n_features = 3
            else:
                n_features = 1
            if self.params['mem_aggr'] == 'concat':
                n_features *= self.params['k']

            mem_info = torch.zeros(batch_size*pop_size, self.problem_size, n_features, dtype=self.float_dtype).to(self.device) if self.memory_type != 'none' else None

            # Initialize state for neural improvement model
            self.state = State(batch_size=batch_size, pop_size=pop_size, problem_size=self.problem_size,
                               graph=adj_matrix, ising_solutions=ising_solutions, solutions=solutions,
                               mask=mask, obj_values=None, mem_info=mem_info, testing=self.testing)

            # Initialize objective values
            self.objective_values = self.compute_obj_value()

            # Initialize environment records
            self.best_objective_values = self.objective_values.clone()
            self.best_solutions = self.state.solutions.clone()
            self.cur_best_objective_values = self.objective_values.clone()
            self.cur_best_solutions = self.state.solutions.clone()
            self.non_improving_steps = torch.zeros(batch_size*pop_size, dtype=self.int_dtype).to(self.device)
            self.last_ms_initializations = None
            self.iteration = 0
        else:
            # Set exploration weight for conditioned network
            if self.params.get('nc_train_mode', None) == 'conditioned_network':
                if self.params['cnc_w_sampling'] == 'uniform':
                    cur_exploration_weight = np.random.rand(batch_size)
                elif self.params['cnc_w_sampling'] == 'beta':
                    # Sampling from a beta distribution
                    alpha = self.params['cnc_beta']
                    beta = self.params['cnc_beta']
                    cur_exploration_weight = np.random.beta(alpha, beta, batch_size)
                elif self.params['cnc_w_sampling'] == 'clipped_normal':
                    w = np.random.normal(loc=0.5, scale=0.2, size=batch_size)
                    cur_exploration_weight = np.clip(w, 0.0, 1.0)
                else:
                    raise NotImplementedError
                cur_exploration_weight = torch.tensor(cur_exploration_weight, dtype=torch.float).to(self.device)

            else:
                cur_exploration_weight = None

            # Set visited solutions
            visited_solutions = None
            if self.params.get('nc_train_mode', None) != 'exploitation':
                visited_solutions = torch.randint(0, 2, (batch_size, self.problem_size, self.params['n_visited_solutions']),
                                                  device=self.device).float()

            # Initialize state for neural constructive
            self.state = State(batch_size=batch_size, pop_size=pop_size, problem_size=self.problem_size, graph=adj_matrix,
                               ising_solutions=None, solutions=None, mask=None, obj_values=None, mem_info=None,
                               visited_solutions=visited_solutions, exploration_weight=cur_exploration_weight,
                               testing=self.testing)

        return self.state, False

    def _build_visited_solutions_reset(self):
        """
        Build visited_solutions for constructive restarts.
        Returns:  -> (B, N, K)
        """
        B = self.batch_size
        P = self.pop_size
        N = self.problem_size
        K = self.params['n_visited_solutions']
        strat = self.params.get('cnc_visit_strategy', 'last_k')

        # Mix across population → (B, N, P) then take last-K along pop axis
        if strat == 'last_k':
            src = self.state.solutions.view(B, P, N).float()
        elif strat == 'best_k_global':
            src = self.best_solutions.view(B, P, N).float()  # per-agent best-so-far
        elif strat == 'best_k_current':
            src = self.cur_best_solutions.view(B, P, N).float()  # per-agent best in current epoch
        else:
            raise ValueError(f"Unknown cnc_visit_strategy: {strat}")

        X = src.permute(0, 2, 1)  # (B, N, P)
        if P >= K:
            return X[:, :, -K:]  # (B, N, K)
        reps = (K + P - 1) // P
        return X.repeat(1, 1, reps)[:, :, :K]  # (B, N, K)

    @torch.no_grad()
    def reset_non_improving_solutions(self, exploration_weight=None):
        """
        Restart the non-improving solutions either at random or using the cNC initializer.
        Produces exactly one new solution per restart slot and keeps dtypes/devices consistent.
        """
        # Which population entries should restart?
        restart_mask = self.non_improving_steps >= self.params['multi_start_patience']
        restart_coords = torch.nonzero(restart_mask, as_tuple=True)[0]
        n_solutions = int(restart_coords.numel())
        if n_solutions == 0:
            return self.state  # nothing to do

        # Update only the selected indices with the random solutions
        if self.params['ms_init_mode'] == 'random':
            # Fresh random solutions for just the restart slots
            new_solutions = torch.randint(
                0, 2, (n_solutions, self.problem_size), dtype=self.int_dtype, device=self.device
            )
        elif self.params['ms_init_mode'] == 'cnc':
            assert self.initializer is not None
            restart_batches = restart_coords // self.pop_size

            exploration_weight = torch.full((self.batch_size,), exploration_weight, dtype=torch.float32, device=self.device)

            visited_solutions = self._build_visited_solutions_reset()

            state = State(batch_size=self.batch_size, pop_size=self.pop_size, problem_size=self.problem_size,
                          graph=self.state.graph, visited_solutions=visited_solutions, exploration_weight=exploration_weight,
                          testing=self.testing)

            # Compute the logits
            logits = self.initializer(state) # (batch, problem, 2)
            probs = torch.nn.functional.softmax(logits, dim=-1) # (batch, problem, 2)

            new_solutions = torch.empty((n_solutions, self.problem_size), dtype=self.int_dtype, device=self.device)
            for b in range(self.batch_size):
                mask_b = (restart_batches == b)
                num_sols_b = int(mask_b.sum().item())
                if num_sols_b == 0:
                    continue

                # per-batch Categorical
                probs_b = probs[b].unsqueeze(0).expand(num_sols_b, -1, -1).contiguous()  # (num_sols_b, N, 2)
                cat = Categorical(probs_b)

                # sample distinct solutions with a capped doubling loop
                needed = num_sols_b
                candidate = None
                tries = 0
                while tries < 6 and (candidate is None or candidate.size(0) < needed):
                    factor = max(2, 2 ** tries)  # 2,4,8,16,32,64
                    draws = cat.sample((factor,)).reshape(-1, self.problem_size)  # (factor*num_sols_b, N)
                    uniq = torch.unique(draws, dim=0)
                    candidate = uniq if candidate is None else torch.unique(torch.cat([candidate, uniq], 0), dim=0)
                    tries += 1

                if candidate.size(0) < needed:
                    # top up with extra (not necessarily unique) samples
                    extra = cat.sample((needed - candidate.size(0),)).reshape(-1, self.problem_size)
                    candidate = torch.cat([candidate, extra], dim=0)

                cand_b = candidate[:needed].to(self.int_dtype)  # (num_sols_b, N)
                cand_b[:, 0] = 1  # force first node = 1

                # place them into positions corresponding to this batch's restart slots
                idxs = torch.nonzero(mask_b, as_tuple=True)[0]  # indices in 0..(n_solutions-1)
                new_solutions[idxs] = cand_b
        else:
            raise NotImplementedError

        # ---- Commit the restart updates ----
        new_solutions[:, 0] = 1  # keep invariant: first node fixed to 1
        self.state.solutions[restart_coords] = new_solutions
        self.state.ising_solutions[restart_coords] = 2 * new_solutions - 1

        # Reset counters for those slots
        self.non_improving_steps[restart_mask] = 0
        self.cur_best_objective_values[restart_mask] = -1e9

        return self.state

    def step(self, action):
        """
        :param action: (torch.Tensor) Action(s) to be executed. Shape: (batch_size)
        """
        self.iteration += 1

        # Save state(key) and action(value) in memory
        self.memory.save_in_memory(self.state.ising_solutions.clone(), action)

        # Update solutions based on actions
        self.state.solutions[self.batch_pop_range, action] = 1 - self.state.solutions[self.batch_pop_range, action]
        self.state.ising_solutions[self.batch_pop_range, action] = - self.state.ising_solutions[self.batch_pop_range, action]

        # Compute obj value of the new solutions
        self.objective_values = self.compute_obj_value()

        # Get memory info: k-nearest neighbors of current solutions
        mem_info, revisited, total_revisited, avg_sim, max_sim, self_perc, gather_idx = self.memory.get_knn(self.state.ising_solutions.clone(), self.params['k'])
        self.state.mem_info = mem_info.to(self.device)
        revisited = revisited.to(self.device)

        # COMPUTE REWARDS (only during training)
        R = self.compute_reward(revisited, total_revisited, avg_sim, max_sim, self_perc, gather_idx)

        # Check if the episode is done
        done = (self.non_improving_steps.sum() >= (self.patience * self.batch_size * self.pop_size)) or (self.iteration >= self.max_iterations)

        # Update obj. value records
        self.non_improving_steps += 1
        best_idx = self.objective_values > self.best_objective_values
        self.best_objective_values[best_idx] = self.objective_values[best_idx]
        self.best_solutions[best_idx, :] = self.state.solutions[best_idx, :]
        cur_best_idx = self.objective_values > self.cur_best_objective_values
        self.cur_best_objective_values[cur_best_idx] = self.objective_values[cur_best_idx]
        self.cur_best_solutions[cur_best_idx, :] = self.state.solutions[cur_best_idx, :]
        self.non_improving_steps[cur_best_idx] = 0

        return self.state, R, done

    def compute_obj_value(self):
        """
        :return: (torch.Tensor) Objective Values of the current solutions.
                 Shape: (batch_size, pop_size) flattened to (batch_size * pop_size,)
        """
        # ising_solutions: (B, P, N)
        # adj_matrix: (B, N, N)
        B, P, N = self.batch_size, self.pop_size, self.state.ising_solutions.shape[-1]

        # (B, P, N)
        ising_solutions = self.state.ising_solutions.clone().reshape(B, P, N)

        # 1) Compute outer products:
        #    ising_solutions.unsqueeze(-1)  => shape (B, P, N, 1)
        #    ising_solutions.unsqueeze(-2)  => shape (B, P, 1, N)
        #    => outer_solutions: (B, P, N, N)
        outer_solutions = ising_solutions.unsqueeze(-1) * ising_solutions.unsqueeze(-2)

        # 2) Compute (1 - outer_solutions), shape (B, P, N, N)
        diff_matrix = 1.0 - outer_solutions

        # 3) Multiply by adj_matrix (broadcast over P dimension):
        #    adj_matrix: (B, N, N)
        #    adj_matrix.unsqueeze(1): (B, 1, N, N) => broadcast to (B, P, N, N)
        product = diff_matrix * self.state.graph.unsqueeze(1)

        # 4) Sum over the last two dimensions (N, N), then multiply by 1/4
        #    => shape (B, P)
        obj_values = 0.25 * product.sum(dim=(-1, -2))

        # 5) Flatten to (B * P,)
        return obj_values.reshape(-1)

    def compute_reward(self, revisited, total_revisited, avg_sim, max_sim, self_perc, gather_idx):
        R = {}

        if self.compute_metrics:

            # EXPLOITATION REWARD: improvement in objective value from the best found
            if self.params.get('normalize_rewards', False):
                obj_value_reward = (self.objective_values - self.best_objective_values) / self.best_objective_values
            else:
                obj_value_reward = (self.objective_values - self.best_objective_values)

            # make zero the negative --> only positive rewards, given when improving the best found
            obj_value_reward[obj_value_reward < 0] = 0

            # MEMORY PUNISHMENT: punish for visiting the same solution again
            R['Re-Visited'] = revisited
            R['Total Re-Visited'] = total_revisited
            revisited_idx = revisited != 0
            if not self.testing:
                obj_value_reward[revisited_idx] -= self.params['revisit_punishment'] * revisited[revisited_idx]
            R['Objective Value Reward'] = obj_value_reward
            R['Avg similarity'] = avg_sim  # average similarity of each solution with its previously visited solutions
            R['Max similarity'] = max_sim  # maximum similarity of each solution with its previously visited solutions
            R['Self memory percentage'] = self_perc
            R['Gathered idx'] = gather_idx  # indices of the gathered threads from memory
            self.reward = obj_value_reward

            R['Reward'] = self.reward

        return R

    @torch.no_grad()
    def generate_batch_of_solutions(self, adj_matrix, seed):
        """
        Generates a batch of binary solutions for the MaxCut problem
        Args:
        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :param seed: (int) Random seed
        :return: (torch.Tensor) Solutions of the graph. Shape: (batch_size*pop_size, n)
        """

        if seed is not None:
            torch.manual_seed(seed)
            if self.device == 'cuda':
                torch.cuda.manual_seed_all(seed)

        if self.initialization == 'cnc':
            assert self.initializer is not None
            state = State(batch_size=self.batch_size, pop_size=self.pop_size, problem_size=self.problem_size,
                          graph=adj_matrix, testing=self.testing)

            exploration_weights = torch.linspace(0, 0.4, steps=self.pop_size, device=self.device)
            n_vis_sols = self.params['n_visited_solutions']
            all_solutions = []
            for p in range(self.pop_size):
                state.exploration_weight = torch.full((self.batch_size,), exploration_weights[p].item(), dtype=torch.float32, device=self.device)
                if len(all_solutions) == 0:
                    visited_solutions = torch.randint(0, 2, (self.batch_size, self.problem_size, n_vis_sols), dtype=self.float_dtype).to(self.device)
                else:
                    prev_solutions = torch.stack(all_solutions, dim=2)  # [batch_size, problem_size, p]

                    available = prev_solutions.shape[2]
                    if available < n_vis_sols:
                        # Calculate how many times we need to tile
                        repeats = (n_vis_sols + available - 1) // available  # Ceiling division
                        prev_solutions = prev_solutions.repeat(1, 1, repeats)  # Repeat only along the 3rd dim (solutions)

                    # Cut exactly to the value accepted by the cNC - n_vis_sols
                    visited_solutions = prev_solutions[:, :, :n_vis_sols]

                state.visited_solutions = visited_solutions
                logits = self.initializer(state)

                # Take the argmax solution
                sol = logits.argmax(dim=-1).to(self.int_dtype)
                sol[:, 0] = 1  # set the first node to 1 always
                all_solutions.append(sol)

            solutions = torch.stack(all_solutions, dim=1).reshape(self.batch_size*self.pop_size, self.problem_size)

        elif self.initialization == 'random':
            # All random
            solutions = torch.randint(0, 2, (self.batch_size*self.pop_size, self.problem_size), dtype=self.int_dtype).to(self.device)

        else:
            raise NotImplementedError

        solutions[:, 0] = 1  # set the first node to 1 always
        return solutions

    def heatmap_inference(self, logits, feasibility_decoder=None, visited_solutions=None, n_rollouts=10, training=True, greedy=False, calib_beta=None, compute_diversity=True):
        """
        :param logits: (torch.Tensor or None) Logits of the actions. Shape: (batch_size, n)
        :param feasibility_decoder: (torch.nn.Module or None) Feasibility decoder for compatibility (not used here)
        :param visited_solutions: (torch.Tensor or None) Visited solutions. Shape: (batch_size, n, n_visited_sol)
        :param n_rollouts: (int) Number of rollouts for sampling
        :param training: (bool) Whether to compute rewards and logprobs for training
        :param greedy: (bool) Whether to use greedy decoding
        :param calib_beta: (float or None) If not None, apply calibration with the given beta
        :param compute_diversity: (bool) Whether to compute env-level diversity rewards
        """
        # Compute the log probabilities
        log_p = F.log_softmax(logits, dim=-1)
        probs = log_p.exp()
        if self.params.get('vectorize_rollouts', True) and n_rollouts > 1 and not greedy:
            return self._heatmap_inference_vectorized(
                log_p,
                probs,
                visited_solutions=visited_solutions,
                n_rollouts=n_rollouts,
                training=training,
                compute_diversity=compute_diversity,
            )

        probs = probs.reshape(self.batch_size*self.problem_size, 2)
        all_solutions = []
        log_probs = []
        diversity_rewards = [] if (compute_diversity and visited_solutions is not None and training) else None
        log_probs_mean = None
        for n_roll in range(n_rollouts):
            if greedy and n_roll == 0:
                if calib_beta is not None:
                    # Calibrated greedy decoding
                    margins = (logits[..., 1] - logits[..., 0]).reshape(self.batch_size * self.problem_size)
                    solutions = (margins + calib_beta >= 0).to(torch.int64).reshape(self.batch_size, self.problem_size)

                else:
                    # Standard greedy decoding
                    solutions = probs.argmax(dim=-1).reshape(self.batch_size, self.problem_size)
            else:
                solutions = probs.multinomial(1).reshape(self.batch_size, self.problem_size)
            solutions[:, 0] = 1  # set the first node to 1 always
            all_solutions.append(solutions)
            if training:
                log_probs.append(log_p.gather(2, solutions.unsqueeze(-1)).squeeze(-1))

                if compute_diversity and visited_solutions is not None:
                    cur_div = self.distance_fn(solutions, visited_solutions, self.state.graph)  # (B,) in [0,1]
                    diversity_rewards.append(cur_div)

        all_solutions = torch.stack(all_solutions, dim=1)
        if training:
            log_probs = torch.stack(log_probs, dim=1)
            if compute_diversity and visited_solutions is not None:
                diversity_rewards = torch.stack(diversity_rewards, dim=1)

            # Reduce over nodes except the first one, which is fixed to 1
            # and thus provides no learning signal.
            if self.params.get('ppo_logprob_reduction', 'mean') == 'sum':
                log_probs_mean = log_probs[:, :, 1:].sum(dim=2)
            else:
                log_probs_mean = log_probs[:, :, 1:].mean(dim=2)
            #log_probs_mean = log_probs.mean(dim=2)  # For other problems not fixing the first node

        # Compute obj values
        ising_solutions = 2 * all_solutions - 1

        # 1) Compute outer products:
        outer_solutions = ising_solutions.unsqueeze(-1) * ising_solutions.unsqueeze(-2)
        # 2) Compute (1 - outer_solutions), shape (B, P, N, N)
        diff_matrix = 1.0 - outer_solutions
        # 3) Multiply by adj_matrix (broadcast over P dimension):
        product = diff_matrix * self.state.graph.unsqueeze(1)
        # 4) Sum over the last two dimensions (N, N), then multiply by 1/4
        obj_values = 0.25 * product.sum(dim=(-1, -2))

        # Normalize objective value reward to [-1, 1]
        obj_reward = (obj_values - self.baseline.unsqueeze(-1)) / self.boost.unsqueeze(-1)

        # Dictionary for additional metrics
        R_dict = {}
        return all_solutions, obj_values, obj_reward, diversity_rewards, log_probs_mean, R_dict

    def _heatmap_inference_vectorized(self, log_p, probs, visited_solutions=None, n_rollouts=10, training=True, compute_diversity=True):
        batch_size, problem_size, n_actions = log_p.shape
        flat_probs = probs.unsqueeze(1).expand(batch_size, n_rollouts, problem_size, n_actions)
        flat_probs = flat_probs.reshape(batch_size * n_rollouts * problem_size, n_actions)
        all_solutions = flat_probs.multinomial(1).reshape(batch_size, n_rollouts, problem_size)
        all_solutions[:, :, 0] = 1

        log_probs_mean = None
        diversity_rewards = None
        if training:
            expanded_log_p = log_p.unsqueeze(1).expand(batch_size, n_rollouts, problem_size, n_actions)
            log_probs = expanded_log_p.gather(3, all_solutions.unsqueeze(-1)).squeeze(-1)
            if self.params.get('ppo_logprob_reduction', 'mean') == 'sum':
                log_probs_mean = log_probs[:, :, 1:].sum(dim=2)
            else:
                log_probs_mean = log_probs[:, :, 1:].mean(dim=2)

            if compute_diversity and visited_solutions is not None:
                flat_solutions = all_solutions.reshape(batch_size * n_rollouts, problem_size)
                flat_visited = visited_solutions.unsqueeze(1).expand(
                    batch_size,
                    n_rollouts,
                    problem_size,
                    visited_solutions.size(-1),
                ).reshape(batch_size * n_rollouts, problem_size, visited_solutions.size(-1))
                flat_graph = self.state.graph.unsqueeze(1).expand(batch_size, n_rollouts, problem_size, problem_size)
                flat_graph = flat_graph.reshape(batch_size * n_rollouts, problem_size, problem_size)
                flat_div = self.distance_fn(flat_solutions, flat_visited, flat_graph)
                diversity_rewards = flat_div.reshape(batch_size, n_rollouts)

        ising_solutions = 2 * all_solutions - 1
        outer_solutions = ising_solutions.unsqueeze(-1) * ising_solutions.unsqueeze(-2)
        diff_matrix = 1.0 - outer_solutions
        product = diff_matrix * self.state.graph.unsqueeze(1)
        obj_values = 0.25 * product.sum(dim=(-1, -2))
        obj_reward = (obj_values - self.baseline.unsqueeze(-1)) / self.boost.unsqueeze(-1)

        R_dict = {}
        return all_solutions, obj_values, obj_reward, diversity_rewards, log_probs_mean, R_dict
