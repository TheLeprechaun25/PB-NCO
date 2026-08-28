import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from env.memory import select_memory
from env.generators import ERGraphGenerator, RBGraphGenerator
from utils.env_utils import State, distance_fn


class MISEnv:
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

        self.expected = None  # For normalizing the NC reward
        self.upper_bound = None  # For normalizing the NC reward

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

        # Other node features: degree
        A = adj_matrix.float()
        deg = A.sum(dim=-1, keepdim=True)  # (B, N, 1)
        deg_norm = deg / (self.problem_size - 1 + 1e-6)

        # 2-hop degree (count of nodes reachable in ≤2 steps, excluding self) approx
        A2 = (A @ A).clamp(max=1.0)
        twohop = (A + A2).clamp(max=1.0)  # adjacency of nodes within 1 or 2 hops
        twohop = twohop.sum(dim=-1, keepdim=True) - 1.0  # exclude self
        deg2_norm = twohop / (self.problem_size + 1e-6)
        extra_node_feats = torch.cat([deg_norm, deg2_norm], dim=-1)  # (B, N, 2)

        reward_norm = self.params.get('mis_reward_norm', 'upper_bound')
        if train_nc and self.params.get('skip_unused_mis_bounds', False):
            if reward_norm == 'upper_bound':
                self.expected, self.upper_bound = compute_upper_bounds(A)
            elif reward_norm == 'centered_size':
                self.expected = compute_expected_mis_size(A)
                self.upper_bound = None
            else:
                self.expected = None
                self.upper_bound = None
        else:
            # Preserve current RNG consumption and reward normalization by default.
            self.expected, self.upper_bound = compute_upper_bounds(A)

        if not train_nc:  # Training NI model
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
            solutions = self.generate_batch_of_solutions(adj_matrix, extra_node_feats, seed)
            ising_solutions = 2 * solutions - 1

            # Initialize mask
            mask = self.create_action_mask(adj_matrix, solutions)

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
                               graph=adj_matrix, ising_solutions=ising_solutions, solutions=solutions, extra_node_feats=extra_node_feats,
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
                # Random archive context for exploration-oriented cNC training.
                visited_solutions = torch.randint(0, 2, (batch_size, self.problem_size, self.params['n_visited_solutions']),
                                                  device=self.device).float()

            # Initialize state for neural constructive
            self.state = State(batch_size=batch_size, pop_size=pop_size, problem_size=self.problem_size, graph=adj_matrix,
                               ising_solutions=None, solutions=None, mask=None, obj_values=None, mem_info=None, extra_node_feats=extra_node_feats,
                               visited_solutions=visited_solutions, exploration_weight=cur_exploration_weight,
                               testing=self.testing)

        return self.state, False

    def _build_visited_solutions_reset(self):
        """
        For multi-start reset, build visited_solutions from current population state according to strategy.
        Returns: (B, N, K) float
        """
        B = self.batch_size
        N = self.problem_size
        K = self.params['n_visited_solutions']
        strat = self.params.get('cnc_visit_strategy', 'last_k')

        if strat == 'last_k':
            # all current solutions: (B, P, N) -> (B, N, P)
            cur = self.state.solutions.view(B, self.pop_size, N).permute(0, 2, 1).float()
            if self.pop_size >= K:
                return cur[:, :, -K:]
            reps = (K + self.pop_size - 1) // self.pop_size
            return cur.repeat(1, 1, reps)[:, :, :K]

        elif strat == 'best_k_global':
            best_global = self.best_solutions.view(B, self.pop_size, N).permute(0, 2, 1).float()
            if self.pop_size >= K:
                return best_global[:, :, -K:]
            reps = (K + self.pop_size - 1) // self.pop_size
            return best_global.repeat(1, 1, reps)[:, :, :K]

        elif strat == 'best_k_current':
            best_current = self.cur_best_solutions.view(B, self.pop_size, N).permute(0, 2, 1).float()
            if self.pop_size >= K:
                return best_current[:, :, -K:]
            reps = (K + self.pop_size - 1) // self.pop_size
            return best_current.repeat(1, 1, reps)[:, :, :K]

        else:
            raise ValueError(f"Unknown cnc_visit_strategy: {strat}")
        
    @torch.no_grad()
    def reset_non_improving_solutions(self, exploration_weight=None):
        """
        Restart the non-improving solutions with random solutions
        """
        # Which population entries should restart?
        restart_mask = self.non_improving_steps >= self.params['multi_start_patience']
        restart_coords = torch.nonzero(restart_mask, as_tuple=True)[0]
        n_solutions = int(restart_coords.numel())
        if n_solutions == 0:
            return self.state  # nothing to do

        # Update only the selected indices with the random solutions
        if self.params['ms_init_mode'] == 'random':
            # Generate random solutions only for the restart indices
            for r in restart_coords:
                batch_idx = r // self.pop_size
                adj = self.state.graph[batch_idx]
                self.state.solutions[r] = random_greedy_partial(adj, 1)[0]
                self.state.ising_solutions[r] = 2 * self.state.solutions[r] - 1

        elif self.params['ms_init_mode'] == 'random_greedy':
            # Generate random solutions only for the restart indices
            for r in restart_coords:
                batch_idx = r // self.pop_size
                adj = self.state.graph[batch_idx]
                self.state.solutions[r] = random_greedy_initialization(adj, 1)[0]
                self.state.ising_solutions[r] = 2 * self.state.solutions[r] - 1

        elif self.params['ms_init_mode'] == 'zeros':
            self.state.solutions[restart_coords] = torch.zeros((n_solutions, self.problem_size), dtype=self.int_dtype).to(self.device)
            self.state.ising_solutions[restart_coords] = -1 * torch.ones((n_solutions, self.problem_size), dtype=self.int_dtype).to(self.device)

        elif self.params['ms_init_mode'] in ['cnc', 'greedy_cnc']:
            assert self.initializer is not None
            restart_batches = restart_coords // self.pop_size

            if self.params['ms_init_mode'] == 'cnc':
                exploration_weight = torch.full((self.batch_size,), exploration_weight, dtype=torch.float32, device=self.device)
                visited_solutions = self._build_visited_solutions_reset()
            else:  # greedy_cnc
                exploration_weight = None
                visited_solutions = None

            state = State(batch_size=self.batch_size, pop_size=self.pop_size, problem_size=self.problem_size,
                          graph=self.state.graph, visited_solutions=visited_solutions, exploration_weight=exploration_weight,
                          extra_node_feats=self.state.extra_node_feats, testing=self.testing)

            # Compute the logits
            logits = self.initializer(state) # (batch, problem, 2)

            new_solutions = torch.empty((n_solutions, self.problem_size), dtype=self.int_dtype, device=self.device)
            for b in range(self.batch_size):
                mask_b = (restart_batches == b)
                num_sols_b = int(mask_b.sum().item())
                if num_sols_b == 0:
                    continue

                logits_rep = logits[b].unsqueeze(0).expand(num_sols_b, -1, -1)  # (num_sols_b, N, 2)
                adj_matrix_rep = self.state.graph[b].unsqueeze(0).expand(num_sols_b, -1, -1)  # (num_sols_b, N, N)
                cand_b, _, _, _, _, _, _ = heatmap_decoding_vectorised(
                    logits_rep, adj_matrix_rep, greedy=False, removal_order='ascending', calib_beta=None,
                    post_add=False, post_add_mode="sample",  # "greedy" | "sample"
                    post_add_temperature=1.0,
                )
                idxs = torch.nonzero(mask_b, as_tuple=True)[0]  # indices in 0..(n_solutions-1)
                new_solutions[idxs] = cand_b.int()

            self.state.solutions[restart_coords] = new_solutions
            self.state.ising_solutions[restart_coords] = 2 * new_solutions - 1

        else:
            raise NotImplementedError

        # Update the non-improving steps
        self.non_improving_steps[restart_mask] = 0

        # Reset the current best objective values
        self.cur_best_objective_values[restart_mask] = -1e9

        # Recompute masks after resetting a subset of solutions.
        self.state.mask = self.create_action_mask(self.state.graph, self.state.solutions)

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

        # Update masks
        self.state.mask = self.create_action_mask(self.state.graph, self.state.solutions)

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
        :return: (torch.Tensor) Objective value of the current solutions. Shape: (batch_size, pop_size)
        """
        # Calculate obj value as the number of nodes in the set
        return torch.sum(self.state.solutions, dim=1).float()

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

    def create_action_mask(self, adj_matrix, solutions):
        """
        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :param solutions: (torch.Tensor) Solutions of the graph. Shape: (batch_size, pop_size, n) or (batch_size*pop_size, n)
        :return: (torch.Tensor) Action mask. Shape: (batch_size, pop_size, n)
        """
        # repeat adj_matrix for each pomo
        adj_matrix = adj_matrix.unsqueeze(1).expand(-1, self.pop_size, -1, -1).reshape(self.batch_size*self.pop_size, self.problem_size, self.problem_size)

        solutions = solutions.reshape(self.batch_size*self.pop_size, self.problem_size).unsqueeze(2).float()

        # Use batch matrix multiplication to find if any adjacent node is in the set
        adjacent_mask = torch.bmm(adj_matrix.float(), solutions).squeeze(2)  # Shape: (batch_size, n)

        # Nodes that can't be added (any adjacent node is in the set)
        action_mask = (adjacent_mask > 0) & (solutions.squeeze(2) == 0)

        return action_mask.to(self.device)

    def generate_batch_of_solutions(self, adj_matrix, extra_node_feats=None, seed=None):
        """
        Generate a batch of solutions for the max independent set problem.
        Methods:
            - max_degree_greedy (deterministic): Pick the node with the maximum (non-zero) degree among available nodes.
            - randomized_greedy (stochastic): Semi-greedy: among the smallest-degree (non-zero) available nodes, pick one at random.
            - neighbor_degree_heuristic (deterministic): Score(node) = alpha * deg(node) + (1 - alpha) * sum_{u in neighbors(node)} deg(u).
            - weighted_random (stochastic): At each step, pick a node with probability ~ 1/(degree+1).
            - permutation_random (stochastic): Randomly permute the nodes and add them to the set if they don't violate the independent set condition.
            - random (stochastic): Randomly pick nodes until no valid nodes remain.
            - greedy_random (both): First solution uses greedy initialization, the rest use random initialization.

        :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
        :param extra_node_feats: (torch.Tensor or None) Extra node features. Shape: (batch_size, n, n_feats)
        :param seed: (int) Seed for initializing solutions
        :return: (torch.Tensor) Solutions of the graph. Shape: (batch_size, pop_size, n)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == 'cuda':
                torch.cuda.manual_seed(seed)

        solutions = torch.zeros(self.batch_size, self.pop_size, self.problem_size, dtype=torch.float32).to(self.device)

        if self.initialization == 'cnc':
            assert self.initializer is not None
            B, N = adj_matrix.shape[0], adj_matrix.shape[1]
            K = self.params['n_visited_solutions']

            exploration_weights = torch.linspace(0, 1, steps=self.pop_size, device=self.device)
            sols = []
            for p in range(self.pop_size):
                if p == 0:
                    visited = torch.randint(0, 2, (B, N, K), dtype=self.float_dtype, device=self.device)
                else:
                    prev = torch.stack(sols, dim=2)  # (B, N, p)
                    reps = (K + prev.size(2) - 1) // prev.size(2)
                    visited = prev.repeat(1, 1, reps)[:, :, :K].float()

                st = State(batch_size=B, pop_size=self.pop_size, problem_size=N, graph=adj_matrix,
                           visited_solutions=visited, extra_node_feats=extra_node_feats,
                           exploration_weight=torch.full((B,), exploration_weights[p], device=self.device),
                           testing=self.testing)

                logits = self.initializer(st)  # (B, N, 2)

                sol_p, _, _, _, _, _, _ = heatmap_decoding_vectorised(
                    logits, adj_matrix, greedy=False, removal_order='ascending', calib_beta=None,
                    post_add=False, post_add_mode="sample",  # "greedy" | "sample"
                    post_add_temperature=1.0,
                )

                sols.append(sol_p.int())

            solutions = torch.stack(sols, dim=1)  # (B, P, N)

        elif self.initialization == 'greedy_cnc':
            assert self.initializer is not None
            B, N = adj_matrix.shape[0], adj_matrix.shape[1]
            st = State(batch_size=B, pop_size=self.pop_size, problem_size=N, graph=adj_matrix, visited_solutions=None,
                       extra_node_feats=extra_node_feats, exploration_weight=None, testing=self.testing)

            logits = self.initializer(st)  # (B, N, 2)
            # Repeat pop_size times
            logits_rep = logits.unsqueeze(1).expand(-1, self.pop_size, -1, -1).reshape(B * self.pop_size, N, 2)
            adj_matrix_rep = adj_matrix.unsqueeze(1).expand(-1, self.pop_size, -1, -1).reshape(B * self.pop_size, N, N)

            sol_p, _, _, _, _, _, _ = heatmap_decoding_vectorised(
                logits_rep, adj_matrix_rep, greedy=False, removal_order='ascending', calib_beta=None,
                post_add=False, post_add_mode="sample",  # "greedy" | "sample"
                post_add_temperature=1.0,
            )

            solutions = sol_p.reshape(B, self.pop_size, N)

        elif self.initialization == 'random':
            for b in range(self.batch_size):
                solutions[b, :] = random_greedy_partial(adj_matrix[b], self.pop_size)

        elif self.initialization == 'random_greedy':
            for b in range(self.batch_size):
                solutions[b, :] = random_greedy_initialization(adj_matrix[b], self.pop_size)

        elif self.initialization == 'zeros':
            solutions = torch.zeros(self.batch_size, self.pop_size, self.problem_size, dtype=self.int_dtype).to(self.device)

        else:
            raise NotImplementedError

        return solutions.int().reshape(self.batch_size*self.pop_size, self.problem_size)

    def heatmap_inference(self, logits, feasibility_decoder='heatmap', visited_solutions=None, n_rollouts=10, training=True, greedy=False, calib_beta=None, compute_diversity=True):
        """
        :param logits: (torch.Tensor or None) Logits of the actions. Shape: (batch_size, n)
        :param feasibility_decoder: (str) Feasibility decoder to use: 'heatmap' or 'sequential'
        :param visited_solutions: (torch.Tensor or None) Visited solutions. Shape: (batch_size, n, n_visited_sol)
        :param n_rollouts: (int) Number of rollouts for sampling
        :param training: (bool) Whether to compute rewards and logprobs for training
        :param greedy: (bool) Whether to use greedy decoding
        :param calib_beta: (float or None) If not None, apply calibration with the given beta
        :param compute_diversity: (bool) Whether to compute env-level diversity rewards
        """
        if (
            self.params.get('vectorize_rollouts', True)
            and feasibility_decoder == 'heatmap'
            and n_rollouts > 1
            and not greedy
        ):
            return self._heatmap_inference_vectorized(
                logits,
                visited_solutions=visited_solutions,
                n_rollouts=n_rollouts,
                training=training,
                calib_beta=calib_beta,
                compute_diversity=compute_diversity,
            )

        # Compute the log probabilities
        all_solutions = []
        all_ppo_actions = []
        all_repaired_solutions = []
        all_select_probs = []
        log_probs = []
        diversity_rewards = [] if (compute_diversity and visited_solutions is not None and training) else None
        log_probs_mean = None
        all_num_of_removals = []
        all_num_of_conflicts = []
        for n_roll in range(n_rollouts):
            use_greedy = greedy and n_roll == 0

            if feasibility_decoder == 'sequential':
                solutions, log_p_mean = vec_sequential_decoding(logits, self.state.graph, greedy=use_greedy)
                ppo_actions = None
                num_removals = torch.zeros(self.state.graph.size(0), device=logits.device)
                num_conflicts = torch.zeros_like(num_removals)
            elif feasibility_decoder == 'sequential_with_shaping':
                solutions, log_p_mean = sequential_with_shaping(logits, self.state.graph, greedy=use_greedy)
                ppo_actions = None
                num_removals = torch.zeros(self.state.graph.size(0), device=logits.device)
                num_conflicts = torch.zeros_like(num_removals)
            else:
                solutions, log_p_mean, num_removals, num_conflicts, ppo_actions, repaired_solutions, select_probs = heatmap_decoding_vectorised(
                    logits,
                    self.state.graph,
                    greedy=use_greedy,
                    removal_order='ascending',
                    calib_beta=calib_beta,
                    post_add=self.params.get('mis_heatmap_post_add', False),
                    post_add_mode=self.params.get('mis_heatmap_post_add_mode', 'greedy'),
                    post_add_temperature=self.params.get('mis_heatmap_post_add_temperature', 1.0),
                    logprob_reduction=self.params.get('ppo_logprob_reduction', 'mean'),
                )
            all_num_of_removals.append(num_removals)
            all_num_of_conflicts.append(num_conflicts)

            all_solutions.append(solutions)
            if ppo_actions is not None:
                all_ppo_actions.append(ppo_actions)
                all_repaired_solutions.append(repaired_solutions)
                all_select_probs.append(select_probs)
            if training:
                log_probs.append(log_p_mean)

                if compute_diversity and visited_solutions is not None:
                    cur_div = self.distance_fn(solutions, visited_solutions, self.state.graph)  # (B,) in [0,1]
                    diversity_rewards.append(cur_div)

        all_solutions = torch.stack(all_solutions, dim=1)
        if training:
            log_probs = torch.stack(log_probs, dim=1)
            if compute_diversity and visited_solutions is not None:
                diversity_rewards = torch.stack(diversity_rewards, dim=1)

            # Average log-probabilities over nodes
            log_probs_mean = log_probs
            #log_probs_mean = log_probs.mean(dim=2)

        # Compute obj values
        obj_values = torch.sum(all_solutions, dim=2).float()

        punish_value = torch.stack(all_num_of_removals, dim=1).float()
        if self.params['punish_unfeasible']:
            punish_weight = self.params['punish_w']
            raw_score = obj_values - punish_weight * punish_value
        else:
            raw_score = obj_values

        reward_norm = self.params.get('mis_reward_norm', 'upper_bound')
        if reward_norm == 'upper_bound':
            denom = (self.upper_bound - self.expected).clamp(min=1e-6).unsqueeze(dim=1)
            obj_reward = 2.0 * (raw_score - self.expected.unsqueeze(dim=1)) / denom - 1.0
        elif reward_norm == 'size':
            obj_reward = raw_score / max(self.problem_size, 1)
        elif reward_norm == 'centered_size':
            obj_reward = (raw_score - self.expected.unsqueeze(dim=1)) / max(self.problem_size, 1)
        elif reward_norm == 'rollout_rank':
            if raw_score.size(1) <= 1:
                obj_reward = torch.zeros_like(raw_score)
            else:
                order = raw_score.argsort(dim=1)
                ranks = torch.empty_like(raw_score, dtype=torch.float32)
                rank_values = torch.arange(raw_score.size(1), device=raw_score.device, dtype=torch.float32)
                ranks.scatter_(1, order, rank_values.unsqueeze(0).expand_as(raw_score))
                obj_reward = 2.0 * ranks / (raw_score.size(1) - 1) - 1.0
        else:
            raise ValueError(f"Invalid mis_reward_norm: {reward_norm}")

        R_dict = {
            'all_num_of_removals': torch.stack(all_num_of_removals, dim=1).float(),
            'all_num_of_conflicts': torch.stack(all_num_of_conflicts, dim=1).float(),
        }
        if all_ppo_actions:
            R_dict['ppo_actions'] = torch.stack(all_ppo_actions, dim=1)
            R_dict['raw_solutions'] = R_dict['ppo_actions']
            R_dict['repaired_solutions'] = torch.stack(all_repaired_solutions, dim=1)
            R_dict['final_solutions'] = all_solutions
            R_dict['post_added'] = (all_solutions - R_dict['repaired_solutions']).sum(dim=-1).float()
            R_dict['select_probs'] = torch.stack(all_select_probs, dim=1)
            R_dict['raw_score'] = raw_score

        return all_solutions, obj_values, obj_reward, diversity_rewards, log_probs_mean, R_dict

    def _heatmap_inference_vectorized(self, logits, visited_solutions=None, n_rollouts=10, training=True, calib_beta=None, compute_diversity=True):
        batch_size, problem_size, n_actions = logits.shape
        flat_logits = logits.unsqueeze(1).expand(batch_size, n_rollouts, problem_size, n_actions)
        flat_logits = flat_logits.reshape(batch_size * n_rollouts, problem_size, n_actions)
        flat_graph = self.state.graph.unsqueeze(1).expand(batch_size, n_rollouts, problem_size, problem_size)
        flat_graph = flat_graph.reshape(batch_size * n_rollouts, problem_size, problem_size)

        flat_solutions, flat_log_probs, flat_removals, flat_conflicts, flat_actions, flat_repaired, flat_select_probs = heatmap_decoding_vectorised(
            flat_logits,
            flat_graph,
            greedy=False,
            removal_order='ascending',
            calib_beta=calib_beta,
            post_add=self.params.get('mis_heatmap_post_add', False),
            post_add_mode=self.params.get('mis_heatmap_post_add_mode', 'greedy'),
            post_add_temperature=self.params.get('mis_heatmap_post_add_temperature', 1.0),
            logprob_reduction=self.params.get('ppo_logprob_reduction', 'mean'),
        )

        all_solutions = flat_solutions.reshape(batch_size, n_rollouts, problem_size)
        log_probs_mean = flat_log_probs.reshape(batch_size, n_rollouts) if training else None
        all_num_of_removals = flat_removals.reshape(batch_size, n_rollouts).float()
        all_num_of_conflicts = flat_conflicts.reshape(batch_size, n_rollouts).float()

        diversity_rewards = None
        if compute_diversity and visited_solutions is not None and training:
            flat_visited = visited_solutions.unsqueeze(1).expand(
                batch_size,
                n_rollouts,
                problem_size,
                visited_solutions.size(-1),
            ).reshape(batch_size * n_rollouts, problem_size, visited_solutions.size(-1))
            flat_div = self.distance_fn(flat_solutions, flat_visited, flat_graph)
            diversity_rewards = flat_div.reshape(batch_size, n_rollouts)

        obj_values = torch.sum(all_solutions, dim=2).float()
        if self.params['punish_unfeasible']:
            raw_score = obj_values - self.params['punish_w'] * all_num_of_removals
        else:
            raw_score = obj_values

        reward_norm = self.params.get('mis_reward_norm', 'upper_bound')
        if reward_norm == 'upper_bound':
            denom = (self.upper_bound - self.expected).clamp(min=1e-6).unsqueeze(dim=1)
            obj_reward = 2.0 * (raw_score - self.expected.unsqueeze(dim=1)) / denom - 1.0
        elif reward_norm == 'size':
            obj_reward = raw_score / max(self.problem_size, 1)
        elif reward_norm == 'centered_size':
            obj_reward = (raw_score - self.expected.unsqueeze(dim=1)) / max(self.problem_size, 1)
        elif reward_norm == 'rollout_rank':
            if raw_score.size(1) <= 1:
                obj_reward = torch.zeros_like(raw_score)
            else:
                order = raw_score.argsort(dim=1)
                ranks = torch.empty_like(raw_score, dtype=torch.float32)
                rank_values = torch.arange(raw_score.size(1), device=raw_score.device, dtype=torch.float32)
                ranks.scatter_(1, order, rank_values.unsqueeze(0).expand_as(raw_score))
                obj_reward = 2.0 * ranks / (raw_score.size(1) - 1) - 1.0
        else:
            raise ValueError(f"Invalid mis_reward_norm: {reward_norm}")

        R_dict = {
            'all_num_of_removals': all_num_of_removals,
            'all_num_of_conflicts': all_num_of_conflicts,
            'ppo_actions': flat_actions.reshape(batch_size, n_rollouts, problem_size),
            'raw_solutions': flat_actions.reshape(batch_size, n_rollouts, problem_size),
            'repaired_solutions': flat_repaired.reshape(batch_size, n_rollouts, problem_size),
            'final_solutions': all_solutions,
            'post_added': (all_solutions - flat_repaired.reshape(batch_size, n_rollouts, problem_size)).sum(dim=-1).float(),
            'select_probs': flat_select_probs.reshape(batch_size, n_rollouts, problem_size),
            'raw_score': raw_score,
        }
        return all_solutions, obj_values, obj_reward, diversity_rewards, log_probs_mean, R_dict


def vec_sequential_decoding(logits, adj_matrix, greedy=False):
    """
    :param logits: (torch.Tensor) Logits of the actions. Shape: (batch_size, n, 2) with each nodes probability of belonging to the independent set
    :param adj_matrix: (torch.Tensor) Adjacency matrix of the graph. Shape: (batch_size, n, n)
    :param greedy: (bool) Whether to use greedy decoding
    """
    batch_size, problem_size, _ = logits.size()
    device = logits.device
    solution = torch.zeros(batch_size, problem_size, dtype=torch.long, device=device)
    available = torch.ones(batch_size, problem_size, dtype=torch.bool, device=device)

    scores = F.softmax(logits, dim=-1)[..., 1]  # (B, N)
    # set to 0 every score below 0.5
    scores[scores < 0.5] = 0.0
    while True:
        # Mask out unavailable nodes and compute sums
        masked = scores.clone()
        masked[~available] = 0.0
        sums = masked.sum(dim=1)

        # Identify graphs that can still sample (sum>0)
        act_idx = (sums > 0).nonzero(as_tuple=False).squeeze(1)
        if act_idx.numel() == 0:
            break

        if greedy:
            # pick the highest-logit available node
            choice_active = masked[act_idx].argmax(dim=1)
        else:
            # normalize over available nodes & sample
            probs_active = masked[act_idx] / sums[act_idx].unsqueeze(1)
            choice_active = torch.multinomial(probs_active, 1).squeeze(1)

        # Map back to global indices
        batch_idx = act_idx
        chosen = choice_active

        # Set solution
        solution[batch_idx, chosen] = 1

        # Build removal mask: chosen nodes + neighbors
        neighbors_mask = adj_matrix[batch_idx, chosen] > 0  # (num_active, n)
        removal = torch.zeros(batch_size, problem_size, dtype=torch.bool, device=device)
        removal[batch_idx] = neighbors_mask
        removal[batch_idx, chosen] = True

        # Update availability
        available[removal] = False

    log_p = F.log_softmax(logits, dim=-1)
    log_p_mean = log_p.gather(2, solution.unsqueeze(-1)).squeeze(-1).mean(dim=1)

    return solution, log_p_mean


def sequential_with_shaping(logits, adj, greedy=False,
                            lam_conf=1.0, mu_deg=0.0):
    B, N, _ = logits.shape
    device = logits.device
    base_scores = F.softmax(logits, dim=-1)[..., 1]  # (B,N)
    selected = torch.zeros(B, N, dtype=torch.bool, device=device)
    available = torch.ones(B, N, dtype=torch.bool, device=device)
    logp_sum = logits.new_zeros(B)                  # accumulate per-step log-probs

    while True:
        # conflict count for each node given current selected
        conf = (adj.float() @ selected.float().unsqueeze(-1)).squeeze(-1)  # (B,N)

        # degree among currently available neighbors (optional)
        avail_neighbors = available.unsqueeze(1) & (adj > 0)       # (B,N,N)
        deg_avail = avail_neighbors.float().sum(-1)                 # (B,N)

        shaped = base_scores - lam_conf * conf - mu_deg * deg_avail
        shaped = shaped.masked_fill(~available, float('-inf'))      # respect feasibility

        # stop if nothing left
        sums = torch.isfinite(shaped).float().sum(dim=1)
        act_idx = (sums > 0).nonzero(as_tuple=False).squeeze(1)
        if act_idx.numel() == 0:
            break

        if greedy:
            choice = shaped[act_idx].argmax(dim=1)
            # logprob under softmax(shaped)
            logp = F.log_softmax(shaped[act_idx], dim=1)[torch.arange(act_idx.numel(), device=device), choice]
        else:
            logp = F.log_softmax(shaped[act_idx], dim=1)
            dist = torch.distributions.Categorical(logits=logp)
            choice = dist.sample()
            logp = logp[torch.arange(act_idx.numel(), device=device), choice]

        logp_sum[act_idx] += logp

        # apply the pick
        selected[act_idx, choice] = True
        # forbid picked and its neighbors
        forbid = (adj[act_idx, choice] > 0)
        available[act_idx, choice] = False
        available[act_idx] &= ~forbid

    sol = selected.long()
    log_p_mean = (logp_sum / (selected.sum(dim=1).clamp_min(1))).nan_to_num(0.0)
    return sol, log_p_mean


def heatmap_decoding_vectorised(
    node_logits,
    adj_matrix,
    greedy: bool = False,
    removal_order: str = "ascending",
    calib_beta: float | None = None,
    # --- NEW options ---
    post_add: bool = False,
    post_add_mode: str = "greedy",           # "greedy" | "sample"
    post_add_temperature: float = 1.0,
    post_add_max_steps: int | None = None,
    logprob_reduction: str = "mean",
):
    """
    Decode “heatmap” scores into a valid independent set, mask-aware and batch-parallel.
    Optionally, after conflict removal, greedily/sample-wise add more feasible nodes.

    Args:
        node_logits:   (B, N, 2) logits for each node: [:,:,1] = “in set” score.
        adj_matrix:    (B, N, N) 0/1 adjacency for each graph.
        greedy:        if True, initial per-node decode uses argmax; else sample.
        removal_order: which conflicting nodes to drop: "ascending" or "descending".
        calib_beta:    margin calibration offset used only in greedy initial decode.

        post_add:              if True, try to add more feasible nodes after removals.
        post_add_mode:         "greedy" (argmax over available) or "sample" (categorical).
        post_add_temperature:  temperature for sampling during post-add.
        post_add_max_steps:    optional safety cap on post-add iterations.
        logprob_reduction:     "mean" preserves current PPO scale; "sum" uses total log-prob.

    Returns:
        final_sol:     (B, N)  0/1 final independent set.
        log_p_mean:    (B,)    mean log-prob of *all* choices (initial per-node + post-add steps).
                              (Greedy steps contribute 0.)
        num_removals:  (B,)    how many conflict-removals each graph did.
        num_conflicts: (B,)    how many nodes were in conflict initially.
        init_sol:      (B, N)  0/1 raw per-node sample before conflict repair.
        repaired_sol:  (B, N)  0/1 independent set after repair and before post-add.
        probs:         (B, N)  select probabilities before decoding.
    """
    B, N, _ = node_logits.shape
    device = node_logits.device

    # 1) per-node “in-set” probabilities (for heuristics & post-add ordering)
    probs = F.softmax(node_logits, dim=-1)[..., 1]   # (B, N)

    # 2) initial unconstrained per-node decode + log-prob
    flat_logits = node_logits.reshape(B * N, 2)         # (B*N, 2)
    if greedy:
        if calib_beta is None:
            flat_actions = flat_logits.argmax(dim=1)           # (B*N,)
        else:
            # Apply calibration to the margin (logit1 - logit0)
            margins      = flat_logits[:, 1] - flat_logits[:, 0]
            flat_actions = (margins + calib_beta >= 0).long()  # (B*N,)
        flat_logp = torch.zeros(B * N, device=device)
    else:
        dist       = Categorical(logits=flat_logits)
        flat_actions = dist.sample()                            # (B*N,)
        flat_logp    = dist.log_prob(flat_actions)             # (B*N,)

    init_sol     = flat_actions.view(B, N)                     # (B, N) 0/1 mask
    # sum then mean; later we'll add post-add logp and average over (N + added)
    init_logp_sum = flat_logp.view(B, N).sum(dim=1)            # (B,)
    count_choices = torch.full((B,), float(N), device=device)  # start with N decisions

    # 3) batch-parallel conflict removal
    final_sol    = init_sol.clone()                            # (B, N)
    adj_bool     = adj_matrix.to(torch.bool)                   # (B, N, N)
    num_removals = torch.zeros(B, dtype=torch.int64, device=device)

    # conflicts present initially?
    sel           = final_sol.bool()                           # (B, N)
    any_sel       = (adj_bool & sel.unsqueeze(1)).any(dim=2)   # (B, N)
    conflict_mask = sel & any_sel                              # (B, N)
    num_conflicts = conflict_mask.sum(dim=1)                   # (B,)

    while True:
        sel           = final_sol.bool()
        any_sel       = (adj_bool & sel.unsqueeze(1)).any(dim=2)
        conflict_mask = sel & any_sel

        active = conflict_mask.any(dim=1)                      # which graphs still have conflicts
        if not active.any():
            break

        # pick one node *per active graph* to drop
        masked_p = probs.clone()                               # (B, N)
        if removal_order == "descending":
            # keep only conflicting nodes, set rest = -inf → drop highest-prob conflict
            masked_p[~conflict_mask] = float("-inf")
            drop_idx = masked_p.argmax(dim=1)                  # (B,)
        else:
            # keep only conflicting nodes, set rest = +inf → drop lowest-prob conflict
            masked_p[~conflict_mask] = float("inf")
            drop_idx = masked_p.argmin(dim=1)                  # (B,)

        batch_idx = torch.nonzero(active, as_tuple=True)[0]    # (K ≤ B,)
        to_drop   = drop_idx[batch_idx]                        # (K,)
        final_sol[batch_idx, to_drop] = 0
        num_removals[batch_idx]      += 1

    repaired_sol = final_sol.clone()

    # 4) OPTIONAL: post-add completion — add more feasible nodes
    if post_add:
        # availability: not selected + no selected neighbor
        sel = final_sol.bool()                                 # (B, N)
        has_sel_neigh = (adj_bool & sel.unsqueeze(1)).any(dim=2)  # (B, N)
        available = (~sel) & (~has_sel_neigh)                  # (B, N)

        add_logp_sum = torch.zeros(B, device=device)
        add_count    = torch.zeros(B, device=device)
        steps = 0
        inf = float("-inf")

        while True:
            active = available.any(dim=1)
            if not active.any():
                break
            if (post_add_max_steps is not None) and (steps >= post_add_max_steps):
                break

            bidx = torch.nonzero(active, as_tuple=True)[0]     # (K,)

            # scores over available nodes only
            scores = probs[bidx].clone()                       # (K, N)
            scores[~available[bidx]] = inf

            if post_add_mode == "greedy":
                choice = scores.argmax(dim=1)                  # (K,)
                # greedy contributes 0 to log-prob (to match your convention)
                lp = torch.zeros_like(bidx, dtype=scores.dtype, device=device)
            else:  # "sample"
                # temperature (applies only on available nodes since masked are -inf)
                logits_local = scores / max(post_add_temperature, 1e-8)
                logp_local   = F.log_softmax(logits_local, dim=1)
                dist         = Categorical(logits=logp_local)
                choice       = dist.sample()                   # (K,)
                lp           = logp_local[torch.arange(bidx.size(0), device=device), choice]
                add_logp_sum[bidx] += lp
                add_count[bidx]    += 1

            # commit the pick and update availability
            final_sol[bidx, choice] = 1
            neigh = adj_bool[bidx, choice]                     # (K, N) bool
            available[bidx, choice] = False
            available[bidx] &= ~neigh

            steps += 1

        # merge log-probs: average over total number of choices (N + added)
        init_logp_sum = init_logp_sum + add_logp_sum
        count_choices = count_choices + add_count

    # 5) Final mean log-prob
    # (avoid div by 0: if a graph made 0 choices in post-add & greedy initial, denominator is still N)
    if logprob_reduction == "sum":
        log_p_mean = init_logp_sum.nan_to_num(0.0)
    elif logprob_reduction == "mean":
        log_p_mean = (init_logp_sum / count_choices.clamp_min(1.0)).nan_to_num(0.0)
    else:
        raise ValueError(f"Invalid logprob_reduction: {logprob_reduction}")

    return final_sol, log_p_mean, num_removals, num_conflicts, init_sol, repaired_sol, probs


@torch.no_grad()
def random_greedy_partial(adj: torch.Tensor, n_solutions: int, skip_p: float = 0.3) -> torch.Tensor:
    """
    Random-order greedy, but intentionally skip some allowed picks.
    Feasible (independent), typically non-maximal (so 1-add improvements exist).

    adj: (N,N) 0/1 symmetric tensor
    returns: (B,N) int {0,1}
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1)
    N = adj.size(0)
    device = adj.device
    B = n_solutions

    perm = torch.argsort(torch.rand(B, N, device=device), dim=1)  # random order per solution
    selected  = torch.zeros(B, N, dtype=torch.bool, device=device)
    forbidden = torch.zeros_like(selected)
    rows = torch.arange(B, device=device)

    for step in range(N):
        v = perm[:, step]                        # (B,)
        allow = ~forbidden[rows, v]             # which sols could take v
        # deliberately skip some allowed vertices
        take  = allow & (torch.rand(B, device=device) > skip_p)
        if take.any():
            selected[rows[take], v[take]] = True
            neigh = adj[v[take]].bool()         # (take_count, N)
            forbidden[take] |= neigh
            forbidden[rows[take], v[take]] = True

    return selected.to(torch.int)


def random_greedy_initialization(adj_matrix: torch.Tensor, n_solutions) -> torch.Tensor:
    """
    Greedily build a random independent set using the adjacency matrix.
    adj_matrix  (N, N)   symmetric 0/1 adjacency
    n_solutions (int) number of solutions to generate
    Returns:
        Solution with selected nodes set to 1
    """
    N = adj_matrix.size(0)
    device = adj_matrix.device

    # Make a random permutation for each solution by argsorting random keys
    keys = torch.rand(n_solutions, N, device=device)
    perm = keys.argsort(dim=1)  # shape (n_solutions, N)

    selected  = torch.zeros(n_solutions, N, dtype=torch.bool, device=device)
    forbidden = torch.zeros_like(selected)
    batch_idx = torch.arange(n_solutions, device=device)

    for step in range(N):
        nodes = perm[:, step]           # candidate node for each solution
        is_forb = forbidden[batch_idx, nodes]
        allow = ~is_forb               # which sols can pick their candidate

        # mark selected nodes
        selected[batch_idx[allow], nodes[allow]] = True
        neigh_mask = adj_matrix[nodes].to(torch.bool)
        forbidden[allow] |= neigh_mask[allow]
        forbidden[batch_idx[allow], nodes[allow]] = True  # also forbid the node itself

    return selected.to(torch.int)


def compute_upper_bounds(adj):
    B, N, _ = adj.shape
    # 1) expected MIS size via random‐greedy (Caro–Wei):
    expected = compute_expected_mis_size(adj)
    ub = compute_matching_upper_bound_batched(adj)

    return expected, ub


def compute_expected_mis_size(adj):
    deg = adj.sum(dim=-1)
    return (1.0 / (deg + 1.0)).sum(dim=-1)


def compute_matching_upper_bound_batched(adj):
    B, N, _ = adj.shape
    device = adj.device
    edge_i, edge_j = torch.triu_indices(N, N, offset=1, device=device)
    if edge_i.numel() == 0:
        return adj.new_full((B,), float(N))

    edge_exists = adj[:, edge_i, edge_j] > 0
    max_edges = int(edge_exists.sum(dim=1).max().item())
    if max_edges == 0:
        return adj.new_full((B,), float(N))

    priorities = torch.rand(B, edge_i.numel(), device=device)
    priorities = priorities.masked_fill(~edge_exists, float('inf'))
    order = priorities.argsort(dim=1)[:, :max_edges]

    matched = torch.zeros(B, N, dtype=torch.bool, device=device)
    match_count = adj.new_zeros(B, dtype=torch.float)
    batch_idx = torch.arange(B, device=device)

    for step in range(max_edges):
        edge_idx = order[:, step]
        valid = edge_exists.gather(1, edge_idx.unsqueeze(1)).squeeze(1)
        i = edge_i[edge_idx]
        j = edge_j[edge_idx]
        can_match = valid & ~matched[batch_idx, i] & ~matched[batch_idx, j]
        if can_match.any():
            active_batch = batch_idx[can_match]
            matched[active_batch, i[can_match]] = True
            matched[active_batch, j[can_match]] = True
            match_count[can_match] += 1.0

    return N - match_count
