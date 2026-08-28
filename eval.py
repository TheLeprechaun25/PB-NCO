import time
import numpy as np
import torch
import pickle
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from torch import autocast
from nets.models import GraphNIModel, MCNCModel, MISNCModel
from env.MCEnv import MCEnv
from env.MISEnv import MISEnv
from utils.utils import load_test_data
from utils.env_utils import pop_diversity
from args.eval_args import get_args


class Evaluator:
    def __init__(self, params, ni_params, nc_params):
        # Set parameters
        self.params = params
        self.verbose = self.params['verbose']

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.params['device'] = device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)

        # Set seed
        if self.params['seed'] is not None:
            torch.manual_seed(self.params['seed'])
            np.random.seed(self.params['seed'])

        # Set environment
        self.ni_model = GraphNIModel(**ni_params).to(device) if ni_params else None
        if self.params['problem'] == 'mc':
            Env = MCEnv
            self.nc_model = MCNCModel(**nc_params).to(device) if nc_params else None
        elif self.params['problem'] == 'mis':
            Env = MISEnv
            self.nc_model = MISNCModel(**nc_params).to(device) if nc_params else None
        else:
            raise ValueError(f"Invalid problem: {self.params['problem']}")

        if self.ni_model:
            # Restore ni model weights
            checkpoint = torch.load(self.params['ni_model_load_path'], map_location=self.device, weights_only=False)
            self.ni_model.load_state_dict(checkpoint['model_state_dict'])
            if self.verbose:
                print(f'NI model loaded from {self.params["ni_model_load_path"]}')

        if self.nc_model:
            # Restore nc model weights
            checkpoint = torch.load(self.params['nc_model_load_path'], map_location=self.device, weights_only=False)
            self.nc_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if self.verbose:
                print(f'NC model loaded from {self.params["nc_model_load_path"]}')

        self.eval_mode = self._get_eval_mode()

        # Compile
        if self.params['compile'] and self.device.type != 'cpu':
            if self.ni_model:
                self.compiled_ni_model = torch.compile(self.ni_model)
            if self.nc_model:
                self.compiled_nc_model = torch.compile(self.nc_model)

        # Testing environment
        if self.nc_model:
            initializer = self.compiled_nc_model if self.params['compile'] else self.nc_model
        else:
            initializer = None

        self.test_env = Env(self.params, device, testing=True, compute_metrics=False, initializer=initializer)

        # Load eval data
        self.test_graphs = load_test_data(self.params['problem'], self.params['eval_graph_types'], self.params['num_eval_graphs'])
        if self.params['eval_graph_idx'] > -1:
            self.test_graphs = [[self.test_graphs[i][self.params['eval_graph_idx']]] for i in range(len(self.test_graphs))]

        # Diversity function
        self.diversity_fn = pop_diversity(self.params['distance_metric'])

        if self.params['save_results']:
            # Build the run-directory path
            debug_suffix = "_debug" if self.params.get("debug") else ""
            model_path = self.params['ni_model_load_path'] or self.params['nc_model_load_path']
            model_name = Path(model_path).stem
            exp_root = Path("runs/eval") / f"{self.params['problem']}"

            if self.params.get('run_name'):  # single shared folder for all jobs
                run_dir = exp_root / f"{self.params['run_name']}{debug_suffix}"
            else:
                run_dir = exp_root / f"{model_name}_pop{self.params['pop_size']}_topk{self.params['topk']}_init{self.params['initialization']}_steps{self.params['test_max_steps_multiplier']}{debug_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{self.params['seed']}"

            run_dir.mkdir(parents=True, exist_ok=True)
            self.run_path = run_dir

    def _get_eval_mode(self):
        if self.ni_model is not None and self.nc_model is not None:
            return 'cni_cnc'
        if self.ni_model is not None:
            return 'cni'
        if self.nc_model is not None:
            return 'cnc'
        raise ValueError("No model loaded.")

    @torch.no_grad()
    def run_tests(self, compute_diversity=True):
        if self.eval_mode == 'cnc':
            return self.run_cnc_tests(compute_diversity)
        return self.run_cni_tests(compute_diversity)

    @torch.no_grad()
    def run_cni_tests(self, compute_diversity=True):
        self.ni_model.eval()
        if self.test_env.initializer is not None:
            self.test_env.initializer.eval()
        print("=== Starting test ===")
        test_results = {}
        for i, (cur_test_graph_list, graph_type) in enumerate(zip(self.test_graphs, self.params['eval_graph_types'])):
            sum_revisited = 0.0
            sum_total_revisited = 0.0
            sum_self_mem_perc = 0.0
            best_obj_values = []
            best_solutions = []
            all_time_best_obj_values = []
            diversity_values = []
            elapsed_times = []
            for j, cur_test_graph in enumerate(cur_test_graph_list):
                if len(cur_test_graph.size()) == 2:  # Different sizes
                    if self.verbose:
                        print(f"\nSingle graph inference mode {j}/{len(cur_test_graph_list)}.")
                    problem_size, _ = cur_test_graph.size()
                    test_batch_size = 1
                    cur_test_graph = cur_test_graph.unsqueeze(0)
                else:
                    if self.verbose:
                        print("\nBatch inference mode.")
                    test_batch_size, problem_size, _ = cur_test_graph.size()

                pop_size = self.params['pop_size']

                self.test_env.problem_size = problem_size
                self.ni_model.edge_embeddings_computed = False

                state, done = self.test_env.reset(test_batch_size, pop_size, test_graph=cur_test_graph.to(self.device), seed=self.params['seed'])

                if self.verbose:
                    print(f"Testing with {test_batch_size} graphs of size {problem_size}. {pop_size} agents per graph. "
                          f"Max steps: {self.test_env.max_iterations}. Patience: {self.test_env.patience}.")

                steps = 0
                elapsed_time = [0.0]
                all_time_obj = [self.test_env.best_objective_values.reshape(test_batch_size, pop_size).max(dim=1).values.cpu().numpy().tolist()]
                diversity_vals = []
                start_time = time.time()
                while not done:
                    with autocast(self.device.type, enabled=self.params.get('amp', False)):
                        logits = self.compiled_ni_model(state) if self.params['compile'] else self.ni_model(state)

                    topk_actions = torch.topk(logits, self.params['topk'], dim=1).indices

                    for k in range(self.params['topk']):
                        steps += 1

                        actions = topk_actions[:, k]

                        if k > 0 and state.mask is not None:
                            # Check whether all actions are valid or masked by state.masks
                            exit_loop = False
                            for s in range(pop_size):
                                if state.mask[s, actions[s]] == 1:
                                    exit_loop = True
                            if exit_loop:
                                break

                        # Perform the action
                        state, _, done = self.test_env.step(actions)

                        # Multi start
                        if self.params['multi_start']:
                            #  exploration weight schedule: w = w_start * (1 - (t/T))^phi
                            w = 1.0 * (1 - (steps / self.test_env.max_iterations)) ** self.params['phi']
                            state = self.test_env.reset_non_improving_solutions(exploration_weight=w)

                        # Gather data
                        elapsed_time.append(time.time() - start_time)
                        all_time_obj.append(self.test_env.best_objective_values.reshape(test_batch_size, pop_size).max(dim=1).values.cpu().numpy().tolist())

                    # Compute diversity
                    if compute_diversity:
                        S = state.ising_solutions.reshape(test_batch_size, pop_size, problem_size)
                        cur_div = self.diversity_fn(S, cur_test_graph)  # scalar tensor in [0,1]
                        diversity_vals.append(float(cur_div.detach().item()))

                    # End while not done or timeout
                    if elapsed_time[-1] >= self.params.get('max_time_per_instance', float('inf')):
                        if self.verbose:
                            print(f"Max time per instance {self.params['max_time_per_instance']}s reached. Stopping inference.")
                        break

                # Store current best objective values
                best_objective_values = self.test_env.best_objective_values.reshape(test_batch_size, pop_size)
                all_best_obj_values, best_indices = best_objective_values.max(dim=1)
                all_avg_obj_values = best_objective_values.mean(dim=1)

                # Store results
                best_obj_values.extend(all_best_obj_values.cpu().numpy().tolist())
                best_sols = self.test_env.best_solutions.reshape(test_batch_size, pop_size, problem_size)
                best_solutions.extend(best_sols[torch.arange(test_batch_size), best_indices].cpu().numpy().tolist())

                # Compute objective values
                all_best_obj_values = all_best_obj_values.mean().item()
                all_avg_obj_values = all_avg_obj_values.mean().item()

                # Print results
                if self.verbose:
                    total_time = elapsed_time[-1]
                    print(f"Best obj: {all_best_obj_values:.3f}. Average obj: {all_avg_obj_values:.3f}. "
                          f"Avg steps: {steps:.2f}. Tot time: {total_time:.2f}s ({total_time / test_batch_size:.2f}s/instance).")

                elapsed_times.append(elapsed_time)
                all_time_best_obj_values.append(all_time_obj)
                diversity_values.append(diversity_vals)

            n_graph_batches = len(cur_test_graph_list)
            test_results[f'{graph_type}/Revisited'] = sum_revisited / n_graph_batches
            test_results[f'{graph_type}/Total Revisited'] = sum_total_revisited / n_graph_batches
            test_results[f'{graph_type}/Self Memory'] = sum_self_mem_perc / n_graph_batches
            test_results[f'{graph_type}/Objective Values'] = best_obj_values
            test_results[f'{graph_type}/Solutions'] = best_solutions
            test_results[f'{graph_type}/Elapsed Times'] = elapsed_times
            test_results[f'{graph_type}/All Time Best Objective Values'] = all_time_best_obj_values
            test_results[f'{graph_type}/Diversity Values'] = diversity_values
            print(f"\nOverall best objective value: {np.mean(best_obj_values):.3f}")

        return test_results

    @torch.no_grad()
    def run_cnc_tests(self, compute_diversity=True):
        self.test_env.initializer.eval()
        if self.params.get('cnc_calibrate_threshold', False):
            return self.run_cnc_threshold_calibration_tests(compute_diversity)
        print(f"=== Starting cNC {self.params['cnc_eval_mode']} test ===")
        if self.params['cnc_eval_mode'] == 'greedy_once' and self.params['problem'] != 'mis':
            print(f"Greedy one-shot archive context: {self._resolve_cnc_greedy_once_archive_context()}")
        test_results = {}
        for cur_test_graph_list, graph_type in zip(self.test_graphs, self.params['eval_graph_types']):
            best_obj_values = []
            best_solutions = []
            all_time_best_obj_values = []
            diversity_values = []
            elapsed_times = []
            for j, cur_test_graph in enumerate(cur_test_graph_list):
                if len(cur_test_graph.size()) == 2:  # Different sizes
                    if self.verbose:
                        print(f"\nSingle graph inference mode {j}/{len(cur_test_graph_list)}.")
                    problem_size, _ = cur_test_graph.size()
                    test_batch_size = 1
                    cur_test_graph = cur_test_graph.unsqueeze(0)
                else:
                    if self.verbose:
                        print("\nBatch inference mode.")
                    test_batch_size, problem_size, _ = cur_test_graph.size()

                self.test_env.problem_size = problem_size

                start_time = time.time()
                state, _ = self.test_env.reset(
                    test_batch_size,
                    1,
                    test_graph=cur_test_graph.to(self.device),
                    train_nc=True,
                )
                if self.params['cnc_eval_mode'] == 'greedy_once':
                    archive_solutions, archive_obj, elapsed_time, all_time_obj, diversity_vals = self._run_greedy_once(
                        state,
                        start_time,
                        compute_diversity,
                        archive_seed=self._cnc_archive_seed(graph_type, j),
                        threshold=self.params.get('cnc_threshold', 0.5),
                    )
                elif self.params['cnc_eval_mode'] == 'cnc_pop':
                    archive_solutions, archive_obj, elapsed_time, all_time_obj, diversity_vals = self._run_cnc_population(
                        state,
                        start_time,
                        compute_diversity,
                        archive_seed=self._cnc_archive_seed(graph_type, j),
                    )
                elif self.params['cnc_eval_mode'] == 'guided_min_degree':
                    archive_solutions, archive_obj, elapsed_time, all_time_obj, diversity_vals = self._run_guided_min_degree(
                        state,
                        start_time,
                        compute_diversity,
                    )
                else:
                    raise ValueError(f"Invalid cNC eval mode: {self.params['cnc_eval_mode']}")

                all_best_obj_values, best_indices = archive_obj.max(dim=1)
                all_avg_obj_values = archive_obj.mean(dim=1)
                best_obj_values.extend(all_best_obj_values.cpu().numpy().tolist())
                best_solutions.extend(archive_solutions[torch.arange(test_batch_size), best_indices].cpu().numpy().tolist())
                elapsed_times.append(elapsed_time)
                all_time_best_obj_values.append(all_time_obj)
                diversity_values.append(diversity_vals)

                if self.verbose:
                    total_time = elapsed_time[-1]
                    print(f"Best obj: {all_best_obj_values.mean().item():.3f}. "
                          f"Average obj: {all_avg_obj_values.mean().item():.3f}. "
                          f"Tot time: {total_time:.2f}s ({total_time / test_batch_size:.2f}s/instance).")

            test_results[f'{graph_type}/Revisited'] = 0.0
            test_results[f'{graph_type}/Total Revisited'] = 0.0
            test_results[f'{graph_type}/Self Memory'] = 0.0
            test_results[f'{graph_type}/Objective Values'] = best_obj_values
            test_results[f'{graph_type}/Solutions'] = best_solutions
            test_results[f'{graph_type}/Elapsed Times'] = elapsed_times
            test_results[f'{graph_type}/All Time Best Objective Values'] = all_time_best_obj_values
            test_results[f'{graph_type}/Diversity Values'] = diversity_values
            print(f"\nOverall best objective value: {np.mean(best_obj_values):.3f}")

        return test_results

    @torch.no_grad()
    def run_cnc_threshold_calibration_tests(self, compute_diversity=True):
        if self.params['cnc_eval_mode'] != 'greedy_once':
            raise ValueError("cNC threshold calibration is only defined for --cnc_eval_mode greedy_once.")

        thresholds = parse_cnc_threshold_grid(self.params['cnc_threshold_grid'])
        print("=== Starting cNC greedy_once threshold calibration test ===")
        print(
            f"Threshold grid: {format_cnc_thresholds(thresholds)} "
            f"({len(thresholds)} candidates)."
        )
        print(f"Greedy one-shot archive context: {self._resolve_cnc_greedy_once_archive_context()}")
        test_results = {}

        for cur_test_graph_list, graph_type in zip(self.test_graphs, self.params['eval_graph_types']):
            graphs = self._flatten_graph_list(cur_test_graph_list)
            calibration_indices, test_indices = self._calibration_split_indices(len(graphs))
            calibration_graphs = [graphs[i] for i in calibration_indices]
            test_graphs = [graphs[i] for i in test_indices]
            print(
                f"\n{graph_type}: {len(graphs)} instances -> "
                f"{len(calibration_graphs)} calibration / {len(test_graphs)} held-out."
            )
            print(f"{graph_type}: calibration indices {calibration_indices}")
            print(f"{graph_type}: held-out indices {test_indices}")

            calibration_threshold_means = []
            heldout_threshold_means = []
            total_threshold_means = []
            for threshold in thresholds:
                _, calibration_obj, _, _, _ = self._evaluate_cnc_greedy_once_graphs(
                    calibration_graphs,
                    threshold=threshold,
                    compute_diversity=False,
                )
                _, heldout_obj, _, _, _ = self._evaluate_cnc_greedy_once_graphs(
                    test_graphs,
                    threshold=threshold,
                    compute_diversity=False,
                )
                calibration_mean = float(np.mean(calibration_obj))
                heldout_mean = float(np.mean(heldout_obj))
                total_mean = float(np.mean(calibration_obj + heldout_obj))
                calibration_threshold_means.append(calibration_mean)
                heldout_threshold_means.append(heldout_mean)
                total_threshold_means.append(total_mean)
                print(
                    f"{graph_type}: threshold {threshold:.4f} -> "
                    f"calib mean {calibration_mean:.3f} "
                    f"(min {np.min(calibration_obj):.3f}, max {np.max(calibration_obj):.3f}); "
                    f"held-out mean {heldout_mean:.3f} "
                    f"(min {np.min(heldout_obj):.3f}, max {np.max(heldout_obj):.3f}); "
                    f"total mean {total_mean:.3f}"
                )

            best_mean = max(calibration_threshold_means)
            best_candidates = [i for i, value in enumerate(calibration_threshold_means) if np.isclose(value, best_mean)]
            best_grid_idx = min(best_candidates, key=lambda i: abs(thresholds[i] - 0.5))
            best_threshold = thresholds[best_grid_idx]
            baseline_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.5))

            best_solutions, best_obj_values, elapsed_times, all_time_best_obj_values, diversity_values = self._evaluate_cnc_greedy_once_graphs(
                test_graphs,
                threshold=best_threshold,
                compute_diversity=compute_diversity,
            )

            test_results[f'{graph_type}/Revisited'] = 0.0
            test_results[f'{graph_type}/Total Revisited'] = 0.0
            test_results[f'{graph_type}/Self Memory'] = 0.0
            test_results[f'{graph_type}/Objective Values'] = best_obj_values
            test_results[f'{graph_type}/Solutions'] = best_solutions
            test_results[f'{graph_type}/Elapsed Times'] = elapsed_times
            test_results[f'{graph_type}/All Time Best Objective Values'] = all_time_best_obj_values
            test_results[f'{graph_type}/Diversity Values'] = diversity_values
            test_results[f'{graph_type}/Calibration Threshold'] = best_threshold
            test_results[f'{graph_type}/Calibration Candidate Thresholds'] = thresholds
            test_results[f'{graph_type}/Calibration Candidate Mean Objective Values'] = calibration_threshold_means
            test_results[f'{graph_type}/Heldout Candidate Mean Objective Values'] = heldout_threshold_means
            test_results[f'{graph_type}/Total Candidate Mean Objective Values'] = total_threshold_means
            test_results[f'{graph_type}/Calibration Default Threshold Mean Objective Value'] = calibration_threshold_means[baseline_idx]
            test_results[f'{graph_type}/Heldout Default Threshold Mean Objective Value'] = heldout_threshold_means[baseline_idx]
            test_results[f'{graph_type}/Total Default Threshold Mean Objective Value'] = total_threshold_means[baseline_idx]
            test_results[f'{graph_type}/Calibration Indices'] = calibration_indices
            test_results[f'{graph_type}/Test Indices'] = test_indices

            print(
                f"\n{graph_type}: selected threshold {best_threshold:.4f} "
                f"(calib mean {calibration_threshold_means[best_grid_idx]:.3f}; "
                f"held-out mean {heldout_threshold_means[best_grid_idx]:.3f}; "
                f"total mean {total_threshold_means[best_grid_idx]:.3f}; "
                f"nearest 0.5 calib mean {calibration_threshold_means[baseline_idx]:.3f})."
            )
            print(
                f"{graph_type}: held-out best objective value: {np.mean(best_obj_values):.3f} "
                f"(min {np.min(best_obj_values):.3f}, max {np.max(best_obj_values):.3f})"
            )

        return test_results

    def _run_cnc_greedy_once(self, state, start_time, compute_diversity, visited_solutions, threshold=0.5):
        state = self._make_cnc_state(state, visited_solutions, exploration_weight=0.0, greedy_hint=True)
        logits = self.test_env.initializer(state)
        archive_solutions, archive_obj, _, _, _, _ = self.test_env.heatmap_inference(
            logits,
            feasibility_decoder=self._cnc_feasibility_decoder(),
            visited_solutions=visited_solutions,
            n_rollouts=1,
            training=True,
            greedy=True,
            calib_beta=threshold_to_cnc_beta(threshold),
        )
        elapsed_time = [time.time() - start_time]
        all_time_obj = [archive_obj.max(dim=1).values.cpu().numpy().tolist()]
        diversity_vals = self._archive_diversity(archive_solutions, state.graph, compute_diversity)
        return archive_solutions.float(), archive_obj.float(), elapsed_time, all_time_obj, diversity_vals

    def _run_greedy_once(self, state, start_time, compute_diversity, archive_seed, threshold=0.5):
        if self.params['problem'] == 'mis':
            return self._run_guided_min_degree(state, start_time, compute_diversity)

        visited_solutions = self._make_cnc_greedy_once_archive(
            state,
            archive_seed=archive_seed,
        )
        return self._run_cnc_greedy_once(
            state,
            start_time,
            compute_diversity,
            visited_solutions=visited_solutions,
            threshold=threshold,
        )

    def _run_guided_min_degree(self, state, start_time, compute_diversity):
        state = self._make_cnc_state(
            state,
            self._make_zero_archive(state.batch_size, state.problem_size, self.params['n_visited_solutions']),
            exploration_weight=0.0,
            greedy_hint=True,
        )
        logits = self.test_env.initializer(state)
        probs = torch.softmax(logits, dim=-1)[..., 1]
        solutions = self._guided_min_degree_decode(state.graph, probs)
        obj = solutions.sum(dim=1).float().unsqueeze(1)
        archive_solutions = solutions.unsqueeze(1).float()
        elapsed_time = [time.time() - start_time]
        all_time_obj = [obj.max(dim=1).values.cpu().numpy().tolist()]
        diversity_vals = self._archive_diversity(archive_solutions, state.graph, compute_diversity)
        return archive_solutions, obj, elapsed_time, all_time_obj, diversity_vals

    def _guided_min_degree_decode(self, graph, probs):
        batch_size, problem_size, _ = graph.shape
        solutions = torch.zeros(batch_size, problem_size, dtype=torch.float32, device=self.device)
        pool_frac = self.params.get('cnc_guided_pool_frac', 0.02)
        alpha = self.params.get('cnc_guided_alpha', 1.0)

        for b in range(batch_size):
            adj = graph[b].bool()
            available = torch.ones(problem_size, dtype=torch.bool, device=self.device)
            selected = torch.zeros(problem_size, dtype=torch.bool, device=self.device)

            while available.any():
                avail_idx = torch.where(available)[0]
                residual_deg = adj[available][:, available].sum(dim=1).float()

                if pool_frac and pool_frac > 0:
                    k = max(1, int(np.ceil(pool_frac * avail_idx.numel())))
                    pool_local = torch.topk(-residual_deg, k=k).indices
                    pool_idx = avail_idx[pool_local]
                    choice = pool_idx[probs[b, pool_idx].argmax()]
                else:
                    score = residual_deg - alpha * probs[b, avail_idx]
                    choice = avail_idx[score.argmin()]

                selected[choice] = True
                available[choice] = False
                available &= ~adj[choice]

            solutions[b] = selected.float()

        return solutions

    def _evaluate_cnc_greedy_once_graphs(self, graphs, threshold, compute_diversity):
        best_obj_values = []
        best_solutions = []
        all_time_best_obj_values = []
        diversity_values = []
        elapsed_times = []

        for idx, graph_batch in enumerate(self._batch_graphs_by_size(graphs)):
            if len(graph_batch.size()) == 2:
                problem_size, _ = graph_batch.size()
                test_batch_size = 1
                cur_test_graph = graph_batch.unsqueeze(0)
            else:
                test_batch_size, problem_size, _ = graph_batch.size()
                cur_test_graph = graph_batch

            self.test_env.problem_size = problem_size
            start_time = time.time()
            state, _ = self.test_env.reset(
                test_batch_size,
                1,
                test_graph=cur_test_graph.to(self.device),
                train_nc=True,
            )
            archive_solutions, archive_obj, elapsed_time, all_time_obj, diversity_vals = self._run_greedy_once(
                state,
                start_time,
                compute_diversity,
                archive_seed=self._cnc_archive_seed('calibration', idx),
                threshold=threshold,
            )

            all_best_obj_values, best_indices = archive_obj.max(dim=1)
            best_obj_values.extend(all_best_obj_values.cpu().numpy().tolist())
            best_solutions.extend(archive_solutions[torch.arange(test_batch_size), best_indices].cpu().numpy().tolist())
            elapsed_times.append(elapsed_time)
            all_time_best_obj_values.append(all_time_obj)
            diversity_values.append(diversity_vals)

        return best_solutions, best_obj_values, elapsed_times, all_time_best_obj_values, diversity_values

    def _flatten_graph_list(self, graph_list):
        graphs = []
        for graph_tensor in graph_list:
            if len(graph_tensor.size()) == 2:
                graphs.append(graph_tensor)
            elif len(graph_tensor.size()) == 3:
                graphs.extend([graph_tensor[i] for i in range(graph_tensor.size(0))])
            else:
                raise ValueError(f"Expected 2D or 3D graph tensor, got shape {tuple(graph_tensor.size())}.")
        return graphs

    def _batch_graphs_by_size(self, graphs):
        batches_by_shape = {}
        for graph in graphs:
            batches_by_shape.setdefault(tuple(graph.size()), []).append(graph)
        return [torch.stack(cur_graphs, dim=0) for cur_graphs in batches_by_shape.values()]

    def _calibration_split_indices(self, n_instances):
        if n_instances < 2:
            raise ValueError("Need at least two eval instances for calibration/test splitting.")
        if self.params.get('cnc_calibration_percent') is not None:
            fraction = self.params['cnc_calibration_percent'] / 100.0
        else:
            fraction = self.params.get('cnc_calibration_fraction', 0.5)
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "--cnc_calibration_fraction must be in (0, 1), or "
                "--cnc_calibration_percent must be in (0, 100)."
            )
        split_seed = self.params.get('cnc_calibration_split_seed')
        if split_seed is None:
            split_seed = self.params.get('seed', 42)
        rng = np.random.default_rng(split_seed)
        indices = rng.permutation(n_instances).tolist()
        n_calibration = int(round(n_instances * fraction))
        n_calibration = min(max(n_calibration, 1), n_instances - 1)
        return sorted(indices[:n_calibration]), sorted(indices[n_calibration:])

    def _run_cnc_population(self, state, start_time, compute_diversity, archive_seed):
        archive_size = self._cnc_archive_size()
        visited_solutions = self._make_cnc_presample_archive(state, archive_size, archive_seed)
        state = self._make_cnc_state(state, visited_solutions, exploration_weight=0.0, greedy_hint=True)
        logits = self.test_env.initializer(state)
        archive_solutions, archive_obj, _, _, _, _ = self.test_env.heatmap_inference(
            logits,
            feasibility_decoder=self._cnc_feasibility_decoder(),
            visited_solutions=visited_solutions,
            n_rollouts=archive_size,
            training=True,
            greedy=True,
        )
        archive_solutions = archive_solutions.float()
        archive_obj = archive_obj.float()
        archive = archive_solutions.permute(0, 2, 1).contiguous()

        elapsed_time = [time.time() - start_time]
        all_time_obj = [archive_obj.max(dim=1).values.cpu().numpy().tolist()]
        diversity_vals = self._archive_diversity(archive_solutions, state.graph, compute_diversity)
        best_so_far = archive_obj.max(dim=1).values
        best_solutions_so_far = archive_solutions[
            torch.arange(state.batch_size, device=self.device),
            archive_obj.argmax(dim=1),
        ]

        weight_mode = self.params.get('cnc_pop_weight_mode', 'sweep')
        if weight_mode == 'sweep':
            weights = parse_cnc_weights(self.params['cnc_pop_weights'])
            rollouts_per_weight = archive_size
        elif weight_mode == 'rollout_linspace':
            weights = torch.linspace(
                self.params['cnc_pop_weight_min'],
                self.params['cnc_pop_weight_max'],
                steps=archive_size,
                device=self.device,
            ).tolist()
            rollouts_per_weight = 1
        elif weight_mode == 'grouped_linspace':
            num_weights = self.params['cnc_pop_num_weights']
            if num_weights < 1:
                raise ValueError("cnc_pop_num_weights must be at least 1.")
            rollouts_per_weight = self.params['cnc_pop_rollouts_per_weight']
            if rollouts_per_weight == 0:
                rollouts_per_weight = int(np.ceil(archive_size / num_weights))
            if rollouts_per_weight < 1:
                raise ValueError("cnc_pop_rollouts_per_weight must be 0 or at least 1.")
            weights = torch.linspace(
                self.params['cnc_pop_weight_min'],
                self.params['cnc_pop_weight_max'],
                steps=num_weights,
                device=self.device,
            ).tolist()
        else:
            raise ValueError(f"Invalid cNC population weight mode: {weight_mode}")
        if self.params['cnc_pop_keep_policy'] == 'new' and len(weights) * rollouts_per_weight < archive_size:
            raise ValueError(
                "cnc_pop_keep_policy='new' requires at least archive_size new candidates per update; "
                f"got {len(weights) * rollouts_per_weight} candidates for archive_size={archive_size}."
            )

        for _ in range(self.params['cnc_pop_generations']):
            if weight_mode == 'sweep':
                for w in weights:
                    state = self._make_cnc_state(state, archive, exploration_weight=w, greedy_hint=False)
                    logits = self.test_env.initializer(state)
                    new_solutions, new_obj, _, _, _, _ = self.test_env.heatmap_inference(
                        logits,
                        feasibility_decoder=self._cnc_feasibility_decoder(),
                        visited_solutions=archive,
                        n_rollouts=archive_size,
                        training=True,
                        greedy=False,
                    )
                    archive_solutions, archive_obj = self._update_cnc_population_archive(
                        archive_solutions,
                        archive_obj,
                        new_solutions.float(),
                        new_obj.float(),
                        archive_size,
                    )
                    archive = archive_solutions.permute(0, 2, 1).contiguous()
                    best_so_far, best_solutions_so_far = self._update_cnc_best_so_far(
                        archive_solutions,
                        archive_obj,
                        best_so_far,
                        best_solutions_so_far,
                    )
                    self._record_cnc_population_progress(
                        elapsed_time,
                        all_time_obj,
                        diversity_vals,
                        start_time,
                        best_so_far,
                        archive_solutions,
                        state.graph,
                        compute_diversity,
                    )
            elif weight_mode in {'rollout_linspace', 'grouped_linspace'}:
                new_solutions, new_obj = self._run_cnc_weighted_rollouts_batched(
                    state,
                    archive,
                    weights,
                    rollouts_per_weight=rollouts_per_weight,
                )
                archive_solutions, archive_obj = self._update_cnc_population_archive(
                    archive_solutions,
                    archive_obj,
                    new_solutions,
                    new_obj,
                    archive_size,
                )
                archive = archive_solutions.permute(0, 2, 1).contiguous()
                best_so_far, best_solutions_so_far = self._update_cnc_best_so_far(
                    archive_solutions,
                    archive_obj,
                    best_so_far,
                    best_solutions_so_far,
                )
                self._record_cnc_population_progress(
                    elapsed_time,
                    all_time_obj,
                    diversity_vals,
                    start_time,
                    best_so_far,
                    archive_solutions,
                    state.graph,
                    compute_diversity,
                )

        if self.params['cnc_pop_keep_policy'] == 'new':
            archive_solutions = best_solutions_so_far.unsqueeze(1)
            archive_obj = best_so_far.unsqueeze(1)
        return archive_solutions.float(), archive_obj.float(), elapsed_time, all_time_obj, diversity_vals

    def _run_cnc_weighted_rollouts_batched(self, state, archive, weights, rollouts_per_weight=1):
        batch_size = state.batch_size
        n_weights = len(weights)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        expanded_archive = archive.unsqueeze(1).expand(batch_size, n_weights, state.problem_size, archive.size(-1)) \
            .reshape(batch_size * n_weights, state.problem_size, archive.size(-1))
        expanded_state = self._make_cnc_state(
            replace(
                state,
                batch_size=batch_size * n_weights,
                pop_size=1,
                graph=state.graph.unsqueeze(1).expand(batch_size, n_weights, state.problem_size, state.problem_size)
                .reshape(batch_size * n_weights, state.problem_size, state.problem_size),
                extra_node_feats=None if state.extra_node_feats is None else state.extra_node_feats.unsqueeze(1)
                .expand(batch_size, n_weights, state.problem_size, state.extra_node_feats.size(-1))
                .reshape(batch_size * n_weights, state.problem_size, state.extra_node_feats.size(-1)),
            ),
            expanded_archive,
            expanded_1d=weight_tensor.unsqueeze(0).expand(batch_size, n_weights).reshape(batch_size * n_weights),
            greedy_hint=False,
        )

        old_batch_size = self.test_env.batch_size
        old_batch_pop_range = self.test_env.batch_pop_range
        old_state = self.test_env.state
        old_baseline = getattr(self.test_env, 'baseline', None)
        old_boost = getattr(self.test_env, 'boost', None)
        old_expected = getattr(self.test_env, 'expected', None)
        old_upper_bound = getattr(self.test_env, 'upper_bound', None)

        self.test_env.batch_size = batch_size * n_weights
        self.test_env.batch_pop_range = torch.arange(batch_size * n_weights, device=self.device)
        self.test_env.state = expanded_state
        if old_baseline is not None:
            self.test_env.baseline = old_baseline.unsqueeze(1).expand(batch_size, n_weights).reshape(batch_size * n_weights)
        if old_boost is not None:
            self.test_env.boost = old_boost.unsqueeze(1).expand(batch_size, n_weights).reshape(batch_size * n_weights)
        if old_expected is not None:
            self.test_env.expected = old_expected.unsqueeze(1).expand(batch_size, n_weights).reshape(batch_size * n_weights)
        if old_upper_bound is not None:
            self.test_env.upper_bound = old_upper_bound.unsqueeze(1).expand(batch_size, n_weights).reshape(batch_size * n_weights)

        try:
            logits = self.test_env.initializer(expanded_state)
            new_solutions, new_obj, _, _, _, _ = self.test_env.heatmap_inference(
                logits,
                feasibility_decoder=self._cnc_feasibility_decoder(),
                visited_solutions=expanded_state.visited_solutions,
                n_rollouts=rollouts_per_weight,
                training=True,
                greedy=False,
            )
        finally:
            self.test_env.batch_size = old_batch_size
            self.test_env.batch_pop_range = old_batch_pop_range
            self.test_env.state = old_state
            if old_baseline is not None:
                self.test_env.baseline = old_baseline
            if old_boost is not None:
                self.test_env.boost = old_boost
            if old_expected is not None:
                self.test_env.expected = old_expected
            if old_upper_bound is not None:
                self.test_env.upper_bound = old_upper_bound

        return (
            new_solutions.float().reshape(batch_size, n_weights * rollouts_per_weight, state.problem_size),
            new_obj.float().reshape(batch_size, n_weights * rollouts_per_weight),
        )

    def _update_cnc_population_archive(self, archive_solutions, archive_obj, new_solutions, new_obj, archive_size):
        if self.params['cnc_pop_keep_policy'] == 'new':
            if new_solutions.size(1) == archive_size:
                return new_solutions, new_obj
            idx = new_obj.topk(archive_size, dim=1).indices
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, new_solutions.size(-1))
            return new_solutions.gather(1, gather_idx), new_obj.gather(1, idx)

        candidates = torch.cat([archive_solutions, new_solutions], dim=1)
        candidate_obj = torch.cat([archive_obj, new_obj], dim=1)
        return self._keep_cnc_archive(candidates, candidate_obj, archive_size)

    def _update_cnc_best_so_far(self, archive_solutions, archive_obj, best_so_far, best_solutions_so_far):
        cur_best_obj, cur_best_idx = archive_obj.max(dim=1)
        improved = cur_best_obj > best_so_far
        best_so_far = torch.maximum(best_so_far, cur_best_obj)
        best_solutions_so_far[improved] = archive_solutions[
            torch.arange(archive_solutions.size(0), device=self.device),
            cur_best_idx,
        ][improved]
        return best_so_far, best_solutions_so_far

    def _record_cnc_population_progress(
        self,
        elapsed_time,
        all_time_obj,
        diversity_vals,
        start_time,
        best_so_far,
        archive_solutions,
        graph,
        compute_diversity,
    ):
        elapsed_time.append(time.time() - start_time)
        all_time_obj.append(best_so_far.cpu().numpy().tolist())
        diversity_vals.extend(self._archive_diversity(archive_solutions, graph, compute_diversity))

    def _make_cnc_state(self, state, visited_solutions, exploration_weight=None, greedy_hint=False, expanded_1d=None):
        greedy_inference_hint = None
        if self.nc_model.model_params.get('cnc_presample_greedy_feature', False):
            greedy_inference_hint = torch.ones(state.batch_size, dtype=torch.float32, device=self.device) if greedy_hint else torch.zeros(state.batch_size, dtype=torch.float32, device=self.device)
        if expanded_1d is not None:
            exploration_weight = expanded_1d
        elif torch.is_tensor(exploration_weight):
            exploration_weight = exploration_weight.to(dtype=torch.float32, device=self.device)
        else:
            exploration_weight = torch.full((state.batch_size,), exploration_weight, dtype=torch.float32, device=self.device)
        return replace(
            state,
            visited_solutions=visited_solutions.float(),
            exploration_weight=exploration_weight,
            greedy_inference_hint=greedy_inference_hint,
        )

    def _cnc_feasibility_decoder(self):
        return self.params.get('feasibility_decoder', 'heatmap')

    def _make_zero_archive(self, batch_size, problem_size, archive_size):
        archive = torch.zeros(batch_size, problem_size, archive_size, dtype=torch.float32, device=self.device)
        if self.params['problem'] == 'mc':
            archive[:, 0, :] = 1.0
        return archive

    def _make_random_archive(self, batch_size, problem_size, archive_size, archive_seed):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(archive_seed))
        archive = torch.randint(
            0,
            2,
            (batch_size, problem_size, archive_size),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        if self.params['problem'] == 'mc':
            archive[:, 0, :] = 1.0
        return archive

    def _make_cnc_greedy_once_archive(self, state, archive_seed):
        archive_size = self.params['n_visited_solutions']
        context = self._resolve_cnc_greedy_once_archive_context()
        if context == 'zero_archive':
            return self._make_zero_archive(state.batch_size, state.problem_size, archive_size)
        if context == 'random_archive':
            return self._make_random_archive(state.batch_size, state.problem_size, archive_size, archive_seed)
        raise ValueError(f"Invalid cNC greedy-once archive context: {context}")

    def _resolve_cnc_greedy_once_archive_context(self):
        context = self.params.get('cnc_greedy_once_archive_context', 'auto')
        if context == 'auto':
            return self.params.get('cnc_presample_context', 'random_archive')
        return context

    def _make_cnc_presample_archive(self, state, archive_size, archive_seed):
        context = self.params.get('cnc_presample_context', 'random_archive')
        if context == 'zero_archive':
            return self._make_zero_archive(state.batch_size, state.problem_size, archive_size)
        if context == 'random_archive':
            return self._make_random_archive(state.batch_size, state.problem_size, archive_size, archive_seed)
        raise ValueError(f"Invalid cNC presample archive context: {context}")

    def _cnc_archive_seed(self, graph_type, graph_idx):
        base_seed = self.params.get('cnc_eval_archive_seed', self.params.get('seed', 42))
        return int(base_seed + 1009 * int(graph_idx) + 9176 * sum(ord(c) for c in str(graph_type)))

    def _cnc_archive_size(self):
        checkpoint_size = self.params['n_visited_solutions']
        requested_size = self.params['cnc_pop_seed_size']
        if requested_size not in (0, checkpoint_size):
            raise ValueError(
                f"cNC checkpoint expects n_visited_solutions={checkpoint_size}, "
                f"but cnc_pop_seed_size={requested_size}. Use 0 or the checkpoint value."
            )
        return checkpoint_size

    def _keep_cnc_archive(self, candidates, obj_values, archive_size):
        if self.params['cnc_pop_keep_policy'] == 'best_unique':
            selected = []
            selected_obj = []
            for b in range(candidates.size(0)):
                order = obj_values[b].argsort(descending=True)
                cur = []
                cur_obj = []
                seen = set()
                for idx in order:
                    sol_key = candidates[b, idx].detach().to(torch.int8).cpu().numpy().tobytes()
                    if sol_key in seen:
                        continue
                    seen.add(sol_key)
                    cur.append(candidates[b, idx])
                    cur_obj.append(obj_values[b, idx])
                    if len(cur) == archive_size:
                        break
                while len(cur) < archive_size:
                    cur.append(candidates[b, order[0]])
                    cur_obj.append(obj_values[b, order[0]])
                selected.append(torch.stack(cur, dim=0))
                selected_obj.append(torch.stack(cur_obj, dim=0))
            return torch.stack(selected, dim=0), torch.stack(selected_obj, dim=0)

        idx = obj_values.topk(archive_size, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, candidates.size(-1))
        return candidates.gather(1, gather_idx), obj_values.gather(1, idx)

    def _archive_diversity(self, archive_solutions, graph, compute_diversity):
        if not compute_diversity:
            return []
        cur_div = self.diversity_fn(archive_solutions, graph)
        return [float(cur_div.detach().item())]


def save_results(results, run_path, filename_suffix=""):
    parts = []
    if filename_suffix:
        parts.append(filename_suffix)
    suffix = ("_" + "_".join(parts)) if parts else ""
    results_path = run_path / f"results{suffix}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")


def parse_cnc_weights(weights):
    parsed_weights = [float(w.strip()) for w in weights.split(',') if w.strip()]
    if not parsed_weights:
        raise ValueError("cnc_pop_weights must contain at least one weight.")
    return parsed_weights


def parse_cnc_threshold_grid(grid):
    if ':' in grid:
        parts = [p.strip() for p in grid.split(':')]
        if len(parts) != 3:
            raise ValueError("cnc_threshold_grid range format must be start:stop:num.")
        start, stop = float(parts[0]), float(parts[1])
        num = int(parts[2])
        if num < 1:
            raise ValueError("cnc_threshold_grid num must be at least 1.")
        thresholds = np.linspace(start, stop, num).tolist()
    else:
        thresholds = [float(t.strip()) for t in grid.split(',') if t.strip()]

    if not thresholds:
        raise ValueError("cnc_threshold_grid must contain at least one threshold.")
    for threshold in thresholds:
        validate_cnc_threshold(threshold)
    return thresholds


def format_cnc_thresholds(thresholds):
    if len(thresholds) <= 12:
        return ", ".join(f"{threshold:.4f}" for threshold in thresholds)
    head = ", ".join(f"{threshold:.4f}" for threshold in thresholds[:6])
    tail = ", ".join(f"{threshold:.4f}" for threshold in thresholds[-3:])
    return f"{head}, ..., {tail}"


def validate_cnc_threshold(threshold):
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"cNC threshold must be in (0, 1), got {threshold}.")


def threshold_to_cnc_beta(threshold):
    validate_cnc_threshold(threshold)
    return float(np.log((1.0 - threshold) / threshold))


def _uses_nc_initializer(init_mode):
    return init_mode in {'cnc', 'greedy_cnc'}


def normalize_eval_model_config(eval_params, use_ni_model, use_nc_model):
    if not use_ni_model:
        if use_nc_model:
            eval_params['initialization'] = 'cnc'
            eval_params['multi_start'] = False
            eval_params.setdefault('k', 1)
            eval_params.setdefault('memory_type', 'none')
            eval_params.setdefault('mem_aggr', 'linear')
            eval_params.setdefault('memory_size', 0)
            eval_params.setdefault('mem_value_type', 'objective')
            print("cNC evaluation.")
            return
        raise ValueError("No model provided. Set --ni_model_load_path, optionally with --nc_model_load_path.")

    if not use_nc_model:
        fallback_init = 'random'
        nc_dependent_params = ['initialization']
        if eval_params['multi_start']:
            nc_dependent_params.append('ms_init_mode')

        for param_name in nc_dependent_params:
            if _uses_nc_initializer(eval_params[param_name]):
                print(
                    f"Warning: {param_name}={eval_params[param_name]!r} requires an NC model. "
                    f"Defaulting to {fallback_init!r}."
                )
                eval_params[param_name] = fallback_init

        print("cNI evaluation.")
    else:
        print("cNI+cNC evaluation.")


def main():
    eval_params = get_args()
        
    if eval_params['distance_metric'] == 'default':
        if eval_params['problem'] == 'mc':
            eval_params['distance_metric'] = 'edge_hamming'
        elif eval_params['problem'] == 'mis':
            eval_params['distance_metric'] = 'node_hamming'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        eval_params['compile'] = False

    use_ni_model, use_nc_model = False, False

    ni_params = None
    if eval_params['ni_model_load_path'] != '':
        # Restore NI model params
        ni_checkpoint = torch.load(eval_params['ni_model_load_path'], map_location=device, weights_only=False)
        ni_params = ni_checkpoint['params']
        ni_params['device'] = device
        eval_params['k'] = ni_params['k']
        eval_params['memory_type'] = ni_params['memory_type']
        eval_params['mem_aggr'] = ni_params['mem_aggr']
        eval_params['memory_size'] = ni_params['memory_size']
        eval_params['mem_value_type'] = ni_params['mem_value_type']
        use_ni_model = True

    nc_params = None
    if eval_params['nc_model_load_path'] != '':
        # Restore NC model params
        nc_checkpoint = torch.load(eval_params['nc_model_load_path'], map_location=device, weights_only=False)
        nc_params = nc_checkpoint['params']
        nc_params['device'] = device
        eval_params['nc_train_mode'] = nc_params['nc_train_mode']
        eval_params['n_visited_solutions'] = nc_params['n_visited_solutions']
        checkpoint_default_params = [
            'feasibility_decoder',
            'cnc_w_sampling',
            'cnc_beta',
            'punish_unfeasible',
            'punish_w',
            'mis_reward_norm',
            'skip_unused_mis_bounds',
            'ppo_logprob_reduction',
            'vectorize_rollouts',
            'normalize_rewards',
            'cnc_presample_context',
            'cnc_eval_archive_seed',
        ]
        for param_name in checkpoint_default_params:
            if param_name in nc_params:
                eval_params[param_name] = nc_params[param_name]
        if eval_params.get('mis_heatmap_post_add') is None:
            eval_params['mis_heatmap_post_add'] = eval_params['problem'] == 'mis'
        if eval_params.get('mis_heatmap_post_add_mode') is None:
            eval_params['mis_heatmap_post_add_mode'] = 'greedy'
        if eval_params.get('mis_heatmap_post_add_temperature') is None:
            eval_params['mis_heatmap_post_add_temperature'] = 1.0
        use_nc_model = True

    normalize_eval_model_config(eval_params, use_ni_model, use_nc_model)

    evaluator = Evaluator(eval_params, ni_params, nc_params)
    test_results = evaluator.run_tests()
    if eval_params['save_results']:
        test_results['eval_params'] = eval_params
        # Prefer explicit CLI suffix; otherwise build something simple
        suffix = eval_params.get('results_suffix') or f"seed{eval_params['seed']}"
        save_results(test_results, evaluator.run_path, suffix)


if __name__ == '__main__':
    main()
