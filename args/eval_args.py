import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PB-NCO")
    # ENV
    parser.add_argument('--problem', default='mc', type=str, choices=['mc', 'mis'], help='Problem')
    parser.add_argument('--distance_metric', type=str, default='default', choices=['default', 'node_hamming', 'edge_hamming'], help='The distance metric used to compute the distance between solutions in memory.')

    # EVAL GRAPHS
    parser.add_argument('--eval_graph_types', nargs='+', type=str, default=['ER700_800'],
                        help='List of graph type tags, e.g. ER20 ER200 ER700_800')
    parser.add_argument('--num_eval_graphs', nargs='+', type=int, default=[128],
                        help='List of counts per graph type; same length as --eval_graph_types')
    parser.add_argument('--eval_graph_idx', default=-1, type=int, help='Idx of the evaluation graph. Default: -1, for solving all graphs in a batch.')

    # POPULATION BASED NCO
    parser.add_argument('--pop_size', default=20, type=int, help='Number of agents in the population')
    parser.add_argument('--multi_start', action=argparse.BooleanOptionalAction, default=True, help='Use multi-start for inference')
    parser.add_argument('--multi_start_patience', default=200, type=int, help='Multi start patience')
    parser.add_argument('--ms_init_mode', default='cnc', type=str, choices=['cnc', 'random', 'greedy_cnc'], help='Initialization mode for multi start')
    parser.add_argument('--phi', default=1, type=float, help='phi for exploration weight schedule')
    parser.add_argument('--cnc_visit_strategy', default='last_k', type=str, choices=['last_k', 'best_k_global', 'best_k_current'], help='How to form visited_solutions for cNC during init/resets')

    # INFERENCE
    parser.add_argument('--test_max_steps_multiplier', default=4.0, type=float, help='Number of max steps (multiplier) M * problem_size')
    parser.add_argument('--test_patience', default=1e8, type=int, help='Number of non-improving steps before stopping the whole execution. Put large number to disable it.')
    parser.add_argument('--max_time_per_instance', default=float('inf'), type=float,
                        help='Wall-clock limit in seconds for each anytime instance. Default: no time limit.')
    parser.add_argument('--initialization', default='random', type=str, choices=['random', 'cnc'], help='Initialization')
    parser.add_argument('--topk', default=1, type=int, help='Topk')
    # cNC EVAL
    parser.add_argument('--cnc_eval_mode', default='greedy_once', type=str, choices=['greedy_once', 'cnc_pop', 'guided_min_degree'], help='Evaluation mode used when only an NC checkpoint is provided')
    parser.add_argument('--cnc_greedy_once_archive_context', default='auto', type=str, choices=['auto', 'zero_archive', 'random_archive'],
                        help='Visited-solution archive used for greedy one-shot cNC. auto follows checkpoint cnc_presample_context')
    parser.add_argument('--cnc_pop_seed_size', default=0, type=int, help='Number of cNC seed/archive solutions. 0 uses checkpoint n_visited_solutions')
    parser.add_argument('--cnc_pop_generations', default=5, type=int, help='Number of cNC population generations')
    parser.add_argument('--cnc_pop_weight_mode', default='sweep', type=str, choices=['sweep', 'rollout_linspace', 'grouped_linspace'], help='How exploration weights are assigned during cNC population updates')
    parser.add_argument('--cnc_pop_weights', default='0.0,0.1,0.2,0.3,0.4,0.5', type=str, help='Comma-separated exploration weights for cNC population generations')
    parser.add_argument('--cnc_pop_weight_min', default=0.0, type=float, help='Minimum exploration weight for --cnc_pop_weight_mode rollout_linspace')
    parser.add_argument('--cnc_pop_weight_max', default=0.6, type=float, help='Maximum exploration weight for --cnc_pop_weight_mode rollout_linspace')
    parser.add_argument('--cnc_pop_num_weights', default=6, type=int, help='Number of linspace weights for --cnc_pop_weight_mode grouped_linspace')
    parser.add_argument('--cnc_pop_rollouts_per_weight', default=0, type=int, help='Rollouts per grouped linspace weight. 0 uses ceil(archive_size / cnc_pop_num_weights)')
    parser.add_argument('--cnc_pop_keep_policy', default='new', type=str, choices=['best', 'best_unique', 'new'], help='How to keep the next cNC archive')
    parser.add_argument('--cnc_guided_pool_frac', default=0.02, type=float,
                        help='For guided_min_degree, choose the highest model-score node among this lowest residual-degree fraction. Set <=0 to use alpha blending.')
    parser.add_argument('--cnc_guided_alpha', default=1.0, type=float,
                        help='For guided_min_degree with pool_frac <= 0, minimize residual_degree - alpha * model_prob.')
    # Calibration
    parser.add_argument('--cnc_threshold', default=0.5, type=float, help='Probability threshold for greedy one-shot MC cNC decoding')
    parser.add_argument('--cnc_calibrate_threshold', action=argparse.BooleanOptionalAction, default=False,
                        help='Tune the greedy one-shot MC cNC threshold on a calibration split and evaluate on the held-out split')
    parser.add_argument('--cnc_threshold_grid', default='0.05:0.95:19', type=str,
                        help='Threshold grid for calibration. Use comma list or start:stop:num, e.g. 0.1,0.2,0.5 or 0.05:0.95:19')
    parser.add_argument('--cnc_calibration_fraction', default=0.5, type=float,
                        help='Fraction of eval instances used for threshold calibration')
    parser.add_argument('--cnc_calibration_percent', default=None, type=float,
                        help='Percentage of eval instances used for threshold calibration. Overrides --cnc_calibration_fraction when set')
    parser.add_argument('--cnc_calibration_split_seed', default=None, type=int,
                        help='Seed for the calibration/test split. Defaults to --seed')
    parser.add_argument('--mis_heatmap_post_add', action=argparse.BooleanOptionalAction, default=None,
                        help='For MIS heatmap cNC eval, complete repaired solutions by adding feasible nodes. Default follows checkpoint when available.')
    parser.add_argument('--mis_heatmap_post_add_mode', default=None, type=str, choices=['greedy', 'sample'],
                        help='MIS cNC eval post-add mode. Default follows checkpoint when available.')
    parser.add_argument('--mis_heatmap_post_add_temperature', default=None, type=float,
                        help='MIS cNC eval post-add sampling temperature. Default follows checkpoint when available.')

    # MODEL PATHS
    parser.add_argument('--ni_model_load_path', type=str, default='', help='Model load path for NI')
    parser.add_argument('--nc_model_load_path', type=str, default='', help='Model load path for NC')

    # OTHERS
    parser.add_argument('--run_name', type=str, default='', help='Optional run name suffix for reproducibility')
    parser.add_argument('--results_suffix', type=str, default='', help='Suffix appended to results filename, e.g. "seed42_pop16".')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False, help='Compile pytorch model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose')
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=False, help='Save results to file')

    args = parser.parse_args()

    all_args = {
        # ENV
        'problem': args.problem,
        'distance_metric': args.distance_metric,
        # EVAL GRAPHS
        'eval_graph_types': args.eval_graph_types,
        'num_eval_graphs': args.num_eval_graphs,
        'eval_graph_idx': args.eval_graph_idx,
        # Population-based inference
        'pop_size': args.pop_size,
        'multi_start': args.multi_start,
        'multi_start_patience': args.multi_start_patience,
        'ms_init_mode': args.ms_init_mode,
        'phi': args.phi,
        'cnc_visit_strategy': args.cnc_visit_strategy,
        # INFERENCE
        'test_max_steps_multiplier': args.test_max_steps_multiplier,
        'test_patience': args.test_patience,
        'max_time_per_instance': args.max_time_per_instance,
        'initialization': args.initialization,
        'topk': args.topk,
        'cnc_eval_mode': args.cnc_eval_mode,
        'cnc_pop_seed_size': args.cnc_pop_seed_size,
        'cnc_pop_generations': args.cnc_pop_generations,
        'cnc_pop_weight_mode': args.cnc_pop_weight_mode,
        'cnc_pop_weights': args.cnc_pop_weights,
        'cnc_pop_weight_min': args.cnc_pop_weight_min,
        'cnc_pop_weight_max': args.cnc_pop_weight_max,
        'cnc_pop_num_weights': args.cnc_pop_num_weights,
        'cnc_pop_rollouts_per_weight': args.cnc_pop_rollouts_per_weight,
        'cnc_pop_keep_policy': args.cnc_pop_keep_policy,
        'cnc_guided_pool_frac': args.cnc_guided_pool_frac,
        'cnc_guided_alpha': args.cnc_guided_alpha,
        'cnc_threshold': args.cnc_threshold,
        'cnc_calibrate_threshold': args.cnc_calibrate_threshold,
        'cnc_threshold_grid': args.cnc_threshold_grid,
        'cnc_calibration_fraction': args.cnc_calibration_fraction,
        'cnc_calibration_percent': args.cnc_calibration_percent,
        'cnc_calibration_split_seed': args.cnc_calibration_split_seed,
        'cnc_greedy_once_archive_context': args.cnc_greedy_once_archive_context,
        'mis_heatmap_post_add': args.mis_heatmap_post_add,
        'mis_heatmap_post_add_mode': args.mis_heatmap_post_add_mode,
        'mis_heatmap_post_add_temperature': args.mis_heatmap_post_add_temperature,
        # MODEL PATHS
        'ni_model_load_path': args.ni_model_load_path,
        'nc_model_load_path': args.nc_model_load_path,
        # OTHERS
        'run_name': args.run_name,
        'results_suffix': args.results_suffix,
        'compile': args.compile,
        'seed': args.seed,
        'verbose': args.verbose,
        'save_results': args.save_results,
    }
    return all_args
