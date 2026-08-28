import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PB-NCO NC v2")

    # ENV
    parser.add_argument('--problem', type=str, default='mc', choices=['mc', 'mis'], help='Problem: mc or mis')
    parser.add_argument('--train_graph_type', type=str, default='ER', choices=['ER', 'RB'])
    parser.add_argument('--conn_low', type=float, default=0.15, help='Lower bound of the connection probability')
    parser.add_argument('--conn_high', type=float, default=0.15, help='Upper bound of the connection probability')
    parser.add_argument('--laplace_dim', type=int, default=0, help='Number of Laplacian eigenvectors used as positional encodings. 0 for not using positional encodings.')
    parser.add_argument('--feasibility_decoder', type=str, default='heatmap', choices=['sequential', 'heatmap'], help='How to decode solutions.')
    parser.add_argument('--distance_metric', type=str, default='default', choices=['default', 'node_hamming', 'edge_hamming', 'coverage_jaccard'], help='The distance metric used to compute the distance between solutions in memory.')

    # NC args
    parser.add_argument('--nc_train_mode', type=str, default='conditioned_network', choices=['exploitation', 'exploration', 'exploration_exploitation', 'conditioned_network'], help='Train mode.')
    parser.add_argument('--n_visited_solutions', type=int, default=20, help='Number of visited solutions considered for the exploration')
    parser.add_argument('--n_rollouts', type=int, default=10, help='Number of sampling rollouts to train the NC model')
    parser.add_argument('--exploration_weight', type=float, default=0.1, help='Exploration weight to compute reward for the case of exploration_exploitation')

    # cNC args
    parser.add_argument('--cnc_w_sampling', type=str, default='beta', choices=['uniform', 'beta', 'clipped_normal'], help='Which sampling distribution to use for the exploration weight in the cNC training')
    parser.add_argument('--cnc_beta', type=float, default=0.2, help='Beta=alpha parameters for the cNC training with beta sampling')

    # cNC v2 archive pre-sampling args
    parser.add_argument('--cnc_archive_source', type=str, default='model_presample', choices=['model_presample', 'random'], help='How to construct visited_solutions for cNC training')
    parser.add_argument('--cnc_presample_context', type=str, default='zero_archive', choices=['zero_archive', 'random_archive'], help='Which visited_solutions context to use during the w=0 pre-sampling pass')
    parser.add_argument('--cnc_presample_rollouts', type=int, default=None, help='Number of w=0 pre-sampling rollouts. Defaults to n_visited_solutions')
    parser.add_argument('--cnc_archive_select', type=str, default='best_objective', choices=['best_objective', 'random_samples', 'diverse_best'], help='How to select visited_solutions from pre-sampled model solutions')
    parser.add_argument('--cnc_archive_solution_source', type=str, default='repaired', choices=['repaired', 'raw'], help='Whether cNC archives store repaired feasible solutions or raw heatmap samples.')
    parser.add_argument('--cnc_presample_loss_mode', type=str, default='objective_ppo', choices=['objective_ppo', 'none', 'mixed_reward'], help='How the w=0 pre-sampling pass contributes to PPO training')
    parser.add_argument('--cnc_presample_loss_coef', type=float, default=1.0, help='Multiplier applied to PPO loss from the w=0 pre-sampling pass')
    parser.add_argument('--cnc_diversity_metric_train', type=str, default='configured', choices=['configured', 'node_hamming', 'edge_hamming', 'mixed_node_edge'], help='Diversity metric used for cNC training rewards')
    parser.add_argument('--cnc_diversity_solution_source', type=str, default='repaired', choices=['repaired', 'raw', 'prob'], help='Use repaired solutions, raw heatmap samples, or probability-aware diversity for cNC rewards.')
    parser.add_argument('--cnc_mixed_diversity_alpha', type=float, default=0.5, help='Node-Hamming weight for mixed_node_edge diversity reward')
    parser.add_argument('--cnc_archive_model_update_freq', type=int, default=0, help='Refresh frequency for a frozen pre-sampling archive model. 0 uses the online model')
    parser.add_argument('--cnc_presample_greedy_feature', action=argparse.BooleanOptionalAction, default=False, help='Add a binary model input feature that marks the w=0 archive pre-sampling pass')

    # Constraints (MIS)
    parser.add_argument('--punish_unfeasible', action=argparse.BooleanOptionalAction, default=True, help='Use punishment for unfeasibility')
    parser.add_argument('--punish_w', type=float, default=0.1, help='Punishment weight')
    parser.add_argument('--mis_reward_norm', type=str, default='upper_bound', choices=['upper_bound', 'size', 'centered_size', 'rollout_rank'], help='Reward normalization for MIS constructive training.')
    parser.add_argument('--skip_unused_mis_bounds', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip MIS bound computations that are unused by the selected NC reward normalization.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layers', type=int, default=8, help='number of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'instance', 'layer', 'rms'], help='Normalization')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'swiglu', 'relu'], help='Activation')
    parser.add_argument('--bias', action=argparse.BooleanOptionalAction, default=False, help='Use bias')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='Tanh clipping')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False, help='Compile pytorch model')

    # Training
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (number of instances)')
    parser.add_argument('--min_problem_size', type=int, default=20, help='Min problem size')
    parser.add_argument('--max_problem_size', type=int, default=20, help='Max problem size')
    parser.add_argument('--curriculum_learning', action=argparse.BooleanOptionalAction, default=False, help='Curriculum learning')
    parser.add_argument('--max_size_exec_percent', type=float, default=0.8, help='The fraction of the execution when the maximum problem size is reached during training. 0.8 means that the maximum problem size is reached at 80 percent of the training epochs.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max grad norm')
    parser.add_argument('--train_diagnostics', type=str, default='full', choices=['full', 'summary', 'none'],
                        help='Auxiliary per-episode training diagnostics: full quantiles, mean-only summary, or none.')
    parser.add_argument('--vectorize_rollouts', action=argparse.BooleanOptionalAction, default=True,
                        help='Decode heatmap rollouts in one expanded batch. Faster but changes RNG order versus per-rollout loops.')

    # PPO
    parser.add_argument('--ppo_restarts', type=int, default=20, help='Number of PPO restarts')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO epochs')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO clip')
    parser.add_argument('--entropy_coef', type=float, default=0.0, help='Entropy coefficient')
    parser.add_argument('--ppo_update_batch_count', type=int, default=5, help='PPO update batch count')
    parser.add_argument('--ppo_logprob_reduction', type=str, default='mean', choices=['mean', 'sum'],
                        help='Reduce per-node log-probs with mean (default behavior) or sum (trajectory-like PPO ratio).')
    parser.add_argument('--kl_target', type=float, default=0.0,
                        help='Stop PPO epochs early when mean approximate KL exceeds this value. 0 disables early stopping.')
    parser.add_argument('--mis_heatmap_post_add', action=argparse.BooleanOptionalAction, default=False,
                        help='For MIS heatmap decoding, optionally complete repaired solutions by adding feasible nodes.')
    parser.add_argument('--mis_heatmap_post_add_mode', type=str, default='greedy', choices=['greedy', 'sample'],
                        help='How to choose feasible post-add nodes when --mis_heatmap_post_add is enabled.')
    parser.add_argument('--mis_heatmap_post_add_temperature', type=float, default=1.0,
                        help='Sampling temperature for MIS post-add when using sample mode.')

    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), metavar=('BETA1', 'BETA2'), help='AdamW beta coefficients')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')

    # Eval
    parser.add_argument('--eval_graph_types', type=str, nargs='+', default=['ER20', 'ER100', 'ER200', 'ER500'], help='Graph sizes for evaluation')
    parser.add_argument('--num_eval_graphs', type=int, nargs='+', default=[100, 100, 100, 10], help='Number of graphs for evaluation')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency (epochs)')
    parser.add_argument('--cnc_eval_archive_source', type=str, default='fixed_random', choices=['fixed_random', 'best_of_random', 'self_presample'], help='Archive source used for comparable cNC evaluation')
    parser.add_argument('--cnc_eval_archive_seed', type=int, default=12345, help='Base seed for deterministic evaluation archives')
    parser.add_argument('--cnc_eval_presample_rollouts', type=int, default=None, help='Number of eval pre-sampling/random pool rollouts. Defaults to n_visited_solutions')

    # Others
    parser.add_argument('--nc_model_load_path', type=str, default='', help='Model load path. Use '' for not loading any model')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=False, help='Save model checkpoints')
    parser.add_argument('--save_model_freq', type=int, default=10, help='Save model frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False, help='Use wandb')
    parser.add_argument('--run_id', type=str, default='', help='Optional run name shown in wandb')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False, help='Debug')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False, help='Use mixed precision (Automatic Mixed Precision) to reduce memory')

    args = parser.parse_args()
    if args.cnc_presample_rollouts is None:
        args.cnc_presample_rollouts = args.n_visited_solutions
    if args.cnc_eval_presample_rollouts is None:
        args.cnc_eval_presample_rollouts = args.n_visited_solutions
    # group all arguments in one dictionary
    all_args = {
        # ENV
        'problem': args.problem,
        'train_graph_type': args.train_graph_type,
        'conn_low': args.conn_low,
        'conn_high': args.conn_high,
        'laplace_dim': args.laplace_dim,
        'feasibility_decoder': args.feasibility_decoder,
        'distance_metric': args.distance_metric,
        # NC args
        'nc_train_mode': args.nc_train_mode,
        'n_visited_solutions': args.n_visited_solutions,
        'n_rollouts': args.n_rollouts,
        'exploration_weight': args.exploration_weight,
        # cNC args
        'cnc_w_sampling': args.cnc_w_sampling,
        'cnc_beta': args.cnc_beta,
        # cNC v2 archive pre-sampling args
        'cnc_archive_source': args.cnc_archive_source,
        'cnc_presample_context': args.cnc_presample_context,
        'cnc_presample_rollouts': args.cnc_presample_rollouts,
        'cnc_archive_select': args.cnc_archive_select,
        'cnc_archive_solution_source': args.cnc_archive_solution_source,
        'cnc_presample_loss_mode': args.cnc_presample_loss_mode,
        'cnc_presample_loss_coef': args.cnc_presample_loss_coef,
        'cnc_diversity_metric_train': args.cnc_diversity_metric_train,
        'cnc_diversity_solution_source': args.cnc_diversity_solution_source,
        'cnc_mixed_diversity_alpha': args.cnc_mixed_diversity_alpha,
        'cnc_archive_model_update_freq': args.cnc_archive_model_update_freq,
        'cnc_presample_greedy_feature': args.cnc_presample_greedy_feature,
        # Constraints (MIS)
        'punish_unfeasible': args.punish_unfeasible,
        'punish_w': args.punish_w,
        'mis_reward_norm': args.mis_reward_norm,
        'skip_unused_mis_bounds': args.skip_unused_mis_bounds,
        # Model
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'normalization': args.normalization,
        'activation': args.activation,
        'bias': args.bias,
        'tanh_clipping': args.tanh_clipping,
        'dropout': args.dropout,
        'compile': args.compile,
        # Training
        'n_epochs': args.n_epochs,
        'n_episodes': args.n_episodes,
        'batch_size': args.batch_size,
        'min_problem_size': args.min_problem_size,
        'max_problem_size': args.max_problem_size,
        'curriculum_learning': args.curriculum_learning,
        'max_size_exec_percent': args.max_size_exec_percent,
        'max_grad_norm': args.max_grad_norm,
        'train_diagnostics': args.train_diagnostics,
        'vectorize_rollouts': args.vectorize_rollouts,
        # PPO
        'ppo_restarts': args.ppo_restarts,
        'ppo_epochs': args.ppo_epochs,
        'ppo_clip': args.ppo_clip,
        'entropy_coef': args.entropy_coef,
        'ppo_update_batch_count': args.ppo_update_batch_count,
        'ppo_logprob_reduction': args.ppo_logprob_reduction,
        'kl_target': args.kl_target,
        'mis_heatmap_post_add': args.mis_heatmap_post_add,
        'mis_heatmap_post_add_mode': args.mis_heatmap_post_add_mode,
        'mis_heatmap_post_add_temperature': args.mis_heatmap_post_add_temperature,
        # Optimizer
        'lr': args.lr,
        'betas': tuple(args.betas),
        'weight_decay': args.weight_decay,
        # Eval
        'eval_graph_types': args.eval_graph_types,
        'num_eval_graphs': args.num_eval_graphs,
        'eval_freq': args.eval_freq,
        'cnc_eval_archive_source': args.cnc_eval_archive_source,
        'cnc_eval_archive_seed': args.cnc_eval_archive_seed,
        'cnc_eval_presample_rollouts': args.cnc_eval_presample_rollouts,
        # Others
        'nc_model_load_path': args.nc_model_load_path,
        'save_model': args.save_model,
        'save_model_freq': args.save_model_freq,
        'seed': args.seed,
        'wandb': args.wandb,
        'run_id': args.run_id,
        'verbose': args.verbose,
        'debug': args.debug,
        'amp': args.amp,
    }

    return all_args
