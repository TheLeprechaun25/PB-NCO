import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PB-NCO v2")

    # ENV
    parser.add_argument('--problem', type=str, default='mc', choices=['mc', 'mis'], help='Problem to be solved')
    parser.add_argument('--train_graph_type', type=str, default='ER', choices=['ER', 'RB'])
    parser.add_argument('--initialization', type=str, default='random', help='Initialization of solutions')
    parser.add_argument('--conn_low', type=float, default=0.15, help='Lower bound of the connection probability')
    parser.add_argument('--conn_high', type=float, default=0.15, help='Upper bound of the connection probability')
    parser.add_argument('--distance_metric', type=str, default='default', choices=['default', 'node_hamming', 'edge_hamming'], help='The distance metric used to compute the distance between solutions in memory.')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layers', type=int, default=3, help='n_layers')
    parser.add_argument('--n_heads', type=int, default=8, help='n_heads')
    parser.add_argument('--normalization', type=str, default='layer', choices=['batch', 'instance', 'layer', 'rms'], help='Normalization')
    parser.add_argument('--activation', type=str, default='gelu', choices=['gelu', 'swiglu', 'relu'], help='Activation')
    parser.add_argument('--bias', action=argparse.BooleanOptionalAction, default=False, help='Use bias')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='Tanh clipping')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False, help='Compile pytorch model')

    # Memory
    parser.add_argument('--memory_type', type=str, default='shared', choices=['shared', 'individual', 'op_based', 'none'], help='Memory type')
    parser.add_argument('--memory_size', type=int, default=10000, help='Memory size')
    parser.add_argument('--k', type=int, default=20, help='k nearest solutions are gathered from memory in every step')
    parser.add_argument('--mem_aggr', type=str, default='linear', choices=['sum', 'linear', 'exp', 'concat'], help='The aggregation method for the memory. Sum of all k values. Linear weighted sum. Exponential weighted sum.')
    parser.add_argument('--mem_value_type', type=str, default='combined', choices=['actions', 'solutions', 'combined', 'differences'], help='The type of value gathered from memory, actions or solutions or differences.')
    parser.add_argument('--revisit_punishment', type=float, default=1.0, help='Revisit punishment')

    # Training
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--train_max_steps_multiplier', type=float, default=2.0, help='Max inference steps multiplier for training. 2 * problem_size')
    parser.add_argument('--train_patience', type=int, default=200, help='Patience for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (number of instances)')
    parser.add_argument('--pop_size', type=int, default=20, help='Population size')
    parser.add_argument('--min_problem_size', type=int, default=50, help='Min problem size')
    parser.add_argument('--max_problem_size', type=int, default=300, help='Max problem size')
    parser.add_argument('--gamma', type=float, default=0.95, help='Gamma for discounted rewards')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max grad norm')
    parser.add_argument('--normalize_advantages', action=argparse.BooleanOptionalAction, default=False, help='Normalize advantages')
    # PPO
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO epochs per buffer‐update')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.0, help='Weight for the entropy bonus in PPO loss')
    parser.add_argument('--n_stored_states', type=int, default=15, help='Number of states to store in buffer in each episode.')
    parser.add_argument('--ppo_update_batch_count', type=int, default=5, help='Number of batches to include in each PPO update. Total batches: n_stored_states*n_episodes')
    parser.add_argument('--kl_target', type=float, default=0.0,
                        help='KL divergence threshold for early stopping within PPO epochs (0 = disabled)')

    # Optimizer
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    # Eval
    parser.add_argument('--eval_graph_types', type=str, nargs='+', default=['ER20', 'ER100', 'ER200', 'ER500'], help='Graph sizes for evaluation')
    parser.add_argument('--num_eval_graphs', type=int, nargs='+', default=[100, 100, 100, 10], help='Number of graphs for evaluation')
    parser.add_argument('--test_pop_size', type=int, default=8, help='Number of agents in the population')
    parser.add_argument('--test_max_steps_multiplier', type=float, default=1.0, help='Max inference step multiplier for testing. 2 is for 2 * problem_size steps')
    parser.add_argument('--test_patience', type=int, default=10000, help='Patience for testing')
    parser.add_argument('--topk', type=int, default=1, help='Executed actions in each inference step. topk=1 for default inference')

    # Others
    parser.add_argument('--ni_model_load_path', type=str, default='', help='Model load path. Use \'\' for not loading any model')
    parser.add_argument('--save_models', action=argparse.BooleanOptionalAction, default=False, help='Save models')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False, help='Use wandb')
    parser.add_argument('--run_id', type=str, default='', help='Optional run name shown in wandb')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Verbose')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False, help='Debug')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False, help='Use mixed precision (Automatic Mixed Precision) to reduce memory')

    args = parser.parse_args()
    all_args = {
        # ENV
        'problem': args.problem,
        'train_graph_type': args.train_graph_type,
        'initialization': args.initialization,
        'conn_low': args.conn_low,
        'conn_high': args.conn_high,
        'distance_metric': args.distance_metric,
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
        # Memory
        'memory_type': args.memory_type,
        'memory_size': args.memory_size,
        'k': args.k,
        'mem_aggr': args.mem_aggr,
        'revisit_punishment': args.revisit_punishment,
        'mem_value_type': args.mem_value_type,
        # Training
        'n_epochs': args.n_epochs,
        'n_episodes': args.n_episodes,
        'train_max_steps_multiplier': args.train_max_steps_multiplier,
        'train_patience': args.train_patience,
        'batch_size': args.batch_size,
        'pop_size': args.pop_size,
        'min_problem_size': args.min_problem_size,
        'max_problem_size': args.max_problem_size,
        'gamma': args.gamma,
        'max_grad_norm': args.max_grad_norm,
        'normalize_advantages': args.normalize_advantages,
        # PPO
        'ppo_epochs': args.ppo_epochs,
        'ppo_clip': args.ppo_clip,
        'entropy_coef': args.entropy_coef,
        'n_stored_states': args.n_stored_states,
        'ppo_update_batch_count': args.ppo_update_batch_count,
        # v2 improvements
        'kl_target': args.kl_target,
        # Optimizer
        'lr': args.lr,
        'betas': args.betas,
        'weight_decay': args.weight_decay,
        # Eval
        'eval_graph_types': args.eval_graph_types,
        'num_eval_graphs': args.num_eval_graphs,
        'test_pop_size': args.test_pop_size,
        'test_max_steps_multiplier': args.test_max_steps_multiplier,
        'test_patience': args.test_patience,
        'topk': args.topk,
        # Others
        'ni_model_load_path': args.ni_model_load_path,
        'save_models': args.save_models,
        'eval_freq': args.eval_freq,
        'seed': args.seed,
        'wandb': args.wandb,
        'run_id': args.run_id,
        'verbose': args.verbose,
        'debug': args.debug,
        'amp': args.amp,
    }

    return all_args
