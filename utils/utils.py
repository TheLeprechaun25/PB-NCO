import random
import string
import time
import networkx as nx
import numpy as np
import pickle
import torch


def load_test_data(problem, graph_types, test_batch_sizes):
    if problem == 'mc':
        return load_test_mc_data(graph_types, test_batch_sizes)
    elif problem == 'mis':
        return load_test_mis_data(graph_types, test_batch_sizes)
    else:
        return None, None


def load_test_mc_data(graph_types, test_batch_sizes):
    test_graphs = []
    for graph_type, test_batch_size in zip(graph_types, test_batch_sizes):
        cur_test_graphs = []
        if graph_type == 'ER700_800':
            for instance_id in range(test_batch_size):
                graph_path = f"data/ER700_800/ER_700_800_0.15_{instance_id}.gpickle"
                g = pickle.load(open(graph_path, 'rb'))
                cur_test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        elif graph_type in ['RB200_300', 'RB800_1200']:
            for instance_id in range(test_batch_size):
                if graph_type == 'RB200_300':
                    graph_path = f"data/RB200_300/GR_200_300_{instance_id}.gpickle"
                else:
                    graph_path = f"data/RB800_1200/GR_800_1200_{instance_id}.gpickle"
                g = pickle.load(open(graph_path, 'rb'))
                cur_test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        else:  # ER{N} or BA{N}
            test_batch_path = f"data/{graph_type}/{graph_type}_100graphs.pkl"
            cur_test_graphs = [torch.tensor(np.array(pickle.load(open(test_batch_path, 'rb'))))[:test_batch_size]]

        test_graphs.append(cur_test_graphs)
    return test_graphs


def load_test_mis_data(graph_types, test_batch_sizes):
    test_graphs = []
    for graph_type, test_batch_size in zip(graph_types, test_batch_sizes):
        cur_test_graphs = []
        if graph_type == 'ER700_800':
            for instance_id in range(test_batch_size):
                graph_path = f"data/ER700_800/ER_700_800_0.15_{instance_id}.gpickle"
                g = pickle.load(open(graph_path, 'rb'))
                cur_test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        elif graph_type in ['RB200_300', 'RB800_1200']:
            for instance_id in range(test_batch_size):
                if graph_type == 'RB200_300':
                    graph_path = f"data/RB200_300/GR_200_300_{instance_id}.gpickle"
                else:
                    graph_path = f"data/RB800_1200/GR_800_1200_{instance_id}.gpickle"
                g = pickle.load(open(graph_path, 'rb'))
                cur_test_graphs.append(torch.tensor(nx.to_numpy_array(g), dtype=torch.float32).to("cpu"))

        else:  # ER{N} or BA{N}
            test_batch_path = f"data/{graph_type}/{graph_type}_100graphs.pkl"
            cur_test_graphs = [torch.tensor(np.array(pickle.load(open(test_batch_path, 'rb'))))[:test_batch_size]]

        test_graphs.append(cur_test_graphs)
    return test_graphs


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_forgiving(model, state_dict, *, strip_module_prefix=True, allow_partial=False, verbose=True):
    msd = model.state_dict()

    # 1) Optional: strip "module." from DataParallel checkpoints
    if strip_module_prefix:
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    # 2) Build a new state dict that only contains compatible tensors
    new_sd = {}
    skipped = {}
    for k, v in state_dict.items():
        if k not in msd:
            continue

        tgt = msd[k]
        if v.shape == tgt.shape:
            new_sd[k] = v
        elif allow_partial and v.ndim == tgt.ndim:
            # Partial load: copy the overlapping block
            t = tgt.clone()
            sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, tgt.shape))
            t[sl] = v[sl].to(t.dtype)
            new_sd[k] = t
        else:
            skipped[k] = (tuple(v.shape), tuple(tgt.shape))

    # 3) Load without complaining about missing/unexpected keys
    msg = model.load_state_dict(new_sd, strict=False)

    if verbose:
        print(f"Loaded {len(new_sd)} tensors.")
        if skipped:
            print("Skipped/mismatched tensors (ckpt -> model):")
            for k, (a, b) in skipped.items():
                print(f"  {k}: {a} -> {b}")
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

    return msg


def generate_word(length):
    VOWELS = "aeiou"
    CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))
    word = ""
    for i in range(length):
        if i % 2 == 0:
            word += random.choice(CONSONANTS)
        else:
            word += random.choice(VOWELS)
    return word


def format_duration(secs):
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02}m {s:02}s"


def print_epoch_header(epoch, n_epochs, start_time):
    elapsed = time.time() - start_time
    elapsed_str = format_duration(elapsed)
    if epoch > 1:
        eta = elapsed / (epoch - 1) * (n_epochs - epoch + 1)
        eta_str = format_duration(eta)
    else:
        eta_str = "--"
    print(f"\nEpoch {epoch}/{n_epochs}. Elapsed: {elapsed_str}. ETA: {eta_str}")


def print_epoch_results(epoch_results):
    avg_total_reward = epoch_results["train/avg_reward_values"][-1]
    revisited = epoch_results["train/revisited"][-1]
    avg_similarity = epoch_results["train/avg_similarity"][-1]
    max_similarity = epoch_results["train/max_similarity"][-1]
    self_mem_percentage = epoch_results["train/self_mem_percentage"][-1]
    total_steps = epoch_results["train/total_steps"][-1]

    print(f"Reward: {avg_total_reward:.3f}. self-mem-perc: {self_mem_percentage:.3f}. "
          f"Revisit: {revisited:.3f}. Avg/Max Sim: {avg_similarity:.2f}/{max_similarity:.2f}. "
          f"{total_steps} steps.")


def is_pareto_efficient(points):
    """
    Return a boolean mask of shape (n_points,) where True indicates
    the corresponding point is Pareto-efficient (i.e. not dominated by any other).
    Assumes you want to MAXIMIZE all objectives in points[:,0], points[:,1], …
    """
    n_points = points.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_efficient[i]:
            # already known to be dominated; skip
            continue
        # any point that dominates point i?
        # j dominates i if all(points[j] <= points[i]) and any(points[j] < points[i])
        domination = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        # mark all dominated points (other than i itself)
        is_efficient[domination] = False
        # keep i itself
        is_efficient[i] = True
    return is_efficient
