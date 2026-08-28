import torch


def _to_spins(x: torch.Tensor) -> torch.Tensor:
    return (x * 2 - 1) if x.min() >= 0 else x


def _to_bits01(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).float() if x.min() < 0 else x.float()


@torch.no_grad()
def _node_sim01_topk_maxcut(cur: torch.Tensor, mem: torch.Tensor, k: int):
    """Flip-invariant node similarity (Max-Cut): sim = |s·m|/N after mapping to spins."""
    P, N = cur.shape
    s_sp = _to_spins(cur).float()
    m_sp = _to_spins(mem).float()
    overlap = torch.mm(s_sp, m_sp.t()).abs()   # flip-invariant
    sim01 = overlap / N
    return torch.topk(sim01, k, largest=True, sorted=True)


@torch.no_grad()
def _node_sim01_topk_mis(cur: torch.Tensor, mem: torch.Tensor, k: int):
    """Plain node similarity for MIS on {0,1}: sim = 1 - Hamming_frac = 1 - mean XOR."""
    s01 = _to_bits01(cur)            # (P,N)
    m01 = _to_bits01(mem)            # (U,N)
    # pairwise XOR mean: (P,U)
    # (s01.unsqueeze(1) != m01.unsqueeze(0)) -> (P,U,N)
    ham = (s01.unsqueeze(1) != m01.unsqueeze(0)).float().mean(dim=-1)
    sim01 = 1.0 - ham
    return torch.topk(sim01, k, largest=True, sorted=True)


@torch.no_grad()
def _edge_topk_qform(cur: torch.Tensor,
                     mem: torch.Tensor,
                     k: int,
                     W: torch.Tensor,
                     W_sum: torch.Tensor,
                     u_chunk: int = 1024,
                     p_chunk: int = 128):
    """
    Compute top-k edge-similarity without materializing (P,U,E).
    sim = 0.5 + (a^T W a) / (2 * W.sum), with a = z_s ⊙ z_m in spins.
    """
    device = W.device
    # to spins in {-1,+1}, keep float32 for accumulations
    S = (_to_spins(cur).to(device=device, dtype=torch.float32))   # (P,N)
    M = (_to_spins(mem).to(device=device, dtype=torch.float32))   # (U,N)

    P, N = S.shape
    U, _ = M.shape
    k = min(k, U)

    # try a lower-precision W to save memory/bandwidth (optional)
    W_fast = W
    if device.type == "cuda":
        try:
            W_fast = W.to(torch.bfloat16)
        except Exception:
            pass

    best_vals = torch.full((P, k), -1e9, device=device, dtype=torch.float32)
    best_idx  = torch.full((P, k), -1,   device=device, dtype=torch.long)

    for p0 in range(0, P, p_chunk):
        p1 = min(P, p0 + p_chunk)
        S_blk = S[p0:p1]   # (Pc, N)

        row_vals = torch.full((p1 - p0, k), -1e9, device=device, dtype=torch.float32)
        row_idx  = torch.full((p1 - p0, k), -1,   device=device, dtype=torch.long)

        for u0 in range(0, U, u_chunk):
            u1 = min(U, u0 + u_chunk)
            M_blk = M[u0:u1]     # (Uc, N)

            # Build all pairwise a = s ⊙ m for the block, flattened as (Pc*Uc, N)
            A = (S_blk.unsqueeze(1) * M_blk.unsqueeze(0)).reshape(-1, N)

            # Quadratic form a^T W a via (A @ W) • A
            if W_fast.dtype != torch.float32:
                A_fast = A.to(W_fast.dtype)
                Y = A_fast @ W_fast                     # (Pc*Uc, N)
                quad = (Y * A_fast).sum(dim=1).to(torch.float32)
            else:
                Y = A @ W_fast
                quad = (Y * A).sum(dim=1)              # (Pc*Uc,)

            sim = 0.5 + 0.5 * quad / W_sum             # (Pc*Uc,)
            sim = sim.reshape(p1 - p0, u1 - u0)

            # local top-k in this chunk, then merge with running top-k
            v_local, i_local = torch.topk(sim, k=min(k, u1 - u0), dim=1)
            i_local += u0  # make indices global along U

            merged_v = torch.cat([row_vals, v_local], dim=1)
            merged_i = torch.cat([row_idx,  i_local], dim=1)
            row_vals, sel = torch.topk(merged_v, k, dim=1)
            row_idx  = torch.gather(merged_i, 1, sel)

        best_vals[p0:p1] = row_vals
        best_idx[p0:p1]  = row_idx

    return best_vals, best_idx


def make_metric(distance_metric: str, problem, adj_matrix: torch.Tensor, state_dim: int, device: str):
    """
    Returns a callable:
        topk(cur_state: (P,N), memory: (U,N), k:int) -> (sim01(P,k), idx(P,k))
    and, for 'edge', a small context with cached (i,j,w,mem_cuts).
    """
    name = distance_metric.lower()
    if name == "node_hamming":
        if problem == "mc":
            return {"name": "node_hamming_maxcut", "topk": _node_sim01_topk_maxcut, "ctx": None}
        elif problem == "mis":
            return {"name": "node_hamming_mis", "topk": _node_sim01_topk_mis, "ctx": None}
        else:
            raise ValueError("problem must be 'maxcut' or 'mis'")

    if name == "edge_hamming":
        if problem != "mc":
            raise ValueError("edge_hamming is for Max-Cut; not appropriate for MIS.")

        # Prepare dense W and its sum once (no E tensors)
        A = adj_matrix.to(device=device, dtype=torch.float32)
        if A.dim() == 3:
            A = A[0]
        W = A  # symmetric, zero diagonal
        W_sum = W.sum().clamp_min(1e-12)  # equals 2*S

        ctx = {"W": W, "W_sum": W_sum}

        @torch.no_grad()
        def topk(cur: torch.Tensor, mem: torch.Tensor, k: int):
            return _edge_topk_qform(cur, mem, k, ctx["W"], ctx["W_sum"])

        return {"name": "edge_hamming_qform", "topk": topk, "ctx": ctx}

    raise ValueError(f"Unknown distance_metric: {distance_metric}")


class Memory:
    def __init__(self, mem_type, state_dim, memory_aggr, value_type, distance_metric, adj_matrix, batch_size, pop_size, n_memories, problem, memory_limit=100000, device='cpu'):
        """
        Memory that saves State-Action pairs and returns the average value of the k-nearest neighbours of a state
        Key: Solution state of the problem
        Value: One-hot encoding of the performed action
        """
        self.mem_type = mem_type
        self.state_dim = state_dim  # Problem size
        self.action_dim = state_dim  # Problem size
        self.memory_limit = memory_limit
        self.memory_aggr = memory_aggr
        self.mem_value_type = value_type
        # A graph-dependent metric must be built per memory/instance.  In
        # particular, edge-Hamming caches the adjacency matrix; reusing a
        # metric built from adj_matrix[0] makes every other graph in a batch
        # query its memory with the wrong topology.
        if adj_matrix.dim() == 2:
            graph_matrices = [adj_matrix]
        else:
            graph_matrices = [adj_matrix[b] for b in range(batch_size)]
        self.sim_metrics = [
            make_metric(
                distance_metric,
                problem,
                graph_matrices[idx if n_memories == batch_size else idx // pop_size],
                state_dim,
                device,
            )
            for idx in range(n_memories)
        ]
        self.adj_matrix = adj_matrix
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_memories = n_memories # Number of memories to use
        self.shared_memory = (self.n_memories == self.batch_size) and self.pop_size > 1
        self.batch_pop_size = batch_size * pop_size
        self.batch_range = torch.arange(batch_size, device=device)
        self.batch_pop_range = torch.arange(self.batch_pop_size, device=device)
        self.device = device

        # Initialize memories and index
        self.state_memories = [torch.zeros((0, self.state_dim), device=device) for _ in range(self.n_memories)]
        self.action_memories = [torch.zeros((0, self.state_dim, 2), device=device) for _ in range(self.n_memories)]

        self.used_memory = 0

    def save_in_memory(self, state, action):
        assert self.batch_size*self.pop_size == state.shape[0], "State shape does not match batch size x pop size"

        state = state.to(self.device)  # Ensure state is on CPU for memory operations
        action = action.to(self.device)  # Ensure action is on CPU for memory operations

        # Get the current values in solutions
        cur_values = state[self.batch_pop_range, action]

        # Double the actions so that we can distinguish between 0->1 and 1->0
        double_action = action.clone()
        double_action[cur_values == 1] += self.state_dim

        # Perform one-hot encoding
        one_hot_actions = torch.nn.functional.one_hot(double_action, num_classes=2*self.state_dim).float()
        one_hot_actions = one_hot_actions.view(self.batch_pop_size, 2, self.state_dim)
        one_hot_actions = one_hot_actions.transpose(1, 2).contiguous().view(self.batch_pop_size, self.state_dim, 2)

        # Reshape state and actions to share the memory among all threads
        if self.shared_memory:
            state = state.reshape(self.batch_size, self.pop_size, self.state_dim)
            one_hot_actions = one_hot_actions.reshape(self.batch_size, self.pop_size, self.state_dim, 2)

        for idx in range(self.n_memories):
            if self.used_memory >= self.memory_limit:
                # If memory is full, remove the oldest state
                if self.shared_memory:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -self.pop_size, dims=0)
                    self.state_memories[idx][-self.pop_size:] = state[idx]
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -self.pop_size, dims=0)
                    self.action_memories[idx][-self.pop_size:] = one_hot_actions[idx]
                else:
                    self.state_memories[idx] = torch.roll(self.state_memories[idx], -1, dims=0)
                    self.state_memories[idx][-1] = state[idx].unsqueeze(0)
                    self.action_memories[idx] = torch.roll(self.action_memories[idx], -1, dims=0)
                    self.action_memories[idx][-1] = one_hot_actions[idx].unsqueeze(0)

            else:
                # Save state + action in memory
                if self.shared_memory:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], state[idx]])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions[idx]])
                else:
                    self.state_memories[idx] = torch.vstack([self.state_memories[idx], state[idx].unsqueeze(0)])
                    self.action_memories[idx] = torch.vstack([self.action_memories[idx], one_hot_actions[idx].unsqueeze(0)])

            # Keep the configured capacity exact even when memory_limit is not
            # divisible by the population size.
            if self.state_memories[idx].shape[0] > self.memory_limit:
                self.state_memories[idx] = self.state_memories[idx][-self.memory_limit:]
                self.action_memories[idx] = self.action_memories[idx][-self.memory_limit:]

        if self.shared_memory:
            self.used_memory = min(self.memory_limit, self.used_memory + self.pop_size)
        else:
            self.used_memory = min(self.memory_limit, self.used_memory + 1)

    def get_knn(self, state, initial_k):
        assert state.shape[0] == self.batch_size*self.pop_size, "State shape does not match batch size x pop size"

        # Get k nearest states
        state = state.float().to(self.device)
        k = self.used_memory if initial_k > self.used_memory else initial_k

        if self.shared_memory:
            state = state.reshape(self.batch_size, self.pop_size, self.state_dim)
            avg_similarity = torch.zeros((self.batch_size, self.pop_size), device=state.device)
            max_similarity = torch.zeros((self.batch_size, self.pop_size), device=state.device)
            revisited = torch.zeros((self.batch_size, self.pop_size), device=state.device)
            total_revisited = torch.zeros((self.batch_size, self.pop_size), device=state.device)
            gathering_indices = torch.zeros((self.batch_size, self.pop_size, initial_k), device=state.device)
            self_percentage = torch.zeros((self.batch_size, self.pop_size), device=state.device)
        else:
            avg_similarity = torch.zeros(self.batch_pop_size, device=state.device)
            max_similarity = torch.zeros(self.batch_pop_size, device=state.device)
            revisited = torch.zeros(self.batch_pop_size, device=state.device)
            total_revisited = torch.zeros(self.batch_pop_size, device=state.device)
            gathering_indices = torch.zeros((self.batch_pop_size, initial_k), device=state.device)
            self_percentage = torch.zeros(self.batch_pop_size, device=state.device)

        nearest_values = []

        for idx in range(self.n_memories):
            cur_state = state[idx] if self.shared_memory else state[idx].unsqueeze(0)
            # cur_state.shape = (pop_size, state_dim) if shared, else (1, state_dim)
            similarity, indices = self.sim_metrics[idx]["topk"](cur_state, self.state_memories[idx], k)  # sim01 ∈ [0,1]

            revisited[idx] = (similarity >= 1 - 1e-12).sum(axis=1) #(similarity == self.state_dim).sum(axis=1)
            eq = (cur_state.unsqueeze(1) == self.state_memories[idx].unsqueeze(0)).all(-1)
            total_revisited[idx] = eq.sum(-1).float()
            avg_similarity[idx] = torch.mean(similarity, dim=1)
            max_similarity[idx] = similarity[:, 0]

            cur_indices = indices if self.shared_memory else indices.flatten()

            if self.shared_memory:
                arange = torch.arange(self.pop_size, device=state.device)
                a = indices % self.pop_size
                b = a == arange.unsqueeze(1).repeat(1, k)
                self_percentage[idx] = b.float().mean(dim=1)

                gathering_indices[idx, :, :k] = cur_indices

            if self.mem_value_type == 'actions':
                nearest_vals = self.action_memories[idx][cur_indices, :].reshape(indices.shape + (self.state_dim, 2))
            elif self.mem_value_type in ['solutions', 'differences']:
                nearest_vals = self.state_memories[idx][cur_indices, :].reshape(indices.shape + (self.state_dim, ))
            elif self.mem_value_type == 'combined':
                acts = self.action_memories[idx][cur_indices, :].reshape(indices.shape + (self.state_dim, 2))
                # differences: 0/1 difference vs current state (..., N, 1)
                state_for_diff = state[idx].unsqueeze(1) if self.shared_memory else state[idx].unsqueeze(0).unsqueeze(1)
                sols = self.state_memories[idx][cur_indices, :].reshape(indices.shape + (self.state_dim,))
                diffs = (torch.abs(sols - state_for_diff) / 2.0).unsqueeze(-1)  # (..., N, 1)
                # combined: (..., N, 3) -> [act0, act1, diff]
                nearest_vals = torch.cat((acts, diffs), dim=-1)
            else:
                raise NotImplementedError

            # Aggregate among k neighbors: Weighted based on similarity.
            if self.memory_aggr == 'sum':  # No weighting
                if self.mem_value_type == 'differences':
                    state_for_diff = state[idx].unsqueeze(1) if self.shared_memory else state[idx].unsqueeze(0).unsqueeze(1)
                    differences = torch.abs(nearest_vals - state_for_diff) / 2
                    nearest_values.append(torch.sum(differences, dim=1))
                else:
                    nearest_values.append(torch.sum(nearest_vals, dim=1))
            elif self.memory_aggr == 'linear':  # Linear weighted sum
                sim = similarity
                if self.mem_value_type == 'actions':
                    nearest_values.append(torch.sum(nearest_vals * sim[:, :, None, None], dim=1))
                elif self.mem_value_type == 'solutions':
                    nearest_values.append(torch.sum(nearest_vals * sim[:, :, None], dim=1))
                elif self.mem_value_type == 'differences':
                    state_for_diff = state[idx].unsqueeze(1) if self.shared_memory else state[idx].unsqueeze(0).unsqueeze(1)
                    differences = torch.abs(nearest_vals - state_for_diff) / 2
                    nearest_values.append(torch.sum(differences * sim[:, :, None], dim=1))
                elif self.mem_value_type == 'combined':
                    nearest_values.append(torch.sum(nearest_vals * sim[:, :, None, None], dim=1))
                else:
                    raise NotImplementedError

            elif self.memory_aggr == 'exp':  # Exponential weighted sum
                sim = similarity
                if self.mem_value_type == 'actions':
                    nearest_values.append(torch.sum(nearest_vals * (torch.exp(torch.log(torch.tensor(2)) * sim[:, :, None, None]) - 1), dim=1))
                elif self.mem_value_type == 'solutions':
                    nearest_values.append(torch.sum(nearest_vals * (torch.exp(torch.log(torch.tensor(2)) * sim[:, :, None]) - 1), dim=1))
                elif self.mem_value_type == 'differences':
                    state_for_diff = state[idx].unsqueeze(1) if self.shared_memory else state[idx].unsqueeze(0).unsqueeze(1)
                    differences = torch.abs(nearest_vals - state_for_diff) / 2
                    nearest_values.append(torch.sum(differences * (torch.exp(torch.log(torch.tensor(2, device=sim.device)) * sim[:, :, None]) - 1), dim=1))
                elif self.mem_value_type == 'combined':
                    nearest_values.append(torch.sum(nearest_vals * (torch.exp(torch.log(torch.tensor(2)) * sim[:, :, None, None]) - 1), dim=1))
                else:
                    raise NotImplementedError
                # nearest_values.shape = (batch_size, state_dim, 2)
            elif self.memory_aggr == 'concat':
                # nearest_acts.shape = (pop_size, k, state_dim, 2) from (pop_size, k, state_dim, 2) to (pop_size, state_dim, 2, k)
                if self.mem_value_type == 'actions':
                    nearest_values.append(nearest_vals.permute(0, 2, 3, 1))
                elif self.mem_value_type == 'solutions':
                    nearest_values.append(nearest_vals.permute(0, 2, 1))
                elif self.mem_value_type == 'differences':
                    state_for_diff = state[idx].unsqueeze(1) if self.shared_memory else state[idx].unsqueeze(0).unsqueeze(1)
                    differences = torch.abs(nearest_vals - state_for_diff) / 2
                    nearest_values.append(differences.permute(0, 2, 1))
                elif self.mem_value_type == 'combined':
                    nearest_values.append(nearest_vals.permute(0, 2, 3, 1))
                else:
                    raise NotImplementedError

        nearest_values = torch.stack(nearest_values, dim=0)
        # Reshape
        if self.memory_aggr == 'concat':
            if self.mem_value_type in ['combined', 'actions']:
                batch, pop, state, C, k = nearest_values.shape
                # Create a new zeros tensor of shape (batch, pop, state, C, initial_k)
                padded = torch.zeros(batch, pop, state, C, initial_k, device=nearest_values.device, dtype=nearest_values.dtype)
                # Copy the old data into the first k slots along the last dimension
                padded[..., :k] = nearest_values
                # Finally, merge (batch, pop) → batch*pop and (C, initial_k) → C*initial_k:
                nearest_values = padded.reshape(batch * pop, state, C * initial_k)
            else:
                batch, pop, state, k = nearest_values.shape
                # Create a new zeros tensor of shape (batch, pop, state, initial_k)
                padded = torch.zeros(batch, pop, state, initial_k)
                # Copy the old data into the first k slots along the last dimension
                padded[..., :k] = nearest_values
                # Finally, merge (batch, pop) → batch*pop:
                nearest_values = padded.reshape(batch * pop, state, initial_k)

        else:
            if self.mem_value_type == 'actions':
                nearest_values = nearest_values.reshape(self.batch_pop_size, self.state_dim, 2)
            elif self.mem_value_type == 'combined':
                nearest_values = nearest_values.reshape(self.batch_pop_size, self.state_dim, 3)
            else:
                nearest_values = nearest_values.reshape(self.batch_pop_size, self.state_dim, 1)

        revisited = revisited.reshape(self.batch_pop_size)
        avg_similarity = avg_similarity.reshape(self.batch_pop_size)
        max_similarity = max_similarity.reshape(self.batch_pop_size)

        return nearest_values, revisited, total_revisited, avg_similarity, max_similarity, self_percentage, gathering_indices

    def clear_memory(self):
        """
        Clears the memory
        """
        # Reset memories and index
        self.state_memories = [torch.zeros((0, self.state_dim), device=self.device) for _ in range(self.n_memories)]
        self.action_memories = [torch.zeros((0, self.state_dim, 2), device=self.device) for _ in range(self.n_memories)]
        self.used_memory = 0


def select_memory(memory_type, mem_aggr, value_type, state_dim, distance_metric, adj_matrix,
                  batch_size, pop_size, problem, device, memory_limit=100000):
    batch_pop_size = batch_size * pop_size

    if memory_type == 'none':
        n_memories = batch_size if pop_size > 1 else batch_pop_size
    elif memory_type in {'shared', 'marco_shared'}:
        n_memories = batch_size
    elif memory_type == 'individual':
        n_memories = batch_pop_size
    else:
        raise NotImplementedError

    memory = Memory(
        mem_type=memory_type,
        value_type=value_type,
        state_dim=state_dim,
        memory_aggr=mem_aggr,
        distance_metric=distance_metric,
        adj_matrix=adj_matrix,
        batch_size=batch_size,
        pop_size=pop_size,
        problem=problem,
        n_memories=n_memories,
        memory_limit=memory_limit,
        device=device
    )

    return memory
