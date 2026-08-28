from dataclasses import dataclass, fields, replace
from typing import Optional
import torch


@dataclass
class State:
    batch_size: int
    pop_size: int
    problem_size: int
    graph: torch.Tensor
    solutions: Optional[torch.Tensor] = None
    ising_solutions: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    obj_values: Optional[torch.FloatTensor] = None
    mem_info: Optional[torch.FloatTensor] = None
    extra_node_feats: Optional[torch.Tensor] = None
    visited_solutions: Optional[torch.Tensor] = None
    archive_probs: Optional[torch.Tensor] = None
    exploration_weight: Optional[torch.Tensor] = None
    greedy_inference_hint: Optional[torch.Tensor] = None
    testing: bool = False

    def to(self, device: torch.device):
        """
        Returns a copy of this State with all torch.Tensor fields moved to `device`.
        Non‐Tensor fields are left untouched.
        """
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                kwargs[f.name] = val.to(device)
            else:
                kwargs[f.name] = val
        return replace(self, **kwargs)

    def cpu(self):
        return self.to(torch.device('cpu'))

    def cuda(self, device_id: int = 0):
        return self.to(torch.device(f'cuda:{device_id}'))


def pop_diversity(diversity_metric):
    if diversity_metric == 'node_hamming':
        return node_hamming_pop_diversity
    elif diversity_metric == 'edge_hamming':
        return edge_hamming_pop_diversity
    elif diversity_metric == 'coverage_jaccard':
        return coverage_jaccard_pop_diversity
    else:
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")


def distance_fn(diversity_metric):
    if diversity_metric == 'node_hamming':
        return node_hamming_distance
    elif diversity_metric == 'edge_hamming':
        return edge_hamming_distance
    elif diversity_metric == 'coverage_jaccard':
        return coverage_jaccard_distance
    else:
        raise ValueError(f"Unknown diversity metric: {diversity_metric}")


@torch.no_grad()
def node_hamming_pop_diversity(S: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Node-Hamming diversity for binary problems.

    Args
    ----
    S : (B, P, N) spins in {-1,+1} or {0,1}
    adj : unused (for compatibility)

    Returns
    -------
    torch.Tensor scalar in [0,1]:
        mean over batch of mean pairwise node-Hamming distances across the population.
    """
    device = S.device
    S = S.to(device).float()
    B, P, N = S.shape

    if P == 1:
        return S.new_tensor(0.0)

    # Pairwise node disagreements D[b,p,q,n] ∈ {0,1}
    # Robust to either {-1,+1} or {0,1} encodings
    if (S.min() >= 0).item():
        D = (S[:, :, None, :] != S[:, None, :, :]).float()     # {0,1} case
    else:
        D = (S[:, :, None, :] * S[:, None, :, :] < 0).float()  # {-1,+1} case

    # Mean over nodes -> pairwise distances in [0,1]
    D = D.mean(dim=3)                                        # (B, P, P)

    # Mean over unique population pairs, then mean over batch
    iu = torch.triu_indices(P, P, 1, device=device)
    per_inst = D[:, iu[0], iu[1]].mean(dim=1)                # (B,)
    return per_inst.mean()


@torch.no_grad()
def edge_hamming_pop_diversity(S: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Weighted edge-disagreement diversity for Max-Cut.

    Args
    ----
    S   : (B, P, N) spins in {-1,+1} or {0,1}
    adj : (B, N, N) or (N, N) symmetric adjacency / edge weights (any nonneg float)

    Returns
    -------
    torch.Tensor scalar in [0,1]:
        mean over batch of mean pairwise weighted edge-disagreement across the population.
    """
    device = S.device
    S = S.to(device).float()
    B, P, N = S.shape

    if P == 1:
        return S.new_tensor(0.0)

    # Ensure batched adjacency on the right device
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(B, -1, -1)
    adj = adj.to(device).float()

    # Upper-triangular edges (avoid double counting)
    ij = torch.triu_indices(N, N, 1, device=device)
    i, j = ij[0], ij[1]
    E = i.numel()
    if E == 0:
        return S.new_tensor(0.0)

    # Cut indicator C[b,p,e] ∈ {0,1}
    # Robust to either {-1,+1} or {0,1} encodings
    if (S.min() >= 0).item():
        C = (S[:, :, i] != S[:, :, j]).float()            # {0,1} case
    else:
        C = (S[:, :, i] * S[:, :, j] < 0).float()         # {-1,+1} case

    # Edge weights and normalization
    w = adj[:, i, j].clamp_min(0)                         # (B, E)
    totW = w.sum(dim=1, keepdim=True).clamp_min(1e-12)    # (B, 1)

    # Weighted intersection trick:
    # xor_w = sum_e w_e (Cp[e] XOR Cq[e]) = cut_w[p] + cut_w[q] - 2 * inter_w[p,q]
    Cw    = C * w[:, None, :]                              # (B, P, E)
    cut_w = Cw.sum(dim=2, keepdim=True)                    # (B, P, 1)
    inter = torch.matmul(Cw, C.transpose(1, 2))            # (B, P, P)
    xor_w = cut_w + cut_w.transpose(1, 2) - 2.0 * inter    # (B, P, P)
    D     = (xor_w / totW.view(B, 1, 1)).clamp_(0.0, 1.0)  # (B, P, P) in [0,1]

    # Mean over unique population pairs, then mean over batch
    iu = torch.triu_indices(P, P, 1, device=device)
    per_inst = D[:, iu[0], iu[1]].mean(dim=1)              # (B,)
    return per_inst.mean()


@torch.no_grad()
def coverage_jaccard_pop_diversity(S: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Coverage-Jaccard diversity on closed neighborhoods N[S] for MIS.

    Args
    ----
    S : (B, P, N) solutions in {0,1} or {-1,+1}
    adj : (N, N) or (B, N, N) adjacency (0 diag; can be weighted)

    Returns
    -------
    scalar tensor in [0,1]: mean over batch of mean pairwise distances across the P solutions.
    """
    device = S.device
    B, P, N = S.shape

    if P == 1:
        return S.new_tensor(0.0)

    # Ensure {0,1}
    if S.min() < 0:
        S01 = (S > 0).to(torch.float32)
    else:
        S01 = S.to(torch.float32)

    # Batched adjacency
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(B, -1, -1)
    adj = adj.to(device, dtype=S01.dtype)

    # Closed neighborhood coverage C = 1 if selected OR neighbor-of-selected
    # counts = S01 @ adj  -> (B, P, N); >0 marks covered neighbors
    nbr_counts = torch.einsum('bpn,bnm->bpm', S01, adj)           # (B,P,N)
    C = torch.clamp(S01 + (nbr_counts > 0).to(S01.dtype), max=1.)

    # Node weights (degree-based, normalized per instance)
    deg = adj.sum(dim=-1)                                        # (B,N)
    w = (deg / deg.sum(dim=1, keepdim=True).clamp_min(1e-12)).unsqueeze(1)  # (B,1,N)

    # Weighted intersection / union on nodes
    Cw = C * w                                                 # (B,P,N)
    inter = torch.einsum('bpn,bqn->bpq', Cw, C)                # (B,P,P)
    sums  = Cw.sum(dim=2, keepdim=True)                        # (B,P,1)
    union = (sums + sums.transpose(1,2) - inter).clamp_min(1e-12)
    D = (1.0 - inter / union).clamp(0., 1.)                    # (B,P,P)

    # Mean over unique pairs, then over batch
    iu = torch.triu_indices(P, P, 1, device=device)
    per_inst = D[:, iu[0], iu[1]].mean(dim=1) if iu.numel() else D.new_zeros(B)
    return per_inst.mean()


@torch.no_grad()
def node_hamming_distance(s: torch.Tensor, visited: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Non-flip-invariant node Hamming distance between one solution and an archive.

    Args
    ----
    s       : (B, N) in {0,1} or {-1,+1}
    visited : (B, N, K) in {0,1} or {-1,+1}
    adj     : unused (for API symmetry)

    Returns
    -------
    (B,) in [0,1] — average Hamming fraction of s to each visited solution.
    """
    device = s.device
    dtype  = torch.float32

    # Canonicalize to {0,1} so distance is non–flip-invariant
    s01 = (s > 0).to(dtype)                                   # (B, N)
    V01 = (visited > 0).to(dtype).permute(0, 2, 1).contiguous()  # (B, K, N)

    B, K, N = V01.shape
    if K == 0:
        return torch.zeros(B, device=device, dtype=dtype)

    # Hamming fraction via XOR and mean over nodes: (B, K)
    D = (s01.unsqueeze(1) != V01).float().mean(dim=-1)

    # Average over archive members -> (B,)
    return D.mean(dim=1)


@torch.no_grad()
def edge_hamming_distance(s: torch.Tensor, visited: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    (Weighted) edge-Hamming distance between one solution and an archive.

    Args
    ----
    s       : (B, N) in {0,1} or {-1,+1}
    visited : (B, N, K) in {0,1} or {-1,+1}
    adj     : (N, N) or (B, N, N) symmetric adjacency / edge weights (nonneg). Diagonal ignored.

    Returns
    -------
    (B,) in [0,1] — average (weighted) edge-Hamming distance of s to visited.
    """
    device = s.device
    dtype  = torch.float32

    # Normalize to a consistent 0/1 encoding for cut indicator computation
    s01 = (s > 0).to(dtype) if s.min() < 0 else s.to(dtype)      # (B, N)
    V01 = (visited > 0).to(dtype) if visited.min() < 0 else visited.to(dtype)  # (B, N, K)
    V01 = V01.permute(0, 2, 1).contiguous()  # (B, K, N)
    B, K, N = V01.shape
    if K == 0:
        return torch.zeros(s01.shape[0], device=device, dtype=dtype)

    # Batched adjacency, ensure float and device
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(B, -1, -1)
    adj = adj.to(device=device, dtype=dtype)
    # Upper triangle edges
    ij = torch.triu_indices(N, N, offset=1, device=device)
    i, j = ij[0], ij[1]
    E = i.numel()
    if E == 0:
        return torch.zeros(B, device=device, dtype=dtype)

    # Cut indicators: C_s (B, E), C_V (B, K, E)
    C_s = (s01[:, i] != s01[:, j]).to(dtype)                   # (B, E)
    C_V = (V01[:, :, i] != V01[:, :, j]).to(dtype)             # (B, K, E)

    # Edge weights (B, E) and normalization per instance
    w = adj[:, i, j].clamp_min(0)                              # (B, E)
    totW = w.sum(dim=1, keepdim=True).clamp_min(1e-12)         # (B, 1)

    # XOR disagreement per pair (B, K): sum_e w_e * |C_s - C_V|
    xor_w = torch.abs(C_s.unsqueeze(1) - C_V) * w.unsqueeze(1) # (B, K, E)
    D = xor_w.sum(dim=2) / totW                                # (B, K)
    D = D.clamp_(0.0, 1.0)

    return D.mean(dim=1)                                       # (B,)


@torch.no_grad()
def coverage_jaccard_distance(s: torch.Tensor, visited: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Mean coverage–Jaccard distance between one solution and an archive.

    Args
    ----
    s       : (B, N) in {0,1} or {-1,+1}
    visited : (B, N, K) in {0,1} or {-1,+1}
    adj       : (N, N) or (B, N, N) adjacency (0 diagonal)

    Returns
    -------
    (B,) in [0,1] — average distance of s to each visited solution.
    """
    device = s.device
    dtype  = torch.float32

    # Normalize to {0,1}
    s01 = (s > 0).to(dtype) if s.min() < 0 else s.to(dtype)
    V01 = (visited > 0).to(dtype) if visited.min() < 0 else visited.to(dtype)  # (B, N, K)
    V01 = V01.permute(0, 2, 1).contiguous()  # (B, K, N)
    B, K, N = V01.shape
    if K == 0:
        return torch.zeros(s01.shape[0], device=device, dtype=dtype)

    # Batched adjacency
    if adj.dim() == 2:
        adj = adj.unsqueeze(0).expand(s01.shape[0], -1, -1)
    adj = adj.to(device=device, dtype=dtype)  # (B, N, N)

    # Closed-neighborhood coverage C=1 if selected OR neighbor-of-selected
    # s coverage: (B, N)
    nbr_s = torch.einsum('bn,bnm->bm', s01, adj)              # (B, N)
    Cs    = torch.clamp(s01 + (nbr_s > 0).to(dtype), max=1.)

    # archive coverage: (B, K, N)
    nbr_V = torch.einsum('bkn,bnm->bkm', V01, adj)            # (B, K, N)
    CV    = torch.clamp(V01 + (nbr_V > 0).to(dtype), max=1.)

    # degree weights (normalized per instance)
    deg = adj.sum(dim=-1)                                      # (B, N)
    w   = (deg / deg.sum(dim=1, keepdim=True).clamp_min(1e-12))  # (B, N)

    # weighted intersection & union per archive member
    inter = (w.unsqueeze(1) * (Cs.unsqueeze(1) * CV)).sum(dim=-1)     # (B, K)
    sum_s = (w * Cs).sum(dim=-1, keepdim=True)                        # (B, 1)
    sum_V = (w.unsqueeze(1) * CV).sum(dim=-1)                         # (B, K)
    union = (sum_s + sum_V - inter).clamp_min(1e-12)                  # (B, K)

    D = (1.0 - inter / union).clamp(0., 1.)                  # (B, K)
    return D.mean(dim=1)                                               # (B,)
