import torch
from torch import nn
from nets.layers import GTLayer, Activation


class GraphNIModel(nn.Module):
    def __init__(self, **params):
        super(GraphNIModel, self).__init__()
        self.model_params = params
        self.tanh_clipping = self.model_params['tanh_clipping']
        self.hidden_dim: int = self.model_params['hidden_dim']
        self.n_heads: int = self.model_params['n_heads']
        self.bias: bool = self.model_params['bias']

        # Input Node Encoding
        node_dim = 1
        self.sol_embedding = nn.Embedding(2, self.hidden_dim)

        if self.model_params['memory_type'] != 'none':
            if params['mem_value_type'] == 'actions':
                n_features = 2
            elif params['mem_value_type'] == 'combined':
                n_features = 3
            else:
                n_features = 1
            if params['mem_aggr'] == 'concat':
                n_features *= params['k']

            self.mem_linear = nn.Linear(n_features, self.hidden_dim, bias=self.bias)
            node_dim += 1

            nn.init.orthogonal_(self.mem_linear.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(self.mem_linear.bias, 0.0)

        in_linear1 = nn.Linear(node_dim * self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear1.weight, gain=1.0)
        in_linear2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear2.weight, gain=1.0)
        if self.bias:
            nn.init.constant_(in_linear1.bias, 0.0)
            nn.init.constant_(in_linear2.bias, 0.0)

        self.in_mlp = nn.Sequential(
            in_linear1,
            Activation(self.model_params['activation']),
            nn.Dropout(self.model_params['dropout']),
            in_linear2,
        )

        # GNN layers
        self.encoder_layers = nn.ModuleList([GTLayer(**self.model_params) for _ in range(self.model_params['n_layers'])])

        self.edge_embeddings = nn.ModuleList([nn.Embedding(2, 2 * self.n_heads, device=self.model_params['device'],
                                                           ) for _ in range(self.model_params['n_layers'])])

        # Virtual node
        self.virtual_node = nn.Parameter(torch.randn(1, self.hidden_dim, device=self.model_params['device']))

        # Decoder
        self.decoder = LinearDecoder(**self.model_params)

        # Edge embeddings computed in pre_forward
        self.edge_embeddings_computed = False
        self.e1 = []
        self.e2 = []

    def pre_forward(self, state):
        # Compute edge features just once per instance, since they are fixed.
        _, n = state.ising_solutions.size()

        # Edges
        self.e1 = []
        self.e2 = []

        # Virtual node
        e = state.graph.clone().unsqueeze(1).expand(-1, state.pop_size, -1, -1).reshape(state.batch_size * state.pop_size, state.problem_size, state.problem_size)

        virtual_edges = torch.ones(state.batch_size*state.pop_size, 1, state.problem_size,  dtype=torch.long, device=self.model_params['device'])
        e = torch.cat([e, virtual_edges], dim=1)
        virtual_edges_t = torch.ones(state.batch_size*state.pop_size, state.problem_size + 1, 1, dtype=torch.long, device=self.model_params['device'])
        e = torch.cat([e, virtual_edges_t], dim=2)

        for w_e in self.edge_embeddings:
            e1, e2 = w_e(e).split(self.n_heads, dim=-1)
            e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            self.e1.append(e1)
            self.e2.append(e2)

    def forward(self, state):
        # Node Embeddings: solutions, mem_info, population info
        sol = state.solutions.clone().int().to(self.model_params['device'])

        # Current solutions
        h = self.sol_embedding(sol)

        # Memory info
        if self.model_params['memory_type'] != 'none':
            mem_h = self.mem_linear(state.mem_info)  # Use memory info
            h = torch.cat((h, mem_h), dim=-1)

        # Input MLP
        h = self.in_mlp(h)

        # Edge embeddings
        if state.testing and (not self.edge_embeddings_computed):
            self.pre_forward(state)
            self.edge_embeddings_computed = True

        adj = state.graph.clone().unsqueeze(1).expand(-1, state.pop_size, -1, -1).reshape(state.batch_size * state.pop_size,
                                                                                          state.problem_size, state.problem_size)

        # Virtual node
        virtual_node_features = self.virtual_node.unsqueeze(0).repeat(h.size(0), 1, 1)
        h = torch.cat([h, virtual_node_features], dim=1)

        virtual_edges = torch.ones(state.batch_size*state.pop_size, 1, state.problem_size,  dtype=torch.long, device=self.model_params['device'])
        adj = torch.cat([adj, virtual_edges], dim=1)
        virtual_edges_t = torch.ones(state.batch_size*state.pop_size, state.problem_size + 1, 1, dtype=torch.long, device=self.model_params['device'])
        adj = torch.cat([adj, virtual_edges_t], dim=2)

        # GNN layers
        for idx, layer in enumerate(self.encoder_layers):
            if state.testing and self.edge_embeddings_computed:
                h = layer(h, self.e1[idx], self.e2[idx])
            else:
                e1, e2 = self.edge_embeddings[idx](adj).split(self.n_heads, dim=3)
                e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
                e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
                h = layer(h, e1, e2)

        # Graph Embedding
        graph_embedding = h[:, -1, :]
        h = h[:, :-1, :]

        # Decoder
        out = self.decoder(h, graph_embedding).squeeze(-1)

        # Mask
        if state.mask is not None:
            out[state.mask] = -torch.inf

        return out


class MCNCModel(nn.Module):
    def __init__(self, **params):
        super(MCNCModel, self).__init__()
        self.model_params = params
        self.tanh_clipping = self.model_params['tanh_clipping']
        self.hidden_dim: int = self.model_params['hidden_dim']
        self.n_heads: int = self.model_params['n_heads']
        self.bias: bool = self.model_params['bias']

        n_node_feats = 0
        self.laplace_dim = self.model_params.get('laplace_dim', 0)
        if self.laplace_dim > 0:
            self.laplace_linear = nn.Linear(self.laplace_dim, self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(self.laplace_linear.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(self.laplace_linear.bias, 0.0)
            n_node_feats += 1

        # Input Node embeddings: learnable parameters
        if self.model_params['nc_train_mode'] == 'exploitation':
            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            self.h_0 = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            n_node_feats += 1

        elif self.model_params['nc_train_mode'] in ['exploration', 'exploration_exploitation']: # (using previously visited solutions)
            node_projection = nn.Linear(self.model_params['n_visited_solutions'], self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(node_projection.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(node_projection.bias, 0.0)
            self.node_projection = node_projection

            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            self.h_0 = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            n_node_feats += 2

        elif self.model_params['nc_train_mode'] == 'conditioned_network': # (using previously visited solutions)
            node_projection = nn.Linear(self.model_params['n_visited_solutions'], self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(node_projection.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(node_projection.bias, 0.0)
            self.node_projection = node_projection

            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            self.h_0 = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))

            self.exploration_projection = nn.Linear(1, self.hidden_dim, bias=self.bias)
            n_node_feats += 3
            if self.model_params.get('cnc_presample_greedy_feature', False):
                self.greedy_hint_projection = nn.Linear(1, self.hidden_dim, bias=self.bias)
                nn.init.orthogonal_(self.greedy_hint_projection.weight, gain=1.0)
                if self.bias:
                    nn.init.constant_(self.greedy_hint_projection.bias, 0.0)
                n_node_feats += 1

        else:
            raise ValueError(f"Invalid nc_train_mode: {self.model_params['nc_train_mode']}")

        # In node MLP
        in_linear1 = nn.Linear(n_node_feats * self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear1.weight, gain=1.0)
        in_linear2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear2.weight, gain=1.0)
        if self.bias:
            nn.init.constant_(in_linear1.bias, 0.0)
            nn.init.constant_(in_linear2.bias, 0.0)

        self.in_mlp = nn.Sequential(
            in_linear1,
            Activation(self.model_params['activation']),
            nn.Dropout(self.model_params['dropout']),
            in_linear2,
        )

        # GNN layers
        self.encoder_layers = nn.ModuleList([GTLayer(**self.model_params) for _ in range(self.model_params['n_layers'])])

        self.edge_embeddings = nn.ModuleList([nn.Embedding(2, 2 * self.n_heads, device=self.model_params['device'],
                                                           ) for _ in range(self.model_params['n_layers'])])

        # Virtual node
        self.virtual_node = nn.Parameter(torch.randn(1, self.hidden_dim, device=self.model_params['device']))

        # Decoder
        out_dim = 2 if self.model_params['problem'] == 'mc' else 1  # 1 for MIS
        self.decoder = LinearDecoder(out_dim, **self.model_params)

    def forward(self, state):
        """
        Forward pass of the model.
        """
        batch_size, problem_size, _ = state.graph.size()

        # Nodes
        if self.model_params['nc_train_mode'] == 'exploitation':
            # Node features -> learned embeddings
            h_not0 = self.h.repeat(batch_size, problem_size - 1, 1)
            h_0 = self.h_0.repeat(batch_size, 1, 1)
            h = torch.cat([h_0, h_not0], dim=1)

        elif self.model_params['nc_train_mode'] in ['exploration', 'exploration_exploitation']:
            # Node features 1 -> learned embeddings
            h_not0 = self.h.repeat(batch_size, problem_size - 1, 1)
            h_0 = self.h_0.repeat(batch_size, 1, 1)
            h = torch.cat([h_0, h_not0], dim=1)

            # Node features 2 -> visited solutions
            h_visited = self.node_projection(state.visited_solutions.float())
            h = torch.cat([h, h_visited], dim=2)

        elif self.model_params['nc_train_mode'] == 'conditioned_network':
            # Node features 1 -> learned embeddings
            h_not0 = self.h.repeat(batch_size, problem_size - 1, 1)
            h_0 = self.h_0.repeat(batch_size, 1, 1)
            h = torch.cat([h_0, h_not0], dim=1)

            # Node features 2 -> visited solutions
            h_visited = self.node_projection(state.visited_solutions.float())

            # Node features 3 -> exploration weight
            tensor_exploration_w = state.exploration_weight.unsqueeze(1)
            h_exploration = self.exploration_projection(tensor_exploration_w).unsqueeze(1).repeat(1, problem_size, 1)
            pieces = [h, h_visited, h_exploration]
            if self.model_params.get('cnc_presample_greedy_feature', False):
                greedy_hint = getattr(state, 'greedy_inference_hint', None)
                if greedy_hint is None:
                    greedy_hint = torch.zeros(batch_size, dtype=torch.float32, device=self.model_params['device'])
                h_greedy_hint = self.greedy_hint_projection(greedy_hint.unsqueeze(1)).unsqueeze(1).repeat(1, problem_size, 1)
                pieces.append(h_greedy_hint)
            h = torch.cat(pieces, dim=2)

        else:
            raise ValueError(f"Invalid nc_train_mode: {self.model_params['nc_train_mode']}")

        # LAPLACE positional encodings
        if self.laplace_dim > 0:
            pos_h = self.laplace_linear(state.laplace_feats)
            h = torch.cat((h, pos_h), dim=-1)

        # In MLP
        h = self.in_mlp(h)

        # Edges
        adj = (state.graph.clone().reshape(batch_size, problem_size, problem_size))

        # GNN layers
        for idx, layer in enumerate(self.encoder_layers):
            e1, e2 = self.edge_embeddings[idx](adj).split(self.n_heads, dim=-1)
            e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)

            h = layer(h, e1, e2)

        # Decoder
        out = self.decoder(h, None)

        return out


class MISNCModel(nn.Module):
    def __init__(self, **params):
        super(MISNCModel, self).__init__()
        self.model_params = params
        self.tanh_clipping = self.model_params['tanh_clipping']
        self.hidden_dim: int = self.model_params['hidden_dim']
        self.n_heads: int = self.model_params['n_heads']
        self.bias: bool = self.model_params['bias']

        n_node_feats = 0

        # Input Node embeddings: learnable parameters
        if self.model_params['nc_train_mode'] == 'exploitation':
            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            n_node_feats += 1

        elif self.model_params['nc_train_mode'] in ['exploration', 'exploration_exploitation']:  # (using previously visited solutions)
            node_projection = nn.Linear(self.model_params['n_visited_solutions'], self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(node_projection.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(node_projection.bias, 0.0)
            self.node_projection = node_projection

            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))
            n_node_feats += 2

        elif self.model_params['nc_train_mode'] == 'conditioned_network':  # (using previously visited solutions)
            node_projection = nn.Linear(self.model_params['n_visited_solutions'], self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(node_projection.weight, gain=1.0)
            if self.bias:
                nn.init.constant_(node_projection.bias, 0.0)
            self.node_projection = node_projection

            self.h = nn.Parameter(torch.randn(1, 1, self.hidden_dim, device=self.model_params['device']))

            self.exploration_projection = nn.Linear(1, self.hidden_dim, bias=self.bias)
            n_node_feats += 3
            if self.model_params.get('cnc_presample_greedy_feature', False):
                self.greedy_hint_projection = nn.Linear(1, self.hidden_dim, bias=self.bias)
                nn.init.orthogonal_(self.greedy_hint_projection.weight, gain=1.0)
                if self.bias:
                    nn.init.constant_(self.greedy_hint_projection.bias, 0.0)
                n_node_feats += 1

        else:
            raise ValueError(f"Invalid nc_train_mode: {self.model_params['nc_train_mode']}")

        # after setting n_node_feats for base + visited + (maybe exploration)
        use_extra = self.model_params.get('use_extra_feats', True)
        extra_dim = self.model_params.get('extra_feats_dim', 2)
        use_lap = self.model_params.get('laplace_dim', 0) > 0

        if use_extra:
            self.extra_proj = nn.Linear(extra_dim, self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(self.extra_proj.weight, gain=1.0)
            if self.bias: nn.init.constant_(self.extra_proj.bias, 0.0)
            n_node_feats += 1

        if use_lap:
            self.lap_proj = nn.Linear(self.model_params['laplace_dim'], self.hidden_dim, bias=self.bias)
            nn.init.orthogonal_(self.lap_proj.weight, gain=1.0)
            if self.bias: nn.init.constant_(self.lap_proj.bias, 0.0)
            n_node_feats += 1

        # In node MLP
        in_linear1 = nn.Linear(n_node_feats * self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear1.weight, gain=1.0)
        in_linear2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=self.bias)
        nn.init.orthogonal_(in_linear2.weight, gain=1.0)
        if self.bias:
            nn.init.constant_(in_linear1.bias, 0.0)
            nn.init.constant_(in_linear2.bias, 0.0)

        self.in_mlp = nn.Sequential(
            in_linear1,
            Activation(self.model_params['activation']),
            nn.Dropout(self.model_params['dropout']),
            in_linear2,
        )

        # GNN layers
        self.encoder_layers = nn.ModuleList(
            [GTLayer(**self.model_params) for _ in range(self.model_params['n_layers'])])

        self.edge_embeddings = nn.ModuleList([nn.Embedding(2, 2 * self.n_heads, device=self.model_params['device'],
                                                           ) for _ in range(self.model_params['n_layers'])])

        # Virtual node
        self.virtual_node = nn.Parameter(torch.randn(1, self.hidden_dim, device=self.model_params['device']))

        # Decoder
        out_dim = 2  # 1 for MIS
        self.decoder = LinearDecoder(out_dim, **self.model_params)

    def forward(self, state):
        """
        Forward pass of the model.
        """
        batch_size, problem_size, _ = state.graph.size()

        # Nodes
        if self.model_params['nc_train_mode'] == 'exploitation':
            # Node features -> learned embeddings
            h = self.h.repeat(batch_size, problem_size, 1)
            # h = torch.ones((batch_size * pop_size, problem_size, self.hidden_dim))

        elif self.model_params['nc_train_mode'] in ['exploration', 'exploration_exploitation']:
            # Node features 1 -> learned embeddings
            h = self.h.repeat(batch_size, problem_size, 1)

            # Node features 2 -> visited solutions
            h_visited = self.node_projection(state.visited_solutions)
            h = torch.cat([h, h_visited], dim=2)

        elif self.model_params['nc_train_mode'] == 'conditioned_network':
            # Node features 1 -> learned embeddings
            h = self.h.repeat(batch_size, problem_size, 1)

            # Node features 2 -> visited solutions
            h_visited = self.node_projection(state.visited_solutions)

            # Node features 3 -> exploration weight
            tensor_exploration_w = state.exploration_weight.unsqueeze(1)
            h_exploration = self.exploration_projection(tensor_exploration_w).unsqueeze(1).repeat(1, problem_size, 1)
            pieces = [h, h_visited, h_exploration]
            if self.model_params.get('cnc_presample_greedy_feature', False):
                greedy_hint = getattr(state, 'greedy_inference_hint', None)
                if greedy_hint is None:
                    greedy_hint = torch.zeros(batch_size, dtype=torch.float32, device=self.model_params['device'])
                h_greedy_hint = self.greedy_hint_projection(greedy_hint.unsqueeze(1)).unsqueeze(1).repeat(1, problem_size, 1)
                pieces.append(h_greedy_hint)
            h = torch.cat(pieces, dim=2)

        else:
            raise ValueError(f"Invalid nc_train_mode: {self.model_params['nc_train_mode']}")

        pieces = [h]
        # 3) static extras (deg, 2hop, vis stats)
        if getattr(state, "extra_node_feats", None) is not None and self.model_params.get('use_extra_feats', True):
            pieces.append(self.extra_proj(state.extra_node_feats))  # (B,N,H)

        # 4) Laplacian PEs (optional)
        if getattr(state, "laplace_feats", None) is not None and self.model_params.get('laplace_dim', 0) > 0:
            pieces.append(self.lap_proj(state.laplace_feats))  # (B,N,H)

        # 5) fuse to H
        h = torch.cat(pieces, dim=2)  # (B,N, F*H)

        # In MLP
        h = self.in_mlp(h)

        # Edges
        adj = (state.graph.clone().reshape(batch_size, problem_size, problem_size))

        # GNN layers
        for idx, layer in enumerate(self.encoder_layers):
            e1, e2 = self.edge_embeddings[idx](adj).split(self.n_heads, dim=-1)
            e1 = e1.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)
            e2 = e2.transpose(2, 3).transpose(1, 2)  # (B, nh, T, T)

            h = layer(h, e1, e2)

        # Decoder
        out = self.decoder(h, None)

        return out


class LinearDecoder(nn.Module):
    def __init__(self, out_dim=1, **params):
        super(LinearDecoder, self).__init__()
        self.model_params = params
        self.hidden_dim: int = self.model_params['hidden_dim']
        self.bias: bool = self.model_params['bias']

        self.linear_proj = nn.Linear(self.hidden_dim, out_dim, bias=self.bias)
        nn.init.orthogonal_(self.linear_proj.weight, gain=1.0)
        if self.bias:
            nn.init.constant_(self.linear_proj.bias, 0.0)

    def forward(self, h, h_graph=None):
        return self.linear_proj(h)
