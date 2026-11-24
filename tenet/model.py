"""TeNeT: Temporal Neural Transformer for dynamic functional connectivity.

Kim et al., "Learning Dynamic Brain Connectome with Graph Transformers for
Psychiatric Diagnosis Classification", ISBI 2024.

For each sliding window t the connectome embedding X_t holds
  - structure: the windowed correlation matrix R_t without self-loops,
  - position:  the one-hot encoding of each ROI,
  - time:      a GRU encoding of the ROI time series up to the end of the window.
A graph Transformer g attends across ROIs within each window, and a Transformer h
attends across windows to produce the subject representation h_dyn.
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat


class TimeEncoder(nn.Module):
    """GRU over the ROI time series; returns the output at the last time point of each window."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout),
        )

    def forward(self, t, window_endpoints):
        # t: [T, B, N]
        return self.rnn(t[:max(window_endpoints)])[0][[p - 1 for p in window_endpoints]]


class GraphTransformerLayer(nn.Module):
    """Self-attention across ROIs; the FC and degree encoding is concatenated before the MLP."""

    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, structure):
        x_attend, attn = self.multihead_attn(x, x, x, average_attn_weights=False)
        x = self.layer_norm1(x + self.dropout1(x_attend))
        x = self.layer_norm2(x + self.dropout2(self.mlp(torch.cat([x, structure], dim=2))))
        return x, attn


class TransformerLayer(nn.Module):
    """Standard Transformer encoder layer used across time."""

    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_attend, attn = self.multihead_attn(x, x, x, average_attn_weights=True)
        x = self.layer_norm1(x + self.dropout1(x_attend))
        x = self.layer_norm2(x + self.dropout2(self.mlp(x)))
        return x, attn


class TeNeT(nn.Module):
    """Temporal Neural Transformer.

    Args:
        input_dim: number of ROIs (N)
        num_classes: number of classes (1 for regression)
        hidden_dim: hidden size D
        num_layers: number of graph Transformer layers (across ROIs)
        num_time_layers: number of Transformer layers across time
        num_heads: attention heads
        dropout: dropout before the final linear layer
        cls_token: 'param' (learnable token), 'mean', or 'sum' readout
    """

    def __init__(self, input_dim, num_classes, hidden_dim=1024, num_layers=4, num_time_layers=4,
                 num_heads=4, dropout=0.5, cls_token='param', layer_dropout=0.2):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.readout = lambda x: x.sum(0)
        elif cls_token == 'mean':
            self.readout = lambda x: x.mean(0)
        else:
            self.readout = lambda x: x[-1]

        self.cls_token = nn.Parameter(torch.randn(hidden_dim)) if cls_token == 'param' else None
        self.cls_token_time = nn.Parameter(torch.randn(hidden_dim)) if cls_token == 'param' else None
        self.num_classes = num_classes

        # graph embedding: [R_t - I | I] -> D
        self.graph_mlp = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.time_encoder = TimeEncoder(input_dim, hidden_dim)

        self.graph_layers = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_dim, num_classes)
        self.structure_linear = nn.Linear(input_dim, hidden_dim)

        for _ in range(num_layers):
            self.graph_layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout=layer_dropout))
        for _ in range(num_time_layers):
            self.time_layers.append(TransformerLayer(hidden_dim, hidden_dim, num_heads, dropout=layer_dropout))

    def embed_connectome(self, x, window_starts, window_size):
        """Connectome embeddings for every window.

        Args:
            x: [B, T, N] ROI time series
        Returns:
            graph: [W, N, B, D] graph embeddings, time: [W, 1, B, D] time embeddings,
            structure: [W, N, B, N] Fisher-z transformed FC without self-loops
        """
        graph, structure = [], []
        eye = torch.eye(x.shape[2], device=x.device)
        for s in window_starts:
            fc = torch.stack([corrcoef(rearrange(w, 't n -> n t')) - eye for w in x[:, s:s + window_size]])
            fc = torch.clip(torch.atanh(fc), min=-5.0, max=5.0)  # Fisher z, clipped for numerical stability
            position = repeat(eye, 'n c -> b n c', b=fc.shape[0])
            g = rearrange(torch.cat([fc, position], dim=2), 'b n c -> n b c')
            graph.append(self.graph_mlp(g))
            structure.append(rearrange(fc.detach(), 'b n c -> n b c'))
        graph = torch.stack(graph)
        structure = torch.stack(structure)
        time = self.time_encoder(rearrange(x, 'b t n -> t b n') + 1e-8, [s + window_size for s in window_starts])
        return graph, time.unsqueeze(1), structure

    def forward(self, x, window_starts, window_size):
        """
        Args:
            x: [B, T, N] ROI time series (z-scored per ROI)
            window_starts: start index of each sliding window
            window_size: window length
        Returns:
            logit [B, num_classes], latent [B, D], attention dict
        """
        attention = {'node-attention-source': [], 'node-attention-target': [], 'time-attention': []}
        graph, time, structure = self.embed_connectome(x, window_starts, window_size)

        # FC rows for the N ROI tokens and the degree vector for the time token
        structure = torch.cat([structure, structure.sum(1, keepdim=True)], dim=1)
        structure = self.structure_linear(structure)

        num_windows, num_batch = graph.shape[0], graph.shape[2]
        h = torch.cat([rearrange(graph, 'w n b c -> n (b w) c'), rearrange(time, 'w n b c -> n (b w) c')])
        se = rearrange(structure, 'w n b c -> n (b w) c')
        if self.cls_token is not None:
            token = repeat(self.cls_token, 'c -> 1 b c', b=h.shape[1])
            h, se = torch.cat([h, token]), torch.cat([se, token])

        # g: attention across ROIs within each window
        for layer in self.graph_layers:
            h, attn = layer(h, se)
            attention['node-attention-source'].append(attn[:, :, -1])
            attention['node-attention-target'].append(attn[:, -1, :])
        h = rearrange(self.readout(h), '(b w) c -> w b c', w=num_windows)

        # h: attention across windows
        if self.cls_token_time is not None:
            h = torch.cat([h, repeat(self.cls_token_time, 'c -> 1 b c', b=num_batch)])
        for layer in self.time_layers:
            h, attn = layer(h)
            attention['time-attention'].append(attn)
        attention = {k: torch.stack(v, dim=1).detach().cpu() for k, v in attention.items()}

        latent = self.readout(h)
        logit = self.final_linear(self.dropout(latent)).squeeze(1)
        return logit, latent, attention


def corrcoef(x, eps=1e-9):
    """Pearson correlation between rows of x ([N, T])."""
    xm = x - x.mean(1, keepdim=True)
    c = xm.mm(xm.t()) / (x.size(1) - 1)
    std = torch.diag(c).pow(0.5)
    c = c / (std.expand_as(c) + eps)
    c = c / (std.expand_as(c).t() + eps)
    return torch.clamp(c, -1.0, 1.0)
