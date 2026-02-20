from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, dims: List[int], dropout: float):
        super().__init__()
        layers = []
        cur = in_dim
        for d in dims:
            layers.append(nn.Linear(cur, d))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = d
        self.net = nn.Sequential(*layers)
        self.out_dim = cur

    def forward(self, x):
        return self.net(x)


class DeepFMFieldWise(nn.Module):
    """
    DeepFM with field-wise embeddings:
      - each categorical field j has its own embedding table
      - avoids cross-field hash collisions
    sparse input: (B, num_sparse), where each column is in [0, hash_bucket_size_per_field)
    """
    def __init__(
        self,
        num_dense: int,
        num_sparse: int,
        hash_bucket_size_per_field: int,
        embed_dim: int,
        mlp_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.embed_dim = embed_dim
        self.hash_bucket_size_per_field = hash_bucket_size_per_field

        # First-order part
        self.linear_dense = nn.Linear(num_dense, 1)
        self.linear_sparse = nn.ModuleList(
            [nn.Embedding(hash_bucket_size_per_field, 1) for _ in range(num_sparse)]
        )

        # Field-wise embeddings for FM2 + Deep
        self.embed = nn.ModuleList(
            [nn.Embedding(hash_bucket_size_per_field, embed_dim) for _ in range(num_sparse)]
        )

        deep_in_dim = num_dense + num_sparse * embed_dim
        self.mlp = MLP(deep_in_dim, mlp_dims, dropout)
        self.deep_out = nn.Linear(self.mlp.out_dim, 1)

        self.bias = nn.Parameter(torch.zeros(1))

        # Init
        for emb in self.embed:
            nn.init.xavier_uniform_(emb.weight.data)
        for emb1 in self.linear_sparse:
            nn.init.xavier_uniform_(emb1.weight.data)

    def fm_second_order(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, F, K)
        returns: (B, 1)
        """
        sum_v = emb.sum(dim=1)                 # (B, K)
        sum_v_sq = sum_v * sum_v               # (B, K)
        v_sq = emb * emb                       # (B, F, K)
        sum_v_sq_fields = v_sq.sum(dim=1)      # (B, K)
        out = 0.5 * (sum_v_sq - sum_v_sq_fields).sum(dim=1, keepdim=True)  # (B,1)
        return out

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """
        dense: (B, num_dense) float
        sparse: (B, num_sparse) int64, each col in [0, hash_bucket_size_per_field)
        returns logits: (B,)
        """
        # Linear term
        y_linear = self.linear_dense(dense) + self.bias  # (B,1)
        # Add per-field sparse linear weights
        # each: (B,1), summed over fields -> (B,1)
        y_linear = y_linear + torch.stack(
            [self.linear_sparse[j](sparse[:, j]) for j in range(self.num_sparse)],
            dim=1
        ).sum(dim=1)

        # Field-wise embeddings -> (B, F, K)
        emb = torch.stack(
            [self.embed[j](sparse[:, j]) for j in range(self.num_sparse)],
            dim=1
        )

        # FM second-order
        y_fm2 = self.fm_second_order(emb)  # (B,1)

        # Deep part
        deep_in = torch.cat([dense, emb.flatten(start_dim=1)], dim=1)
        y_deep = self.deep_out(self.mlp(deep_in))  # (B,1)

        logits = (y_linear + y_fm2 + y_deep).squeeze(1)
        return logits
