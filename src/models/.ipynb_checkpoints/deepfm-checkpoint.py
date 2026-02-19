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


class DeepFM(nn.Module):
    """
    DeepFM with:
      - dense features fed into both FM (as 1st order via linear) and deep
      - sparse features via embeddings
      - FM 2nd-order uses embeddings only (typical for CTR setups)
    """
    def __init__(
        self,
        num_dense: int,
        num_sparse: int,
        hash_bucket_size: int,
        embed_dim: int,
        mlp_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.embed_dim = embed_dim

        # first-order (linear) part
        self.linear_dense = nn.Linear(num_dense, 1)
        self.linear_sparse = nn.Embedding(hash_bucket_size, 1)

        # embeddings for FM second-order + Deep
        self.embed = nn.Embedding(hash_bucket_size, embed_dim)

        deep_in_dim = num_dense + num_sparse * embed_dim
        self.mlp = MLP(deep_in_dim, mlp_dims, dropout)
        self.deep_out = nn.Linear(self.mlp.out_dim, 1)

        self.bias = nn.Parameter(torch.zeros(1))

        # init
        nn.init.xavier_uniform_(self.embed.weight.data)
        nn.init.xavier_uniform_(self.linear_sparse.weight.data)
        # linear_dense and deep_out use default init (fine)

    def fm_second_order(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, num_sparse, embed_dim)
        FM 2nd-order: 0.5 * ( (sum v)^2 - sum(v^2) ) over fields
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
        dense: (B, 13) float
        sparse: (B, 26) int64
        returns logits: (B,)
        """
        # linear part
        y_linear = self.linear_dense(dense) + self.linear_sparse(sparse).sum(dim=1) + self.bias  # (B,1)

        # embeddings
        emb = self.embed(sparse)  # (B, 26, K)

        # FM second-order
        y_fm2 = self.fm_second_order(emb)  # (B,1)

        # Deep part
        deep_in = torch.cat([dense, emb.flatten(start_dim=1)], dim=1)  # (B, 13 + 26*K)
        y_deep = self.deep_out(self.mlp(deep_in))  # (B,1)

        logits = (y_linear + y_fm2 + y_deep).squeeze(1)  # (B,)
        return logits
