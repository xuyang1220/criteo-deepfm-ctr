import os
import math
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def stable_hash(text: str, mod: int) -> int:
    # stable across runs / machines
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


@dataclass
class CriteoBatch:
    y: torch.Tensor         # (B,)
    dense: torch.Tensor     # (B, 13)
    sparse: torch.Tensor    # (B, 26)  int64 indices into [0, hash_bucket_size)


class CriteoIterable(Dataset):
    """
    Minimal "map-style" dataset reading the entire file offsets into memory
    (works fine for Kaggle train.txt; for 1TB dataset you'd do true streaming).
    """
    def __init__(
        self,
        path: str,
        indices: np.ndarray,
        hash_bucket_size: int,
        hash_bucket_size_per_field: int,
        num_dense: int = 13,
        num_sparse: int = 26,
        cat_missing_token: str = "__MISSING__",
    ):
        self.path = path
        self.indices = indices
        self.hash_bucket_size = hash_bucket_size
        self.hash_bucket_size_per_field = hash_bucket_size_per_field
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.cat_missing_token = cat_missing_token

        # Pre-read all lines into memory (Kaggle size is manageable).
        # If you don't want this, we can switch to mmap or an offset index.
        with open(path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def __len__(self) -> int:
        return len(self.indices)

    def _parse_line(self, line: str) -> Tuple[int, np.ndarray, np.ndarray]:
        parts = line.rstrip("\n").split("\t")
        # label + 13 ints + 26 cats = 40 columns
        y = int(parts[0])

        # dense
        dense = np.zeros((self.num_dense,), dtype=np.float32)
        for i in range(self.num_dense):
            v = parts[1 + i]
            if v == "" or v is None:
                dense[i] = 0.0
            else:
                # standard trick: log(1 + x)
                x = float(v)
                dense[i] = math.log(1.0 + x) if x > 0 else 0.0

        # sparse
        sparse = np.zeros((self.num_sparse,), dtype=np.int64)
        base = 1 + self.num_dense
        for j in range(self.num_sparse):
            raw = parts[base + j]
            if raw == "" or raw is None:
                raw = self.cat_missing_token
            # Field-aware hashing: include field id to reduce collisions
            # key = f"C{j+1}={raw}"
            # sparse[j] = stable_hash(key, self.hash_bucket_size)
            bucket = self.hash_bucket_size_per_field
            sparse[j] = stable_hash(raw, bucket)

        return y, dense, sparse

    def __getitem__(self, idx: int):
        line = self.lines[int(self.indices[idx])]
        y, dense, sparse = self._parse_line(line)
        return (
            torch.tensor(y, dtype=torch.float32),
            torch.from_numpy(dense),
            torch.from_numpy(sparse),
        )


def criteo_collate(batch):
    ys, ds, ss = zip(*batch)
    return CriteoBatch(
        y=torch.stack(ys, dim=0),
        dense=torch.stack(ds, dim=0),
        sparse=torch.stack(ss, dim=0),
    )
