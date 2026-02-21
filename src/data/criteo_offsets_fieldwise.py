import math
import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def stable_hash(text: str, mod: int) -> int:
    # stable across runs / machines
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


@dataclass
class CriteoBatch:
    y: torch.Tensor
    dense: torch.Tensor
    sparse: torch.Tensor

def build_line_offsets(path: str, max_lines: int | None = None) -> np.ndarray:
    offsets = []
    with open(path, "rb") as f:
        while True:
            offsets.append(f.tell())
            line = f.readline()
            if not line:
                offsets.pop()  # remove last EOF offset
                break
            if max_lines is not None and len(offsets) >= max_lines:
                break
    return np.asarray(offsets, dtype=np.int64)


class CriteoOffsetDatasetFieldWise(Dataset):
    """
    Field-wise hashing dataset:
      sparse[j] in [0, bucket_per_field)
    """
    def __init__(
        self,
        path: str,
        offsets: np.ndarray,
        bucket_per_field: int,
        num_dense: int = 13,
        num_sparse: int = 26,
        cat_missing_token: str = "__MISSING__",
    ):
        self.path = path
        self.offsets = offsets
        self.bucket_per_field = int(bucket_per_field)
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.cat_missing_token = cat_missing_token
        self._fh = None  # per-process file handle

    def _get_fh(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")
        return self._fh

    def __len__(self) -> int:
        return int(len(self.offsets))

    def _parse_line(self, line: str) -> Tuple[int, np.ndarray, np.ndarray]:
        parts = line.rstrip("\n").split("\t")
        y = int(parts[0])

        dense = np.zeros((self.num_dense,), dtype=np.float32)
        for i in range(self.num_dense):
            v = parts[1 + i]
            if v:
                x = float(v)
                dense[i] = math.log(1.0 + x) if x > 0 else 0.0

        sparse = np.zeros((self.num_sparse,), dtype=np.int64)
        base = 1 + self.num_dense
        for j in range(self.num_sparse):
            raw = parts[base + j] if base + j < len(parts) else ""
            if not raw:
                raw = self.cat_missing_token
            # field-wise hashing: hash within the field bucket space
            sparse[j] = stable_hash(raw, self.bucket_per_field)

        return y, dense, sparse

    def __getitem__(self, idx: int):
        fh = self._get_fh()
        fh.seek(int(self.offsets[idx]))
        line = fh.readline().decode("utf-8", errors="ignore")
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