import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.criteo import CriteoIterable, criteo_collate
from src.models.deepfm import DeepFMFieldWise
from src.utils.seed import seed_everything
from src.utils.metrics import compute_metrics

torch.set_num_threads(20)

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        y = batch.y.to(device)
        dense = batch.dense.to(device)
        sparse = batch.sparse.to(device)
        logits = model(dense, sparse)
        prob = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return compute_metrics(y_true, y_prob)


def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    seed_everything(cfg["train"]["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_path = cfg["data"]["train_path"]
    assert os.path.exists(train_path), f"File not found: {train_path}"

    # indices split
    with open(train_path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(f"number of lines in train.txt: {n}")

    idx = np.arange(n)
    np.random.shuffle(idx)
    max_rows = cfg["data"].get("max_rows", None)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0:
            idx = idx[: min(len(idx), max_rows)]
    
    valid_ratio = cfg["data"]["valid_ratio"]
    n_val = int(len(idx) * valid_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    # datasets
    ds_tr = CriteoIterable(
        train_path, tr_idx,
        hash_bucket_size=cfg["features"]["hash_bucket_size"],
        hash_bucket_size_per_field=cfg["features"]["hash_bucket_size_per_field"],
        num_dense=cfg["features"]["num_dense"],
        num_sparse=cfg["features"]["num_sparse"],
        cat_missing_token=cfg["features"]["cat_missing_token"],
    )
    ds_val = CriteoIterable(
        train_path, val_idx,
        hash_bucket_size=cfg["features"]["hash_bucket_size"],
        hash_bucket_size_per_field=cfg["features"]["hash_bucket_size_per_field"],
        num_dense=cfg["features"]["num_dense"],
        num_sparse=cfg["features"]["num_sparse"],
        cat_missing_token=cfg["features"]["cat_missing_token"],
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=criteo_collate,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=criteo_collate,
        drop_last=False,
    )

    model = DeepFMFieldWise(
        num_dense=cfg["features"]["num_dense"],
        num_sparse=cfg["features"]["num_sparse"],
        hash_bucket_size_per_field=cfg["features"]["hash_bucket_size_per_field"],
        embed_dim=cfg["features"]["embed_dim"],
        mlp_dims=cfg["model"]["mlp_dims"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    global_step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        pbar = tqdm(dl_tr, desc=f"epoch {epoch}", ncols=100)
        for batch in pbar:
            y = batch.y.to(device)
            dense = batch.dense.to(device)
            sparse = batch.sparse.to(device)

            logits = model(dense, sparse)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        metrics = evaluate(model, dl_val, device)
        print(f"[epoch {epoch}] embed_dim={cfg['features']['embed_dim']} val_auc={metrics['auc']:.6f} val_logloss={metrics['logloss']:.6f}")

    os.makedirs("artifacts", exist_ok=True)
    ckpt_path = "artifacts/deepfm.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    main(args.config)
