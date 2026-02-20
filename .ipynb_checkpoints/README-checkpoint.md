# Criteo DeepFM (PyTorch)

## Setup
pip install -r requirements.txt

## Train
Edit configs/base.yaml -> data.train_path
python -m src.train --config configs/base.yaml

### Baseline performance
DeepFM (hash, embed_dim=16, full Criteo):
epoch 1
AUC ≈ 0.804
LogLoss ≈ 0.447

#### Explanation
- Epoch 1 already sees millions of examples
  - Model quickly learns:
  - per-category biases
  - dominant pairwise interactions

- Later epochs:
  - overfit rare hashed buckets
  - reinforce noise
  - worsen calibration (logloss increases faster than AUC drops)

- This is classic behavior for:
  - large-scale CTR
  - hashed embeddings
  - neural models without heavy regularization

