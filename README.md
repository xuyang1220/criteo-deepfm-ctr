# Criteo DeepFM (PyTorch)

## Setup
pip install -r requirements.txt

## Train
Edit configs/base.yaml -> data.train_path  
python -m src.train --config configs/base.yaml

### Baseline performance
## Results on Full Criteo Dataset (45M rows)

### Experimental setup
- Dataset: **Kaggle Criteo Display Ads Challenge** (~45M samples)
- Model: **DeepFM (PyTorch)**
  - Global hash-based embeddings
  - `embed_dim = 16`
  - FM (1st + 2nd order) + Deep MLP
- Training:
  - Optimizer: AdamW
  - Loss: Binary Cross-Entropy (logloss)
  - Hardware: CPU
  - Epochs: 3
- Evaluation:
  - Validation split held out from training data
  - Metrics: **AUC**, **LogLoss**

### Validation metrics by epoch

| Epoch | Validation AUC | Validation LogLoss |
|------:|---------------:|-------------------:|
| 1 | **0.8043** | **0.4468** |
| 2 | 0.8033 | 0.4495 |
| 3 | 0.7987 | 0.4563 |

### Interpretation

- The model reaches its **best performance after the first epoch**, both in terms of AUC and logloss.
- Training beyond one epoch **degrades validation logloss and AUC**, indicating overfitting to noise and long-tail hashed features.
- This behavior is **expected on large-scale CTR datasets**, where:
  - One epoch already exposes the model to tens of millions of samples.
  - Dominant linear and pairwise interaction signals converge quickly.
  - Additional passes tend to overfit rare feature combinations and hurt probability calibration.

### Early stopping decision

Given that:
- Validation **logloss increases monotonically after epoch 1**
- Logloss is the primary business metric for CTR calibration

We apply **early stopping at epoch 1**, and treat the epoch-1 checkpoint as the final model for this configuration.

This is consistent with common industry practice for large-scale CTR training, where models are often trained for **≤1 full pass over the data**.

### Baseline assessment

The achieved performance:
- **AUC ≈ 0.804**
- **LogLoss ≈ 0.447**

is in line with expected baseline results for DeepFM on the full Criteo dataset using hash-based embeddings and minimal feature engineering.

This run serves as a **reference baseline** for subsequent experiments (e.g. embedding dimension tuning, field-wise embeddings, regularization).


