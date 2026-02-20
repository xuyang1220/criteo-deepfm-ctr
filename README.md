# Criteo DeepFM (PyTorch)

## Setup
pip install -r requirements.txt

## Data
Download the [Criteo Display Ads dataset](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?select=dac) 
and place train.txt under:  
data/criteo/train.txt

## Train
Edit configs/base.yaml -> data.train_path  
python -m src.train --config configs/base.yaml

## Baseline results on Full Criteo Dataset (45M rows)

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

## Embedding Dimension Study (5M-row Subset)

To efficiently tune model capacity, we trained DeepFM on a 5M-row subset of the Criteo dataset and applied early stopping after one epoch (based on validation logloss).

### Experimental setup
- Dataset: 5M training rows + validation split
- Early stopping: epoch 1
- Other settings identical to full-data baseline

### Results

| Embedding Dim | Validation AUC | Validation LogLoss |
|--------------:|---------------:|-------------------:|
| 8  | 0.7913 | 0.4586 |
| 16 | **0.7929** | **0.4573** |
| 32 | 0.7930 | 0.4573 |

### Interpretation

- Increasing embedding dimension improves performance, but with **diminishing returns**.
- `embed_dim = 16` provides the best tradeoff between model capacity and generalization.
- Increasing to `embed_dim = 32` yields negligible gains in AUC and does not improve logloss.
- The performance trends on the 5M subset are consistent with full-dataset results, validating the use of the subset for hyperparameter tuning.

Based on these results, `embed_dim = 16` is selected as the default embedding size for subsequent experiments.

# Next

Field-wise embeddings instead of global hashing

Regularization:
Increase dropout (e.g. 0.3)
Add L2 on embeddings
Lower learning rate (e.g. 5e-4)

## Field-wise Hash Buckets Study (5M-row Subset)

To reduce harmful hash collisions across categorical fields, we replaced a single global hash table with **field-wise embedding tables** (one embedding table per categorical feature). We trained on a 5M-row subset and applied early stopping after one epoch based on validation logloss.

### Results (early stop @ epoch 1)

| Hash Buckets per Field | Validation AUC | Validation LogLoss |
|-----------------------:|---------------:|-------------------:|
| 2^15 (32768)  | 0.792782 | 0.457598 |
| 2^16 (65536)  | 0.792831 | 0.457367 |
| 2^17 (131072) | **0.793517** | **0.457130** |

### Interpretation

- Increasing the number of buckets per field reduces within-field collisions and improves model quality.
- Gains are more pronounced in **logloss** (probability calibration) than AUC, which is typical for CTR modeling.
- `2^17` buckets per field achieved the best performance in this sweep and is selected as the default for subsequent experiments.

### Draft below
Seed = 40 embed_dim=16 val_auc=0.794836 val_logloss=0.456404