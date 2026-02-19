import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    return {"auc": float(auc), "logloss": float(ll)}
