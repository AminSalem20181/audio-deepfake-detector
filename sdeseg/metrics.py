"""Binary metrics: positive label is spoof=1, higher scores mean more spoof.

EER linearly interpolates the ROC segment crossing FPR=1-TPR.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def _validate(y_true, scores):
    y, s = np.asarray(y_true), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError('Expected equal-length finite score and label vectors')
    if set(y.tolist()) != {0, 1}:
        raise ValueError('Metrics require both binary classes (0 and 1)')
    return y, s


def eer(y_true, scores) -> float:
    y, s = _validate(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    delta = fpr - (1-tpr)
    j = int(np.flatnonzero(delta >= 0)[0])
    if delta[j] == 0:
        return float(fpr[j])
    i = j-1
    weight = -delta[i]/(delta[j]-delta[i])
    return float(fpr[i] + weight*(fpr[j]-fpr[i]))


def auc(y_true, scores) -> float:
    return float(roc_auc_score(*_validate(y_true, scores)))
