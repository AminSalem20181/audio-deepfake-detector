from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def eer(y_true, scores) -> float:
    y, s = np.asarray(y_true), np.asarray(scores)
    fpr, tpr, _ = roc_curve(y, s, pos_label=1)
    fnr = 1 - tpr
    idx = np.where(np.diff(np.sign(fpr - fnr)) != 0)[0]
    if len(idx) == 0:
        return float((fpr[np.argmin(np.abs(fpr-fnr))] + fnr[np.argmin(np.abs(fpr-fnr))]) / 2)
    i = idx[0]; x0, x1 = fpr[i]-fnr[i], fpr[i+1]-fnr[i+1]
    w = -x0 / (x1-x0) if x1 != x0 else 0.0
    return float(fpr[i] + w*(fpr[i+1]-fpr[i]))

def auc(y_true, scores) -> float:
    return float(roc_auc_score(y_true, scores))
