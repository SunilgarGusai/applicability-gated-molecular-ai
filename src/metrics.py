from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score,
    brier_score_loss, log_loss,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def regression_metrics(y: np.ndarray, pred: np.ndarray, std: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "Spearman": float(spearmanr(y, pred).statistic),
    }
    if std is not None:
        std = np.clip(std, 1e-4, None)
        out["NLL"] = float(np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y - pred) / std) ** 2))
        lo, hi = pred - 1.96 * std, pred + 1.96 * std
        out["Coverage95"] = float(np.mean((y >= lo) & (y <= hi)))
        out["Width95"] = float(np.mean(hi - lo))
        out["UncertaintyErrorSpearman"] = float(spearmanr(std, np.abs(y - pred)).statistic)
    return out


def classification_metrics(y: np.ndarray, p: np.ndarray, std: np.ndarray | None = None) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    label = (p >= 0.5).astype(int)
    out = {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "MCC": float(matthews_corrcoef(y, label)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, label)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": expected_calibration_error(y, p),
        "LogLoss": float(log_loss(y, p)),
    }
    if std is not None:
        out["UncertaintyErrorSpearman"] = float(spearmanr(std, np.abs(y - p)).statistic)
    return out


def selective_curve(y: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, task: str, points: int = 20):
    order = np.argsort(uncertainty)
    coverages = np.linspace(0.2, 1.0, points)
    risks = []
    for c in coverages:
        n = max(2, int(round(c * len(y))))
        take = order[:n]
        if task == "regression":
            risk = np.sqrt(np.mean((y[take] - pred[take]) ** 2))
        else:
            risk = np.mean(((pred[take] >= 0.5).astype(int) != y[take]))
        risks.append(float(risk))
    return coverages, np.asarray(risks)
