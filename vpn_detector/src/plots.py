from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_roc(y_true, y_score, path: Path, label: str = "") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=None, estimator_name=label).plot()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr(y_true, y_score, path: Path, label: str = "") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=label).plot()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_calibration(y_true, y_prob, path: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion(y_true, y_pred, path: Path, normalize: Optional[str] = "true") -> None:
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format=".2f")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_feature_importance(importances, feature_names: List[str], path: Path, title: str) -> None:
    indices = np.argsort(importances)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), top_importances[::-1])
    plt.yticks(range(len(indices)), top_features[::-1])
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_per_capture_bar(capture_ids: List[str], values: List[float], counts: List[int], path: Path, title: str, metric_label: str) -> None:
    # Sort by metric to highlight weakest captures
    order = np.argsort(values)
    ids_sorted = [capture_ids[i] for i in order]
    vals_sorted = [values[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    plt.figure(figsize=(10, max(4, len(ids_sorted) * 0.2)))
    bars = plt.barh(range(len(ids_sorted)), vals_sorted, color="#1f77b4")
    plt.yticks(range(len(ids_sorted)), ids_sorted)
    plt.xlabel(metric_label)
    plt.title(title)
    # annotate support on bars
    for idx, (bar, cnt) in enumerate(zip(bars, counts_sorted)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"n={cnt}", va="center", fontsize=8)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_score_hist(y_true, y_score, path: Path, bins: int = 30) -> None:
    """Overlay score distributions for negatives vs positives."""
    y_true = np.asarray(y_true)
    plt.figure(figsize=(8, 4))
    plt.hist(y_score[y_true == 0], bins=bins, alpha=0.6, label="Non-VPN", color="#1f77b4", density=True)
    plt.hist(y_score[y_true == 1], bins=bins, alpha=0.6, label="VPN", color="#d62728", density=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
