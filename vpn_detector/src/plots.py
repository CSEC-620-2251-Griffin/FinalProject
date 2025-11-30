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
