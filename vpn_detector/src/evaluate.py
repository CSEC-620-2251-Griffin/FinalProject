import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import data_io
from .plots import (
    plot_calibration,
    plot_confusion,
    plot_feature_importance,
    plot_per_capture_bar,
    plot_pr,
    plot_roc,
    plot_score_hist,
)
from .splits import load_splits


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)


def compute_recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, pos_label: int = 1) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, zero_division=0, pos_label=pos_label),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "recall_at_fpr_1pct": compute_recall_at_fpr(y_true, y_prob, max_fpr=0.01),
    }
    return metrics


def check_group_leakage(groups_train, groups_test) -> None:
    overlap = set(groups_train).intersection(set(groups_test))
    if overlap:
        print(f"[eval] Warning: capture_id leakage detected: {len(overlap)} overlaps")
    else:
        print("[eval] capture_id leakage check passed (train/test disjoint)")


def load_preprocess_artifacts():
    prep_path = Path("artifacts/models/preprocess.joblib")
    if not prep_path.exists():
        raise FileNotFoundError("Preprocessing artifacts not found. Run training first.")
    artifacts = joblib.load(prep_path)
    return artifacts["encoder"], artifacts["imputer"], artifacts["feature_names"]


def per_capture_report(
    model_name: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    captures: pd.Series,
) -> pd.DataFrame:
    rows = []
    for cap in sorted(captures.unique()):
        mask = captures == cap
        yt = y_true[mask]
        yp = y_pred[mask]
        prob_cap = y_prob[mask]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        pos = int(yt.sum())
        neg = int((yt == 0).sum())
        recall = tp / pos if pos > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / neg if neg > 0 else np.nan
        rows.append(
            {
                "capture_id": cap,
                "n": int(mask.sum()),
                "pos": pos,
                "neg": neg,
                "accuracy": float((yp == yt).mean()),
                "recall": float(recall),
                "precision": float(precision),
                "fpr": fpr,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "avg_score": float(prob_cap.mean()),
            }
        )
    df = pd.DataFrame(rows)
    csv_path = Path(f"artifacts/reports/per_capture_{model_name}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[eval] Per-capture metrics saved to {csv_path}")
    return df


def evaluate(config_path: str) -> Dict:
    ensure_dirs()
    config = load_config(config_path)
    df = data_io.load_dataset(config)
    splits_path = Path("artifacts/datasets/splits.json")
    if not splits_path.exists():
        raise FileNotFoundError("Splits not found. Run training to create consistent splits.")
    splits = load_splits(splits_path)

    encoder, imputer, feature_names = load_preprocess_artifacts()
    label_col = config["data"]["label_col"]
    group_col = config["data"]["group_col"]

    y_all = df[label_col]
    groups_all = df[group_col]
    drop_cols = config["data"].get("drop_cols", []) + [label_col, group_col, config["data"]["id_col"]]
    X_all = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train_raw = X_all.iloc[splits["train"]]
    X_val_raw = X_all.iloc[splits["val"]]
    X_test_raw = X_all.iloc[splits["test"]]
    y_train = y_all.iloc[splits["train"]]
    y_val = y_all.iloc[splits["val"]]
    y_test = y_all.iloc[splits["test"]]
    g_train = groups_all.iloc[splits["train"]]
    g_test = groups_all.iloc[splits["test"]]
    check_group_leakage(g_train, g_test)

    X_train = encoder.transform(X_train_raw)
    X_val = encoder.transform(X_val_raw)
    X_test = encoder.transform(X_test_raw)
    X_train = data_io.apply_imputer(X_train, imputer, cast_float32=config["data"].get("cast_float32", True))
    X_val = data_io.apply_imputer(X_val, imputer, cast_float32=config["data"].get("cast_float32", True))
    X_test = data_io.apply_imputer(X_test, imputer, cast_float32=config["data"].get("cast_float32", True))

    thresholds = {}
    thr_path = Path("artifacts/models/thresholds.json")
    if thr_path.exists():
        thresholds = json.loads(thr_path.read_text())

    report_lines = []
    metrics_all: Dict[str, Dict] = {}

    if Path("artifacts/models/best_rf.joblib").exists():
        rf = joblib.load("artifacts/models/best_rf.joblib")
        rf_prob = rf.predict_proba(X_test)[:, 1]
        thr = thresholds.get("rf", 0.5)
        rf_pred = (rf_prob >= thr).astype(int)
        rf_metrics = compute_metrics(y_test.values, rf_prob, thr, pos_label=config["imbalance"]["positive_label"])
        metrics_all["rf"] = rf_metrics
        print(f"[eval] RF metrics: {rf_metrics}")
        plot_roc(y_test, rf_prob, Path("artifacts/figures/roc_rf.png"), label="RandomForest")
        plot_pr(y_test, rf_prob, Path("artifacts/figures/pr_rf.png"), label="RandomForest")
        plot_calibration(y_test, rf_prob, Path("artifacts/figures/calibration_rf.png"))
        plot_confusion(y_test, rf_pred, Path("artifacts/figures/confusion_rf.png"))
        plot_score_hist(y_test.values, rf_prob, Path("artifacts/figures/scores_rf.png"))
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            plot_feature_importance(
                importances, feature_names, Path("artifacts/figures/importance_rf.png"), title="RF Feature Importance"
            )
            top_idx = np.argsort(importances)[::-1][:20]
            top_feats = [(feature_names[i], float(importances[i])) for i in top_idx]
            print("[eval] RF top 20 importances:", top_feats)
        rf_pc = per_capture_report("rf", y_test, rf_prob, rf_pred, g_test)
        vpn_caps = rf_pc[rf_pc["pos"] > 0]
        nonvpn_caps = rf_pc[rf_pc["pos"] == 0]
        if not vpn_caps.empty:
            plot_per_capture_bar(
                vpn_caps["capture_id"].astype(str).tolist(),
                vpn_caps["recall"].tolist(),
                vpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_recall_rf.png"),
                "RF Recall by VPN Capture",
                "Recall",
            )
        if not nonvpn_caps.empty:
            # For non-VPN captures, surface false positive rate
            fpr_vals = [v if not np.isnan(v) else 0.0 for v in nonvpn_caps["fpr"].tolist()]
            plot_per_capture_bar(
                nonvpn_caps["capture_id"].astype(str).tolist(),
                fpr_vals,
                nonvpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_fpr_rf.png"),
                "RF False Positive Rate by Non-VPN Capture",
                "FPR",
            )

    if Path("artifacts/models/best_xgb.joblib").exists():
        xgb = joblib.load("artifacts/models/best_xgb.joblib")
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        thr = thresholds.get("xgb", 0.5)
        xgb_pred = (xgb_prob >= thr).astype(int)
        xgb_metrics = compute_metrics(y_test.values, xgb_prob, thr, pos_label=config["imbalance"]["positive_label"])
        metrics_all["xgb"] = xgb_metrics
        print(f"[eval] XGB metrics: {xgb_metrics}")
        plot_roc(y_test, xgb_prob, Path("artifacts/figures/roc_xgb.png"), label="XGBoost")
        plot_pr(y_test, xgb_prob, Path("artifacts/figures/pr_xgb.png"), label="XGBoost")
        plot_calibration(y_test, xgb_prob, Path("artifacts/figures/calibration_xgb.png"))
        plot_confusion(y_test, xgb_pred, Path("artifacts/figures/confusion_xgb.png"))
        plot_score_hist(y_test.values, xgb_prob, Path("artifacts/figures/scores_xgb.png"))
        if hasattr(xgb, "feature_importances_"):
            importances = xgb.feature_importances_
            plot_feature_importance(
                importances, feature_names, Path("artifacts/figures/importance_xgb.png"), title="XGB Feature Importance"
            )
            top_idx = np.argsort(importances)[::-1][:20]
            top_feats = [(feature_names[i], float(importances[i])) for i in top_idx]
            print("[eval] XGB top 20 importances:", top_feats)
        xgb_pc = per_capture_report("xgb", y_test, xgb_prob, xgb_pred, g_test)
        vpn_caps = xgb_pc[xgb_pc["pos"] > 0]
        nonvpn_caps = xgb_pc[xgb_pc["pos"] == 0]
        if not vpn_caps.empty:
            plot_per_capture_bar(
                vpn_caps["capture_id"].astype(str).tolist(),
                vpn_caps["recall"].tolist(),
                vpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_recall_xgb.png"),
                "XGB Recall by VPN Capture",
                "Recall",
            )
        if not nonvpn_caps.empty:
            fpr_vals = [v if not np.isnan(v) else 0.0 for v in nonvpn_caps["fpr"].tolist()]
            plot_per_capture_bar(
                nonvpn_caps["capture_id"].astype(str).tolist(),
                fpr_vals,
                nonvpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_fpr_xgb.png"),
                "XGB False Positive Rate by Non-VPN Capture",
                "FPR",
            )

    if Path("artifacts/models/best_logreg.joblib").exists():
        logreg = joblib.load("artifacts/models/best_logreg.joblib")
        logreg_prob = logreg.predict_proba(X_test)[:, 1]
        thr = thresholds.get("logreg", 0.5)
        logreg_pred = (logreg_prob >= thr).astype(int)
        logreg_metrics = compute_metrics(y_test.values, logreg_prob, thr, pos_label=config["imbalance"]["positive_label"])
        metrics_all["logreg"] = logreg_metrics
        print(f"[eval] LogReg metrics: {logreg_metrics}")
        plot_roc(y_test, logreg_prob, Path("artifacts/figures/roc_logreg.png"), label="LogReg")
        plot_pr(y_test, logreg_prob, Path("artifacts/figures/pr_logreg.png"), label="LogReg")
        plot_calibration(y_test, logreg_prob, Path("artifacts/figures/calibration_logreg.png"))
        plot_confusion(y_test, logreg_pred, Path("artifacts/figures/confusion_logreg.png"))
        plot_score_hist(y_test.values, logreg_prob, Path("artifacts/figures/scores_logreg.png"))
        log_pc = per_capture_report("logreg", y_test, logreg_prob, logreg_pred, g_test)
        vpn_caps = log_pc[log_pc["pos"] > 0]
        nonvpn_caps = log_pc[log_pc["pos"] == 0]
        if not vpn_caps.empty:
            plot_per_capture_bar(
                vpn_caps["capture_id"].astype(str).tolist(),
                vpn_caps["recall"].tolist(),
                vpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_recall_logreg.png"),
                "LogReg Recall by VPN Capture",
                "Recall",
            )
        if not nonvpn_caps.empty:
            fpr_vals = [v if not np.isnan(v) else 0.0 for v in nonvpn_caps["fpr"].tolist()]
            plot_per_capture_bar(
                nonvpn_caps["capture_id"].astype(str).tolist(),
                fpr_vals,
                nonvpn_caps["n"].tolist(),
                Path("artifacts/figures/per_capture_fpr_logreg.png"),
                "LogReg False Positive Rate by Non-VPN Capture",
                "FPR",
            )

    report_path = Path("artifacts/reports/report.md")
    with report_path.open("w") as f:
        f.write("# VPN Detector Evaluation\n\n")
        if thresholds:
            f.write("## Thresholds\n")
            for model_name, thr in thresholds.items():
                f.write(f"- {model_name}: {thr}\n")
            f.write("\n")
        for model_name, metric_dict in metrics_all.items():
            f.write(f"## {model_name.upper()}\n")
            for k, v in metric_dict.items():
                f.write(f"- {k}: {v:.4f}\n")
            f.write("\n")
        f.write("### Classification report (val for threshold reference)\n")
        if Path("artifacts/models/best_rf.joblib").exists():
            rf = joblib.load("artifacts/models/best_rf.joblib")
            rf_val_pred = (rf.predict_proba(X_val)[:, 1] >= thresholds.get("rf", 0.5)).astype(int)
            f.write("**RandomForest**\n\n")
            f.write(classification_report(y_val, rf_val_pred))
            f.write("\n")
        if Path("artifacts/models/best_xgb.joblib").exists():
            xgb = joblib.load("artifacts/models/best_xgb.joblib")
            xgb_val_pred = (xgb.predict_proba(X_val)[:, 1] >= thresholds.get("xgb", 0.5)).astype(int)
            f.write("**XGBoost**\n\n")
            f.write(classification_report(y_val, xgb_val_pred))
            f.write("\n")
        if Path("artifacts/models/best_logreg.joblib").exists():
            logreg = joblib.load("artifacts/models/best_logreg.joblib")
            logreg_val_pred = (logreg.predict_proba(X_val)[:, 1] >= thresholds.get("logreg", 0.5)).astype(int)
            f.write("**LogReg**\n\n")
            f.write(classification_report(y_val, logreg_val_pred))
            f.write("\n")
    print(f"[eval] Report saved to {report_path}")

    env_out = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=False)
    Path("artifacts/reports/env.txt").write_text(env_out.stdout)
    print("[eval] Environment snapshot saved to artifacts/reports/env.txt")

    return metrics_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VPN detector models.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    evaluate(args.config)
