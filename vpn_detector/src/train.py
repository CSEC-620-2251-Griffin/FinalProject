import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score

from . import data_io, features
from .checks import run_checks
from .models import train_logreg, train_random_forest, train_xgboost
from .splits import group_stratified_split, save_splits


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)


def optimize_threshold(y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf
    thresholds = np.linspace(0.05, 0.95, 50)
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thr
    return best_threshold, best_score


def _train_val_test_split(df: pd.DataFrame, config: Dict):
    X_raw, y, groups = data_io.prepare_features(df, config)
    split_idx = group_stratified_split(
        X_raw, y, groups, test_size=config["split"]["test_size"], val_size=config["split"]["val_size"], random_state=config["split"]["random_state"]
    )
    save_splits(split_idx, Path("artifacts/datasets/splits.json"))
    print("[train] Saved splits to artifacts/datasets/splits.json")

    X_train = X_raw.iloc[split_idx["train"]]
    y_train = y.iloc[split_idx["train"]]
    g_train = groups.iloc[split_idx["train"]]

    X_val = X_raw.iloc[split_idx["val"]]
    y_val = y.iloc[split_idx["val"]]
    g_val = groups.iloc[split_idx["val"]]

    X_test = X_raw.iloc[split_idx["test"]]
    y_test = y.iloc[split_idx["test"]]
    g_test = groups.iloc[split_idx["test"]]
    return (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test)


def _prepare_encodings(train_df, val_df, test_df, config):
    encoder = features.FeatureEncoder()
    X_train_enc = encoder.fit_transform(train_df)
    X_val_enc = encoder.transform(val_df)
    X_test_enc = encoder.transform(test_df)

    imputer = data_io.fit_imputer(X_train_enc, strategy=config["data"]["numeric_impute"])
    X_train_imp = data_io.apply_imputer(X_train_enc, imputer, cast_float32=config["data"].get("cast_float32", True))
    X_val_imp = data_io.apply_imputer(X_val_enc, imputer, cast_float32=config["data"].get("cast_float32", True))
    X_test_imp = data_io.apply_imputer(X_test_enc, imputer, cast_float32=config["data"].get("cast_float32", True))
    joblib.dump({"encoder": encoder, "imputer": imputer, "feature_names": X_train_imp.columns.tolist()}, "artifacts/models/preprocess.joblib")
    print("[train] Saved preprocessing artifacts to artifacts/models/preprocess.joblib")
    return X_train_imp, X_val_imp, X_test_imp, encoder, imputer


def train_models(config_path: str) -> Dict:
    ensure_dirs()
    config = load_config(config_path)
    df = data_io.load_dataset(config)
    check_info = run_checks(df, config)

    (X_train_raw, y_train, g_train), (X_val_raw, y_val, g_val), (X_test_raw, y_test, g_test) = _train_val_test_split(
        df, config
    )
    X_train, X_val, X_test, encoder, imputer = _prepare_encodings(X_train_raw, X_val_raw, X_test_raw, config)

    pos = int((y_train == config["imbalance"]["positive_label"]).sum())
    neg = int((y_train != config["imbalance"]["positive_label"]).sum())
    class_ratio = neg / max(pos, 1)
    scoring = config["training"]["scoring"]
    if class_ratio > 4:
        scoring = "average_precision"
        print("[train] High class imbalance detected; switching scoring to average_precision")

    results = {"scoring": scoring, "checks": check_info}
    thresholds = {}

    if config["models"].get("run_rf", True):
        rf_model, rf_meta = train_random_forest(
            X_train,
            y_train,
            g_train,
            config["rf_params"],
            scoring=scoring,
            random_state=config["split"]["random_state"],
            n_jobs=config["training"]["n_jobs"],
            use_class_weight=config["imbalance"]["use_class_weight"],
            n_splits_cv=config["split"]["n_splits_cv"],
        )
        val_scores = rf_model.predict_proba(X_val)[:, 1]
        thr, _ = optimize_threshold(y_val.values, val_scores, metric=config["threshold"]["optimize_for"])
        thresholds["rf"] = thr
        print(f"[train] RF best params: {rf_meta['best_params']}")
        print(f"[train] RF threshold (val-optimized): {thr:.3f}")

        rf_final = rf_model.__class__(
            **rf_meta["best_params"],
            random_state=config["split"]["random_state"],
            n_jobs=config["training"]["n_jobs"],
            class_weight="balanced" if config["imbalance"]["use_class_weight"] else None,
        )
        rf_final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        joblib.dump(rf_final, "artifacts/models/best_rf.joblib")
        print("[train] Saved RandomForest model to artifacts/models/best_rf.joblib")
        results["rf_params"] = rf_meta["best_params"]

    scale_pos_weight = neg / max(pos, 1)
    if config["models"].get("run_xgb", True):
        # Holdout 10% of training data (by group) for early stopping during search fit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config["split"]["random_state"])
        train_idx_es, eval_idx_es = next(gss.split(X_train, y_train, g_train))
        eval_set = [(X_train.iloc[eval_idx_es], y_train.iloc[eval_idx_es])]

        xgb_model, xgb_meta = train_xgboost(
            X_train.iloc[train_idx_es],
            y_train.iloc[train_idx_es],
            g_train.iloc[train_idx_es],
            config["xgb_params"],
            scoring=scoring,
            random_state=config["split"]["random_state"],
            n_jobs=config["training"]["n_jobs"],
            n_splits_cv=config["split"]["n_splits_cv"],
            scale_pos_weight=scale_pos_weight,
            eval_set=eval_set,
            early_stopping_rounds=config["training"]["early_stopping_rounds"],
        )
        val_scores = xgb_model.predict_proba(X_val)[:, 1]
        thr, _ = optimize_threshold(y_val.values, val_scores, metric=config["threshold"]["optimize_for"])
        thresholds["xgb"] = thr
        print(f"[train] XGB best params: {xgb_meta['best_params']}")
        print(f"[train] XGB threshold (val-optimized): {thr:.3f}")

        X_comb = pd.concat([X_train, X_val])
        y_comb = pd.concat([y_train, y_val])
        X_train_final, X_es, y_train_final, y_es = train_test_split(
            X_comb, y_comb, test_size=0.1, random_state=config["split"]["random_state"], stratify=y_comb
        )
        xgb_final = xgb_model.__class__(
            **xgb_meta["best_params"],
            objective="binary:logistic",
            tree_method="hist",
            random_state=config["split"]["random_state"],
            n_jobs=config["training"]["n_jobs"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        xgb_final.fit(
            X_train_final,
            y_train_final,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        joblib.dump(xgb_final, "artifacts/models/best_xgb.joblib")
        print("[train] Saved XGBoost model to artifacts/models/best_xgb.joblib")
        results["xgb_params"] = xgb_meta["best_params"]

    if config["models"].get("run_logreg", False):
        logreg_model, logreg_meta = train_logreg(
            X_train,
            y_train,
            g_train,
            config["logreg_params"],
            scoring=scoring,
            random_state=config["split"]["random_state"],
            n_jobs=config["training"]["n_jobs"],
            use_class_weight=config["imbalance"]["use_class_weight"],
            n_splits_cv=config["split"]["n_splits_cv"],
        )
        val_scores = logreg_model.predict_proba(X_val)[:, 1]
        thr, _ = optimize_threshold(y_val.values, val_scores, metric=config["threshold"]["optimize_for"])
        thresholds["logreg"] = thr
        print(f"[train] LogReg threshold (val-optimized): {thr:.3f}")
        # Fit on train+val for final model
        logreg_final = logreg_model
        logreg_final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        joblib.dump(logreg_final, "artifacts/models/best_logreg.joblib")
        print("[train] Saved Logistic Regression model to artifacts/models/best_logreg.joblib")
        results["logreg_params"] = logreg_meta["best_params"]

    if thresholds:
        Path("artifacts/models").mkdir(parents=True, exist_ok=True)
        Path("artifacts/models/thresholds.json").write_text(json.dumps(thresholds, indent=2))
        print(f"[train] Saved thresholds: {thresholds}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VPN detector models.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    train_models(args.config)
