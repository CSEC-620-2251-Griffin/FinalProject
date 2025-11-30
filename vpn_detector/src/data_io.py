from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_dataset(config: Dict) -> pd.DataFrame:
    processed_path = Path(config["data"]["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")
    df = pd.read_parquet(processed_path)
    return df


def prepare_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    id_col = config["data"]["id_col"]
    label_col = config["data"]["label_col"]
    group_col = config["data"]["group_col"]
    drop_cols = config["data"].get("drop_cols", [])

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    for col in [id_col, label_col, group_col]:
        if col in df.columns and col not in cols_to_drop:
            cols_to_drop.append(col)
    features = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    y = df[label_col]
    groups = df[group_col]
    return features, y, groups


def fit_imputer(X_train: pd.DataFrame, strategy: str = "median") -> SimpleImputer:
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_train[num_cols])
    return imputer


def apply_imputer(X: pd.DataFrame, imputer: SimpleImputer, cast_float32: bool = True) -> pd.DataFrame:
    num_cols = X.select_dtypes(include=[np.number]).columns
    X_copy = X.copy()
    X_copy[num_cols] = imputer.transform(X_copy[num_cols])
    if cast_float32:
        for col in num_cols:
            X_copy[col] = X_copy[col].astype(np.float32)
    return X_copy
