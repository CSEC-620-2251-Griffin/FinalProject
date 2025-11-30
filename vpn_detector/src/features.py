from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureEncoder:
    def __init__(self, top_k_ports: int = 20):
        self.top_k_ports = top_k_ports
        self.top_src_ports: Optional[List[int]] = None
        self.top_dst_ports: Optional[List[int]] = None
        self.cat_maps: Dict[str, Dict[str, int]] = {}
        self.feature_columns_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        if "src_port" in df.columns:
            self.top_src_ports = (
                df["src_port"]
                .value_counts()
                .head(self.top_k_ports)
                .index.astype(int)
                .tolist()
            )
        if "dst_port" in df.columns:
            self.top_dst_ports = (
                df["dst_port"]
                .value_counts()
                .head(self.top_k_ports)
                .index.astype(int)
                .tolist()
            )
        for col in df.columns:
            if df[col].dtype == "object":
                uniques = df[col].dropna().unique().tolist()
                self.cat_maps[col] = {val: idx for idx, val in enumerate(uniques)}
        self.feature_columns_ = self._build_feature_columns(df)
        return self

    def _build_feature_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for col in df.columns:
            if df[col].dtype == "object":
                cols.append(f"{col}_enc")
            else:
                cols.append(col)
        if "src_port" in df.columns:
            cols.append("src_port_bucket")
        if "dst_port" in df.columns:
            cols.append("dst_port_bucket")
        return cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if "proto" in df_out.columns:
            df_out["proto"] = pd.to_numeric(df_out["proto"], errors="coerce").fillna(-1).astype(int)
        if self.top_src_ports is not None and "src_port" in df_out.columns:
            df_out["src_port_bucket"] = df_out["src_port"].apply(
                lambda x: x if x in self.top_src_ports else -1
            )
        if self.top_dst_ports is not None and "dst_port" in df_out.columns:
            df_out["dst_port_bucket"] = df_out["dst_port"].apply(
                lambda x: x if x in self.top_dst_ports else -1
            )
        for col, mapping in self.cat_maps.items():
            df_out[f"{col}_enc"] = df_out[col].map(mapping).fillna(-1).astype(int)
            df_out = df_out.drop(columns=[col])

        for col in df_out.columns:
            if df_out[col].dtype == "object":
                df_out[col] = df_out[col].astype("category").cat.codes

        for col in self.feature_columns_:
            if col not in df_out.columns:
                df_out[col] = -1
        return df_out[self.feature_columns_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
