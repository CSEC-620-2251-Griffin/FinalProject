from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def class_balance(y: pd.Series) -> Tuple[int, int, float]:
    counts = y.value_counts()
    pos = counts.min() if len(counts) > 1 else counts.iloc[0]
    neg = counts.max() if len(counts) > 1 else counts.iloc[0]
    ratio = neg / max(pos, 1)
    return int(pos), int(neg), float(ratio)


def run_checks(df: pd.DataFrame, config: Dict) -> Dict:
    label_col = config["data"]["label_col"]
    figures_dir = Path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[checks] Dataset shape: {df.shape}")
    missing = df.isnull().sum()
    print("[checks] Missing values (top 10):")
    print(missing.sort_values(ascending=False).head(10))

    pos, neg, ratio = class_balance(df[label_col])
    print(f"[checks] Class balance: pos={pos}, neg={neg}, ratio={ratio:.2f}:1")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stds = df[numeric_cols].std()
    low_variance = stds[stds < 1e-6].index.tolist()
    if low_variance:
        print(f"[checks] Warning: near-zero variance columns: {low_variance}")

    corr_warnings: List[str] = []
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [
            (i, j, upper.loc[i, j])
            for i in upper.columns
            for j in upper.columns
            if (i != j) and (upper.loc[i, j] > 0.95)
        ]
        if high_corr:
            corr_warnings = [f"{a}~{b}:{v:.2f}" for a, b, v in high_corr]
            print(f"[checks] Warning: highly correlated features (>0.95): {corr_warnings}")

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, cbar=True)
        plt.title("Feature Correlation Heatmap")
        heatmap_path = figures_dir / "heatmap.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.close()
        print(f"[checks] Correlation heatmap saved to {heatmap_path}")

    return {"class_ratio": ratio, "corr_warnings": corr_warnings, "low_variance": low_variance}
