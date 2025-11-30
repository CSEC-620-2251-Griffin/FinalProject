import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def _choose_n_splits(target_size: float, n_groups: int) -> int:
    desired = int(round(1 / max(target_size, 1e-6)))
    n_splits = min(max(3, desired), n_groups)
    return max(2, n_splits)


def _pick_balanced_fold(sgkf: StratifiedGroupKFold, X, y, groups, target_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate over SGKF splits and pick the one whose test fold best matches the global
    class distribution and target size. This reduces variance and accidental skew.
    """
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    global_rate = y_array.mean() if len(y_array) else 0.0
    best = None
    best_score = float("inf")

    for train_idx, test_idx in sgkf.split(X, y_array, groups_array):
        test_rate = y_array[test_idx].mean() if len(test_idx) else 0.0
        size_ratio = len(test_idx) / max(len(y_array), 1)
        # small penalty for size deviation; main penalty for class distribution drift
        score = abs(test_rate - global_rate) + 0.5 * abs(size_ratio - target_size)
        if score < best_score:
            best_score = score
            best = (train_idx, test_idx)
    return best


def group_stratified_split(
    X, y, groups, test_size: float, val_size: float, random_state: int = 42
) -> Dict[str, np.ndarray]:
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    n_splits_outer = _choose_n_splits(test_size, n_groups)
    sgkf_outer = StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state
    )
    train_val_idx, test_idx = _pick_balanced_fold(
        sgkf_outer, X, y, groups, target_size=test_size
    )

    remaining_groups = groups[train_val_idx]
    n_splits_inner = _choose_n_splits(val_size, len(np.unique(remaining_groups)))
    sgkf_inner = StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=random_state + 1
    )
    inner_train_idx_rel, val_idx_rel = _pick_balanced_fold(
        sgkf_inner,
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        remaining_groups,
        target_size=val_size / max(1 - test_size, 1e-6),
    )
    train_idx = np.array(train_val_idx)[inner_train_idx_rel]
    val_idx = np.array(train_val_idx)[val_idx_rel]

    return {"train": train_idx, "val": val_idx, "test": np.array(test_idx)}


def save_splits(indices: Dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.tolist() for k, v in indices.items()}
    path.write_text(json.dumps(serializable, indent=2))


def load_splits(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    return {k: np.array(v) for k, v in data.items()}


def cv_splitter(n_splits: int, random_state: int = 42) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
