from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .splits import cv_splitter


def _choose_n_iter(grid: Dict, max_iter: int = 1000) -> int:
    # Prefer exhaustive search unless the grid is extremely large; user prefers maximum coverage.
    grid_list = list(ParameterGrid(grid))
    return min(len(grid_list), max_iter)


def train_random_forest(
    X,
    y,
    groups,
    params: Dict,
    scoring: str,
    random_state: int,
    n_jobs: int,
    use_class_weight: bool,
    n_splits_cv: int,
) -> Tuple[RandomForestClassifier, Dict]:
    param_grid = {
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "max_features": params["max_features"],
    }
    base = RandomForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced" if use_class_weight else None,
    )
    n_iter = _choose_n_iter(param_grid)
    cv = cv_splitter(n_splits_cv, random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    best_params = search.best_params_
    best_model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced" if use_class_weight else None,
    )
    best_model.fit(X, y)
    return best_model, {"best_params": best_params, "cv_results": search.cv_results_}


def train_logreg(
    X,
    y,
    groups,
    params: Dict,
    scoring: str,
    random_state: int,
    n_jobs: int,
    use_class_weight: bool,
    n_splits_cv: int,
) -> Tuple[Pipeline, Dict]:
    param_grid = {
        "clf__C": params["C"],
        "clf__penalty": params["penalty"],
        "clf__solver": params["solver"],
    }
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=n_jobs,
                    class_weight="balanced" if use_class_weight else None,
                    random_state=random_state,
                ),
            ),
        ]
    )
    n_iter = _choose_n_iter(param_grid)
    cv = cv_splitter(n_splits_cv, random_state)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    best_params = search.best_params_
    best_model = search.best_estimator_
    return best_model, {"best_params": best_params, "cv_results": search.cv_results_}


def train_xgboost(
    X,
    y,
    groups,
    params: Dict,
    scoring: str,
    random_state: int,
    n_jobs: int,
    n_splits_cv: int,
    scale_pos_weight: float,
    eval_set=None,
    early_stopping_rounds: int = 50,
) -> Tuple[XGBClassifier, Dict]:
    param_grid = {
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "n_estimators": params["n_estimators"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_lambda": params["reg_lambda"],
    }
    n_iter = _choose_n_iter(param_grid)
    cv = cv_splitter(n_splits_cv, random_state)
    base = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        subsample=1.0,
        colsample_bytree=1.0,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    best_params = search.best_params_
    best_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    if eval_set is not None:
        # Current xgboost version in this environment does not expose callbacks/early_stopping_rounds
        best_model.fit(X, y, eval_set=eval_set, verbose=False)
    else:
        best_model.fit(X, y)
    return best_model, {"best_params": best_params, "cv_results": search.cv_results_}
