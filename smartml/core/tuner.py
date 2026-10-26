"""
smartml.core.tuner
==================
Hyperparameter optimization using RandomizedSearchCV.
Supports LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, GradientBoosting, Ridge, and LogisticRegression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from .trainer import _candidate_models


TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Extra Trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
    },
    "CatBoost": {
        "iterations": [100, 200],
        "learning_rate": [0.03, 0.1],
        "depth": [4, 6, 8],
    },
}


@dataclass
class TuningResult:
    model_name: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_score: float
    baseline_score: float
    scoring: str
    delta: float


class ModelTuner:
    """Hyperparameter tuner for SmartML models."""

    def __init__(
        self,
        task: str = "classification",
        n_iter: int = 10,
        cv: int = 3,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

    def tune(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        baseline_score: float = 0.0,
    ) -> TuningResult:
        candidates = _candidate_models(self.task, random_state=self.random_state)
        # Match case-insensitively
        matched_name = None
        for name in candidates:
            if name.lower() == model_name.lower():
                matched_name = name
                break

        if matched_name is None:
            available = ", ".join(candidates.keys())
            raise ValueError(
                f"Model '{model_name}' is not installed or recognized. Available models: {available}"
            )

        estimator = candidates[matched_name]
        grid = TUNING_GRIDS.get(matched_name, {"n_estimators": [100, 200]})

        scoring = "f1_weighted" if self.task == "classification" else "r2"

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=grid,
            n_iter=min(self.n_iter, np.prod([len(v) for v in grid.values()])),
            cv=self.cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        delta = float(search.best_score_ - baseline_score)

        return TuningResult(
            model_name=matched_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=float(search.best_score_),
            baseline_score=baseline_score,
            scoring=scoring,
            delta=delta,
        )
