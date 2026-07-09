"""
smartml.core.trainer
======================
Trains and compares a curated set of strong baseline models per task type,
picks the best by cross-validated score, and keeps a leaderboard for
explainability. Optional libraries (xgboost, lightgbm, catboost) are used
if installed, otherwise skipped silently.

The model roster covers the most common production-ready algorithms for
classification and regression.  New estimators can be added by appending to
the dict returned by :func:`_candidate_models` or via the plugin system.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    HuberRegressor,
    RidgeClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


# --------------------------------------------------------------------------- #
# Candidate model registry
# --------------------------------------------------------------------------- #

def _candidate_models(task: str, random_state: int = 42) -> dict[str, Any]:
    """Build the dictionary of candidate estimators for a given task.

    Core models (always available via scikit-learn):

    Classification
        Logistic Regression, Random Forest, Extra Trees, Gradient Boosting,
        HistGradientBoosting, GaussianNB, SVC, RidgeClassifier, KNN.

    Regression
        Linear Regression, Ridge, Huber, Random Forest, Extra Trees,
        Gradient Boosting, HistGradientBoosting, SVR, KNN.

    Optional models (imported with try/except — missing libraries are
    silently ignored):
        XGBoost, LightGBM, CatBoost.

    Args:
        task: ``"classification"`` or ``"regression"``.
        random_state: Global random seed for reproducibility.

    Returns:
        Ordered dict mapping human-readable name → unfitted estimator.
    """
    if task == "classification":
        models: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=random_state, n_jobs=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=random_state
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                random_state=random_state
            ),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=random_state),
            "Ridge Classifier": RidgeClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
        }
    else:
        models = {
            "Linear Regression": LinearRegression(n_jobs=-1),
            "Ridge Regression": Ridge(random_state=random_state),
            "Huber Regressor": HuberRegressor(max_iter=300),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=random_state
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                random_state=random_state
            ),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
        }

    # ----- optional boosting libraries -----
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore
        if task == "classification":
            models["XGBoost"] = XGBClassifier(
                eval_metric="logloss", random_state=random_state,
                verbosity=0, use_label_encoder=False,
            )
        else:
            models["XGBoost"] = XGBRegressor(
                random_state=random_state, verbosity=0
            )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
        if task == "classification":
            models["LightGBM"] = LGBMClassifier(
                random_state=random_state, verbosity=-1, n_jobs=-1
            )
        else:
            models["LightGBM"] = LGBMRegressor(
                random_state=random_state, verbosity=-1, n_jobs=-1
            )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
        if task == "classification":
            models["CatBoost"] = CatBoostClassifier(
                random_seed=random_state, verbose=0
            )
        else:
            models["CatBoost"] = CatBoostRegressor(
                random_seed=random_state, verbose=0
            )
    except ImportError:
        pass

    return models


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class LeaderboardEntry:
    """One row of the model comparison leaderboard.

    Attributes:
        name: Human-readable algorithm name.
        cv_score: Mean cross-validated score.
        cv_std: Standard deviation of cross-validated scores.
        test_metrics: Dict of hold-out test metrics (e.g. accuracy, r2, rmse).
        fit_time_s: Approximate training + CV wall-clock time in seconds.
    """

    name: str
    cv_score: float
    cv_std: float
    test_metrics: dict[str, float]
    fit_time_s: float = 0.0


@dataclass
class TrainingResult:
    """Full output of a :class:`Trainer` run.

    Attributes:
        task: ``"classification"`` or ``"regression"``.
        leaderboard: All candidates ranked by CV score (descending).
        best_name: Name of the winning algorithm.
        best_model: Fitted estimator for the winner.
        best_reason: Natural-language explanation of why this model won.
        feature_names: Ordered list of feature column names the model expects.
        X_test: Hold-out feature matrix (used for post-training evaluation).
        y_test: Hold-out labels.
        scoring: Scikit-learn scoring string used for CV.
    """

    task: str
    leaderboard: list[LeaderboardEntry]
    best_name: str
    best_model: Any
    best_reason: str
    feature_names: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    scoring: str = ""


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

class Trainer:
    """Compare candidate models via cross-validation and fit the winner.

    Args:
        task: ``"classification"`` or ``"regression"``.
        random_state: Random seed for reproducibility.
        cv: Number of cross-validation folds.
        test_size: Fraction of data held out for final evaluation.

    Raises:
        ValueError: If *task* is not ``"classification"`` or ``"regression"``.
    """

    def __init__(
        self,
        task: str,
        random_state: int = 42,
        cv: int = 5,
        test_size: float = 0.2,
    ):
        if task not in ("classification", "regression"):
            raise ValueError(
                f"Unsupported task: '{task}'. "
                "Choose 'classification' or 'regression'."
            )
        self.task = task
        self.random_state = random_state
        self.cv = cv
        self.test_size = test_size

    # ------------------------------------------------------------------ #

    def run(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train all candidate models and return a ranked leaderboard.

        Args:
            X: Feature matrix (already cleaned / encoded).
            y: Target vector.

        Returns:
            :class:`TrainingResult` with leaderboard, best model, and
            the hold-out split for post-training evaluation.
        """
        import time

        stratify = (
            y if self.task == "classification" and y.nunique() > 1 else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        scoring = "f1_weighted" if self.task == "classification" else "r2"

        # Guard: ensure enough samples per class for the requested fold count
        if self.task == "classification":
            min_class_count = int(y_train.value_counts().min())
            n_splits = max(2, min(self.cv, min_class_count))
        else:
            n_splits = self.cv

        models = _candidate_models(self.task, self.random_state)
        entries: list[LeaderboardEntry] = []
        fitted: dict[str, Any] = {}

        for name, model in models.items():
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_val_score(
                        model, X_train, y_train, cv=n_splits,
                        scoring=scoring, n_jobs=-1,
                    )
                model.fit(X_train, y_train)
            except Exception as exc:
                # Silently skip models that fail (e.g. SVC on very large data)
                import logging
                logging.getLogger("smartml").debug(
                    "Model '%s' failed during CV: %s", name, exc
                )
                continue

            elapsed = time.perf_counter() - t0
            fitted[name] = model
            preds = model.predict(X_test)

            if self.task == "classification":
                test_metrics: dict[str, float] = {
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "f1_weighted": float(
                        f1_score(y_test, preds, average="weighted", zero_division=0)
                    ),
                }
                # ROC-AUC (binary or macro-OVO for multi-class)
                try:
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X_test)
                        if prob.shape[1] == 2:
                            test_metrics["roc_auc"] = float(
                                roc_auc_score(y_test, prob[:, 1])
                            )
                        else:
                            test_metrics["roc_auc"] = float(
                                roc_auc_score(
                                    y_test, prob,
                                    multi_class="ovo", average="macro"
                                )
                            )
                except Exception:
                    pass
            else:
                mse = mean_squared_error(y_test, preds)
                test_metrics = {
                    "r2": float(r2_score(y_test, preds)),
                    "mae": float(mean_absolute_error(y_test, preds)),
                    "rmse": float(np.sqrt(mse)),
                }
                try:
                    test_metrics["mape"] = float(
                        mean_absolute_percentage_error(y_test, preds) * 100
                    )
                except Exception:
                    pass

            entries.append(
                LeaderboardEntry(
                    name=name,
                    cv_score=float(scores.mean()),
                    cv_std=float(scores.std()),
                    test_metrics=test_metrics,
                    fit_time_s=round(elapsed, 2),
                )
            )

        if not entries:
            raise RuntimeError(
                "All candidate models failed during training. "
                "Check that the dataset is valid and has enough rows."
            )

        entries.sort(key=lambda e: e.cv_score, reverse=True)
        best = entries[0]
        best_model = fitted[best.name]

        reason = self._explain_choice(entries)

        return TrainingResult(
            task=self.task,
            leaderboard=entries,
            best_name=best.name,
            best_model=best_model,
            best_reason=reason,
            feature_names=list(X.columns),
            X_test=X_test,
            y_test=y_test,
            scoring=scoring,
        )

    # ------------------------------------------------------------------ #

    def _explain_choice(self, entries: list[LeaderboardEntry]) -> str:
        """Build a natural-language explanation of why the best model won.

        Args:
            entries: Leaderboard sorted descending by CV score.

        Returns:
            Human-readable explanation string.
        """
        metric_name = "F1 score (weighted)" if self.task == "classification" else "R²"
        best = entries[0]
        runner_up = entries[1] if len(entries) > 1 else None

        text = (
            f"{best.name} was selected because it achieved the highest "
            f"cross-validated {metric_name} of {best.cv_score:.4f} "
            f"(± {best.cv_std:.4f})"
        )
        if runner_up is not None:
            margin = best.cv_score - runner_up.cv_score
            text += (
                f", outperforming the runner-up {runner_up.name} "
                f"({runner_up.cv_score:.4f}) by {margin:.4f} points"
            )
        text += "."

        name_lower = best.name.lower()
        if any(kw in name_lower for kw in ("forest", "boosting", "xgboost", "lightgbm", "catboost", "extra trees")):
            text += (
                " As a tree-based ensemble, it captures non-linear relationships "
                "and feature interactions that simpler linear models miss."
            )
        elif any(kw in name_lower for kw in ("linear", "logistic", "ridge", "lasso")):
            text += (
                " Its strong performance suggests the relationship between the "
                "features and the target is largely linear, so a simpler, "
                "more interpretable model was sufficient."
            )
        elif "neighbor" in name_lower or "knn" in name_lower:
            text += (
                " This suggests the target is well predicted by local similarity "
                "between data points rather than a fixed global formula."
            )
        elif "svm" in name_lower or "svc" in name_lower or "svr" in name_lower:
            text += (
                " The SVM's margin-maximisation objective worked well here, "
                "likely because the data has a clear decision boundary in "
                "the kernel-transformed feature space."
            )
        elif "bayes" in name_lower:
            text += (
                " Naive Bayes' assumption of conditional feature independence "
                "proved valid for this dataset, enabling fast and accurate inference."
            )

        return text
