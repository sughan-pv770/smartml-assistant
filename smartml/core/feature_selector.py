"""
smartml.core.feature_selector
================================
Automatic feature selection for tabular datasets.

Available strategies (applied as an ordered pipeline):
1. **Variance threshold** — drop near-zero-variance features.
2. **Correlation filter** — drop one from each highly-correlated pair
   (keeps the one with higher correlation to the target when a target
   is available, otherwise keeps the first alphabetically).
3. **Univariate selection** — SelectKBest with mutual information
   (classification) or F-statistic (regression).
4. **Model-based selection** — threshold on RandomForest feature
   importance (keeps features above the mean importance by default).

All strategies are composable: you can run a subset by passing the
appropriate flags to :meth:`FeatureSelector.select`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class FeatureSelectionResult:
    """Output of a :class:`FeatureSelector` run.

    Attributes:
        selected_features: Ordered list of features that were kept.
        dropped_features: Features that were removed.
        steps: Plain-English audit log of each filtering step.
        n_original: Number of features before selection.
        n_selected: Number of features after selection.
    """

    selected_features: list[str]
    dropped_features: list[str]
    steps: list[str]
    n_original: int
    n_selected: int

    @property
    def reduction_pct(self) -> float:
        """Percentage of features removed."""
        return round(
            (self.n_original - self.n_selected) / max(self.n_original, 1) * 100, 1
        )


class FeatureSelector:
    """Multi-strategy automatic feature selection.

    Args:
        variance_threshold: Minimum variance a feature must have to be kept.
            Set to 0 to disable (default keeps only near-zero-variance removal).
        correlation_threshold: Drop one from each pair whose absolute Pearson
            correlation exceeds this value (default 0.95).
        k_best: Number of features to retain via univariate scoring.
            If None, the univariate step is skipped.
        model_based: If True, run a RandomForest importance threshold as
            the final selection step.
        random_state: Random seed for the forest.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        k_best: Optional[int] = None,
        model_based: bool = False,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.k_best = k_best
        self.model_based = model_based
        self.random_state = random_state

    # ------------------------------------------------------------------ #

    def select(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        task: str = "classification",
    ) -> FeatureSelectionResult:
        """Run the full feature-selection pipeline.

        Args:
            X: Feature matrix (numeric only; encode categoricals first).
            y: Target vector. Required for univariate and model-based steps.
            task: ``"classification"`` or ``"regression"``.

        Returns:
            :class:`FeatureSelectionResult` with selected/dropped column lists
            and a plain-English audit log.
        """
        features = list(X.columns)
        steps: list[str] = []
        n_original = len(features)

        # 1. Variance threshold
        features, steps = self._variance_step(X[features], features, steps)

        # 2. Correlation filter
        features, steps = self._correlation_step(X[features], features, steps, y)

        # 3. Univariate selection
        if self.k_best is not None and y is not None and len(features) > self.k_best:
            features, steps = self._univariate_step(
                X[features], y, features, steps, task
            )

        # 4. Model-based selection
        if self.model_based and y is not None and len(features) > 1:
            features, steps = self._model_based_step(
                X[features], y, features, steps, task
            )

        if not steps:
            steps.append("No features were removed by any selection criterion.")

        dropped = [f for f in X.columns if f not in features]
        return FeatureSelectionResult(
            selected_features=features,
            dropped_features=dropped,
            steps=steps,
            n_original=n_original,
            n_selected=len(features),
        )

    # ------------------------------------------------------------------ #
    # Private steps
    # ------------------------------------------------------------------ #

    def _variance_step(
        self,
        X: pd.DataFrame,
        features: list[str],
        steps: list[str],
    ) -> tuple[list[str], list[str]]:
        """Remove near-zero-variance features."""
        try:
            numeric_X = X[features].select_dtypes(include=[np.number])
            if numeric_X.empty:
                return features, steps
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(numeric_X)
            kept_mask = selector.get_support()
            kept = list(numeric_X.columns[kept_mask])
            removed = [c for c in numeric_X.columns if c not in kept]
            # Keep non-numeric features untouched
            non_numeric = [c for c in features if c not in numeric_X.columns]
            new_features = kept + non_numeric
            if removed:
                steps.append(
                    f"Variance threshold (< {self.variance_threshold}): removed "
                    f"{len(removed)} near-zero-variance feature(s): {removed}."
                )
            return new_features, steps
        except Exception:
            return features, steps

    def _correlation_step(
        self,
        X: pd.DataFrame,
        features: list[str],
        steps: list[str],
        y: Optional[pd.Series],
    ) -> tuple[list[str], list[str]]:
        """Drop one column from each highly-correlated pair."""
        try:
            numeric_X = X[features].select_dtypes(include=[np.number])
            if numeric_X.shape[1] < 2:
                return features, steps

            corr = numeric_X.corr().abs()
            # Upper triangle only
            upper = corr.where(
                np.triu(np.ones(corr.shape, dtype=bool), k=1)
            )
            to_drop: set[str] = set()
            for col in upper.columns:
                high_corr = upper[col][upper[col] > self.correlation_threshold]
                for partner in high_corr.index:
                    # Drop the one with lower abs correlation to target (if available)
                    if y is not None and pd.api.types.is_numeric_dtype(y):
                        try:
                            corr_col = abs(float(X[[col]].corrwith(y).iloc[0]))
                            corr_partner = abs(float(X[[partner]].corrwith(y).iloc[0]))
                            to_drop.add(col if corr_col < corr_partner else partner)
                        except Exception:
                            to_drop.add(partner)
                    else:
                        to_drop.add(partner)

            non_numeric = [c for c in features if c not in numeric_X.columns]
            new_features = [c for c in features if c not in to_drop]
            if to_drop:
                steps.append(
                    f"Correlation filter (threshold={self.correlation_threshold}): "
                    f"removed {len(to_drop)} collinear feature(s): "
                    f"{sorted(to_drop)}."
                )
            return new_features, steps
        except Exception:
            return features, steps

    def _univariate_step(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
        steps: list[str],
        task: str,
    ) -> tuple[list[str], list[str]]:
        """Select top-k features via univariate scoring."""
        try:
            numeric_X = X[features].select_dtypes(include=[np.number])
            if numeric_X.empty or self.k_best is None:
                return features, steps

            k = min(self.k_best, numeric_X.shape[1])
            score_func = (
                mutual_info_classif if task == "classification" else f_regression
            )
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(numeric_X.fillna(0), y)
            kept = list(numeric_X.columns[selector.get_support()])
            removed = [c for c in numeric_X.columns if c not in kept]
            non_numeric = [c for c in features if c not in numeric_X.columns]
            new_features = kept + non_numeric
            if removed:
                steps.append(
                    f"Univariate selection (SelectKBest, k={k}): kept top {k} "
                    f"features; removed {len(removed)}: {removed}."
                )
            return new_features, steps
        except Exception:
            return features, steps

    def _model_based_step(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: list[str],
        steps: list[str],
        task: str,
    ) -> tuple[list[str], list[str]]:
        """Select features above mean RandomForest importance."""
        try:
            numeric_X = X[features].select_dtypes(include=[np.number]).fillna(0)
            if numeric_X.empty:
                return features, steps

            if task == "classification":
                forest = RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                )
            else:
                forest = RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                )
            forest.fit(numeric_X, y)
            importances = pd.Series(
                forest.feature_importances_, index=numeric_X.columns
            )
            threshold = importances.mean()
            kept = list(importances[importances >= threshold].index)
            removed = list(importances[importances < threshold].index)
            non_numeric = [c for c in features if c not in numeric_X.columns]
            new_features = kept + non_numeric
            if removed:
                steps.append(
                    f"Model-based selection (RandomForest, threshold=mean importance "
                    f"{threshold:.4f}): removed {len(removed)} low-importance "
                    f"feature(s): {removed}."
                )
            return new_features, steps
        except Exception:
            return features, steps
