"""
smartml.core.explainer
========================
Model-agnostic explainability: feature importance (native or permutation
fallback) plus a natural-language summary, so the Assistant never has to
say "I don't know why."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ImportanceResult:
    importances: pd.Series  # sorted descending, indexed by feature name
    method: str
    narrative: str


def explain_feature_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    task: str,
    random_state: int = 42,
) -> ImportanceResult:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        method = "native (model.feature_importances_)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef).reshape(-1) if coef.ndim == 1 else np.abs(coef).mean(axis=0)
        method = "native (absolute coefficient magnitude)"
    else:
        scoring = "f1_weighted" if task == "classification" else "r2"
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=random_state, scoring=scoring
        )
        values = result.importances_mean
        method = "permutation importance"

    n = min(len(values), len(feature_names))
    series = pd.Series(values[:n], index=feature_names[:n]).sort_values(ascending=False)

    top = series.head(3)
    if top.sum() <= 0:
        narrative = "No single feature dominated; importance was spread fairly evenly across features."
    else:
        share = (top.sum() / max(series.sum(), 1e-12)) * 100
        names = ", ".join(f"'{idx}'" for idx in top.index)
        narrative = (
            f"The top 3 features ({names}) account for roughly {share:.1f}% of the model's "
            f"total feature importance, with '{top.index[0]}' being the single strongest "
            f"predictor."
        )

    return ImportanceResult(importances=series, method=method, narrative=narrative)
