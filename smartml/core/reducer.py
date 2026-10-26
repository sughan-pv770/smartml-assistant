"""
smartml.core.reducer
===================
Dimensionality reduction using Principal Component Analysis (PCA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    df_transformed: pd.DataFrame
    n_components: int
    explained_variance_ratio: list[float]
    total_explained_variance: float
    pca_model: PCA
    steps: list[str]


class PCAReducer:
    """Dimensionality reduction wrapper around PCA.

    Args:
        n_components: Desired integer component count (e.g. 3) or variance fraction (e.g. 0.95).
        scale: Whether to standardize features before PCA.
    """

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        scale: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.random_state = random_state

    def fit_transform(
        self,
        X: pd.DataFrame,
        target: Optional[str] = None,
    ) -> PCAResult:
        steps = []
        work = X.copy()
        if target and target in work.columns:
            work = work.drop(columns=[target])

        num_df = work.select_dtypes(include=[np.number]).dropna()
        if num_df.shape[1] < 2:
            raise ValueError("PCA requires at least 2 numeric features.")

        if self.scale:
            scaler = StandardScaler()
            scaled_vals = scaler.fit_transform(num_df)
            steps.append("Standardized numeric features before PCA.")
        else:
            scaled_vals = num_df.values

        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        transformed = pca.fit_transform(scaled_vals)

        n_comps = transformed.shape[1]
        col_names = [f"PCA_{i + 1}" for i in range(n_comps)]
        df_res = pd.DataFrame(transformed, columns=col_names, index=num_df.index)

        var_ratios = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        total_var = round(float(np.sum(pca.explained_variance_ratio_)), 4)

        steps.append(
            f"Fitted PCA: reduced {num_df.shape[1]} features to {n_comps} component(s), "
            f"explaining {total_var * 100:.1f}% of total variance."
        )

        return PCAResult(
            df_transformed=df_res,
            n_components=n_comps,
            explained_variance_ratio=var_ratios,
            total_explained_variance=total_var,
            pca_model=pca,
            steps=steps,
        )
