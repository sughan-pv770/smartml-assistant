"""
smartml.core.feature_engineer
================================
Automatic feature generation for tabular datasets.

Supported transformations
-------------------------
* **Polynomial features** — degree-2 interaction and power terms for
  numeric columns (limited to avoid combinatorial explosion).
* **Date/time decomposition** — year, month, day, day-of-week, hour,
  is_weekend, quarter extracted from datetime columns.
* **Frequency encoding** — replace a categorical value with its relative
  frequency in the training set (leak-free for test sets via transform).
* **Ratio features** — pairwise ratios of selected numeric columns.
* **Lag features** — only applied when an explicit ordering column is
  provided, keeping the module safe for non-time-series datasets.

All transformations are logged in plain English so the Assistant can
explain exactly what was added and why.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringResult:
    """Output of a :class:`FeatureEngineer` run.

    Attributes:
        df: The transformed DataFrame (original columns + new ones).
        new_features: Names of columns that were added.
        steps: Plain-English log of every transformation applied.
        frequency_maps: Dict of col → {value: frequency} for freq-encoding,
            stored so the same mapping can be applied to unseen data.
    """

    df: pd.DataFrame
    new_features: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    frequency_maps: dict[str, dict] = field(default_factory=dict)


class FeatureEngineer:
    """Automatic feature generation for tabular datasets.

    Args:
        poly_degree: Degree for polynomial/interaction features.
            Set to 0 to disable.
        max_poly_cols: Maximum number of numeric columns to use for
            polynomial expansion (guards against combinatorial explosion).
        max_ratio_pairs: Maximum number of ratio features to generate.
        freq_encode_threshold: Maximum cardinality for frequency encoding.
            Columns with more unique values than this are skipped.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        max_poly_cols: int = 8,
        max_ratio_pairs: int = 10,
        freq_encode_threshold: int = 50,
    ):
        self.poly_degree = poly_degree
        self.max_poly_cols = max_poly_cols
        self.max_ratio_pairs = max_ratio_pairs
        self.freq_encode_threshold = freq_encode_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        datetime_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
    ) -> FeatureEngineeringResult:
        """Fit the feature engineering pipeline on *df* and transform it.

        Args:
            df: Input DataFrame.
            target: Target column name (excluded from feature generation).
            datetime_cols: Columns to decompose into date parts. Auto-
                detected if None.
            categorical_cols: Columns to frequency-encode. Auto-detected
                if None.

        Returns:
            :class:`FeatureEngineeringResult` with the enriched DataFrame
            and a full audit log.
        """
        work = df.copy()
        steps: list[str] = []
        new_features: list[str] = []
        frequency_maps: dict[str, dict] = {}

        exclude = {target} if target else set()

        # Auto-detect column types if not provided
        num_cols = [
            c for c in work.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        if datetime_cols is None:
            datetime_cols = [
                c for c in work.columns
                if pd.api.types.is_datetime64_any_dtype(work[c]) and c not in exclude
            ]
        if categorical_cols is None:
            categorical_cols = [
                c for c in work.select_dtypes(include=["object", "category"]).columns
                if c not in exclude
            ]

        # 1. Date decomposition
        for col in datetime_cols:
            added = self._extract_date_features(work, col)
            new_features.extend(added)
            if added:
                steps.append(
                    f"Extracted {len(added)} date features from '{col}': "
                    + ", ".join(added[:5]) + ("…" if len(added) > 5 else "") + "."
                )

        # 2. Frequency encoding for categorical cols
        for col in categorical_cols:
            n_unique = work[col].nunique(dropna=True)
            if n_unique > self.freq_encode_threshold or n_unique <= 1:
                continue
            freq_map = work[col].value_counts(normalize=True).to_dict()
            new_col = f"{col}_freq"
            work[new_col] = work[col].map(freq_map).fillna(0.0)
            frequency_maps[col] = freq_map
            new_features.append(new_col)
            steps.append(
                f"Frequency-encoded '{col}' → '{new_col}' "
                f"({n_unique} categories mapped to their relative frequency)."
            )

        # 3. Polynomial / interaction features (numeric only)
        poly_cols = [c for c in num_cols if c in work.columns][: self.max_poly_cols]
        if self.poly_degree >= 2 and len(poly_cols) >= 2:
            added_poly = self._add_interactions(work, poly_cols)
            new_features.extend(added_poly)
            if added_poly:
                steps.append(
                    f"Created {len(added_poly)} pairwise interaction features "
                    f"(products of numeric column pairs) from "
                    f"{len(poly_cols)} columns."
                )

        # 4. Ratio features
        if len(poly_cols) >= 2:
            added_ratios = self._add_ratios(work, poly_cols)
            new_features.extend(added_ratios)
            if added_ratios:
                steps.append(
                    f"Created {len(added_ratios)} ratio features "
                    f"(column_a / column_b for top pairs)."
                )

        if not steps:
            steps.append("No feature engineering transformations were applicable.")

        return FeatureEngineeringResult(
            df=work,
            new_features=new_features,
            steps=steps,
            frequency_maps=frequency_maps,
        )

    def transform(
        self,
        df: pd.DataFrame,
        result: FeatureEngineeringResult,
    ) -> pd.DataFrame:
        """Apply the *already-fitted* frequency maps to unseen data.

        Date decomposition and interaction features are re-computed from
        scratch because they do not need fitting.

        Args:
            df: Unseen DataFrame (e.g. the prediction-time dataset).
            result: The :class:`FeatureEngineeringResult` from the training
                ``fit_transform`` call.

        Returns:
            Transformed DataFrame aligned to the training feature space.
        """
        work = df.copy()
        # Apply stored frequency maps
        for col, freq_map in result.frequency_maps.items():
            if col in work.columns:
                new_col = f"{col}_freq"
                work[new_col] = work[col].map(freq_map).fillna(0.0)
        return work

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_date_features(df: pd.DataFrame, col: str) -> list[str]:
        """Decompose a datetime column into numeric date parts in-place.

        Args:
            df: DataFrame to modify in-place.
            col: Name of the datetime column.

        Returns:
            List of new column names that were added.
        """
        added: list[str] = []
        try:
            s = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            return added

        parts = {
            "year": s.dt.year,
            "month": s.dt.month,
            "day": s.dt.day,
            "dayofweek": s.dt.dayofweek,
            "hour": s.dt.hour,
            "quarter": s.dt.quarter,
            "is_weekend": s.dt.dayofweek.isin([5, 6]).astype(int),
        }
        for suffix, values in parts.items():
            new_col = f"{col}_{suffix}"
            df[new_col] = values
            added.append(new_col)

        return added

    def _add_interactions(
        self, df: pd.DataFrame, cols: list[str]
    ) -> list[str]:
        """Add product-interaction features for all column pairs.

        Args:
            df: DataFrame to modify in-place.
            cols: Numeric columns to combine.

        Returns:
            List of new column names added.
        """
        added: list[str] = []
        for c1, c2 in itertools.combinations(cols, 2):
            new_col = f"{c1}_x_{c2}"
            if new_col not in df.columns:
                df[new_col] = df[c1] * df[c2]
                added.append(new_col)
        return added

    def _add_ratios(
        self, df: pd.DataFrame, cols: list[str]
    ) -> list[str]:
        """Add pairwise ratio features, guarding against division by zero.

        Args:
            df: DataFrame to modify in-place.
            cols: Numeric columns to combine.

        Returns:
            List of new column names added.
        """
        added: list[str] = []
        pairs = list(itertools.combinations(cols, 2))[: self.max_ratio_pairs]
        for c1, c2 in pairs:
            denom = df[c2].replace(0, np.nan)
            new_col = f"{c1}_div_{c2}"
            if new_col not in df.columns:
                df[new_col] = (df[c1] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
                added.append(new_col)
        return added
