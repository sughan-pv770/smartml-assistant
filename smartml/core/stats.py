"""
smartml.core.stats
==================
In-depth statistical diagnostics, distribution summaries, and normality tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    count: int
    missing: int
    unique: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    shapiro_p: Optional[float] = None
    is_normal: Optional[bool] = None


def summarize_column(df: pd.DataFrame, col: str) -> ColumnSummary:
    """Compute five-number summary, IQR, skewness, kurtosis, and Shapiro-Wilk test."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    s = df[col]
    count = int(s.count())
    missing = int(s.isna().sum())
    unique = int(s.nunique())
    dtype_str = str(s.dtype)

    if not pd.api.types.is_numeric_dtype(s) or count < 2:
        return ColumnSummary(
            name=col,
            dtype=dtype_str,
            count=count,
            missing=missing,
            unique=unique,
        )

    valid_s = s.dropna()
    q25, med, q75 = valid_s.quantile([0.25, 0.5, 0.75])
    iqr_val = q75 - q25

    skew = float(valid_s.skew())
    kurt = float(valid_s.kurt())

    # Shapiro-Wilk normality test (on sample if large)
    sample = valid_s.sample(n=min(500, len(valid_s)), random_state=42) if len(valid_s) > 500 else valid_s
    try:
        shapiro_stat, p_val = stats.shapiro(sample)
        shapiro_p = round(float(p_val), 4)
        is_normal = shapiro_p >= 0.05
    except Exception:
        shapiro_p = None
        is_normal = None

    return ColumnSummary(
        name=col,
        dtype=dtype_str,
        count=count,
        missing=missing,
        unique=unique,
        mean=round(float(valid_s.mean()), 4),
        std=round(float(valid_s.std()), 4),
        min=round(float(valid_s.min()), 4),
        q25=round(float(q25), 4),
        median=round(float(med), 4),
        q75=round(float(q75), 4),
        max=round(float(valid_s.max()), 4),
        iqr=round(float(iqr_val), 4),
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        shapiro_p=shapiro_p,
        is_normal=is_normal,
    )


def compute_correlations(
    df: pd.DataFrame,
    col1: str,
    col2: Optional[str] = None,
    method: str = "pearson",
) -> Any:
    """Compute correlation between two columns or one column with all numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if col1 not in num_df.columns:
        raise KeyError(f"Column '{col1}' is not numeric or not in dataset.")

    if col2 is not None:
        if col2 not in num_df.columns:
            raise KeyError(f"Column '{col2}' is not numeric or not in dataset.")
        clean = num_df[[col1, col2]].dropna()
        if len(clean) < 2:
            return 0.0
        r, _ = (
            stats.pearsonr(clean[col1], clean[col2])
            if method == "pearson"
            else stats.spearmanr(clean[col1], clean[col2])
        )
        return round(float(r), 4)

    # Correlation of col1 with all other numeric columns
    corrs = {}
    for other in num_df.columns:
        if other == col1:
            continue
        clean = num_df[[col1, other]].dropna()
        if len(clean) >= 2:
            r = clean.corr(method=method).iloc[0, 1]
            corrs[other] = round(float(r), 4)
    return pd.Series(corrs).sort_values(ascending=False)
