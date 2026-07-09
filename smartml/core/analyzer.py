"""
smartml.core.analyzer
======================
Lightweight dataset analysis used by the rest of the engine (and surfaced
in plain English by the conversational Assistant). Keeps no external
dependencies beyond pandas/numpy so it stays fast on every ``ask()`` call.

Key functions
-------------
analyze(df, target)
    Run the full diagnostic pass and return a :class:`DatasetReport`.
detect_task(df, target)
    Heuristically infer the ML task type from the target column.
compute_data_quality_score(report)
    Return a 0–100 quality score plus an emoji grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# DatasetReport
# --------------------------------------------------------------------------- #

@dataclass
class DatasetReport:
    """Comprehensive diagnostic snapshot of a dataset.

    All fields are computed by :func:`analyze` and exposed as plain Python
    types (lists, dicts, floats) so they can be safely serialised to JSON
    or rendered in a report without further processing.
    """

    # ---- shape ----
    n_rows: int
    n_cols: int
    memory_mb: float

    # ---- column taxonomy ----
    numeric_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    boolean_cols: list[str]
    text_cols: list[str]              # free-text columns (high avg word count)

    # ---- missing values ----
    missing_counts: dict[str, int]
    missing_pct: dict[str, float]
    duplicate_rows: int

    # ---- cardinality / variance ----
    constant_cols: list[str]
    near_constant_cols: list[str]     # one value covers ≥ 95% of rows
    high_cardinality_cols: list[str]

    # ---- distribution ----
    skewness: dict[str, float]        # per numeric column
    kurtosis: dict[str, float]        # per numeric column

    # ---- outliers ----
    outlier_counts: dict[str, int]    # IQR-based per numeric column

    # ---- correlations ----
    correlations_with_target: dict[str, float]  # Pearson r with target (numeric only)

    # ---- quality scores ----
    data_quality_score: float         # 0.0 – 100.0
    data_quality_grade: str           # emoji grade (A+ / B / C / D / F)

    # ---- supervised task ----
    target: Optional[str] = None
    task: Optional[str] = None
    task_reason: str = ""
    class_balance: Optional[dict] = field(default=None)

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def total_missing(self) -> int:
        """Total count of missing cells across all columns."""
        return int(sum(self.missing_counts.values()))

    @property
    def has_missing(self) -> bool:
        """True if any cell in the DataFrame is missing."""
        return self.total_missing > 0

    @property
    def missing_pct_total(self) -> float:
        """Overall percentage of missing cells (0.0 – 100.0)."""
        total_cells = self.n_rows * self.n_cols
        return round(self.total_missing / max(total_cells, 1) * 100, 2)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _split_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Classify each column into one of five semantic buckets.

    Returns:
        Tuple of (numeric, categorical, datetime, boolean, text) column lists.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    datetime: list[str] = []
    boolean: list[str] = []
    text: list[str] = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            # Heuristic: if the average word count is > 4 it is likely free text
            try:
                avg_words = s.dropna().astype(str).str.split().str.len().mean()
                if avg_words > 4:
                    text.append(col)
                else:
                    categorical.append(col)
            except Exception:
                categorical.append(col)

    return numeric, categorical, datetime, boolean, text


def _compute_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Count outliers per column using the 1.5×IQR rule.

    Args:
        df: DataFrame.
        cols: Numeric columns to check.

    Returns:
        Dict mapping column name → outlier count.
    """
    result: dict[str, int] = {}
    for col in cols:
        s = df[col].dropna()
        if len(s) < 4:
            result[col] = 0
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result[col] = int(((s < lower) | (s > upper)).sum())
    return result


def _compute_data_quality_score(
    n_rows: int,
    n_cols: int,
    missing_pct_total: float,
    duplicate_rows: int,
    constant_cols: list[str],
    outlier_counts: dict[str, int],
    numeric_cols: list[str],
) -> tuple[float, str]:
    """Compute a composite data quality score (0 – 100).

    Penalties:
    * Missing values  : up to −40 pts
    * Duplicate rows  : up to −20 pts
    * Constant cols   : up to −15 pts
    * Outlier density : up to −15 pts
    * Tiny dataset    : up to −10 pts

    Returns:
        Tuple of (score, grade_emoji).
    """
    score = 100.0

    # Missing
    score -= min(40.0, missing_pct_total * 2)

    # Duplicates
    dup_pct = duplicate_rows / max(n_rows, 1) * 100
    score -= min(20.0, dup_pct * 4)

    # Constant columns
    const_pct = len(constant_cols) / max(n_cols, 1) * 100
    score -= min(15.0, const_pct * 1.5)

    # Outliers
    if numeric_cols:
        total_outliers = sum(outlier_counts.values())
        total_numeric_cells = n_rows * len(numeric_cols)
        outlier_pct = total_outliers / max(total_numeric_cells, 1) * 100
        score -= min(15.0, outlier_pct * 3)

    # Very small dataset
    if n_rows < 50:
        score -= 10.0
    elif n_rows < 200:
        score -= 5.0

    score = max(0.0, min(100.0, round(score, 1)))

    if score >= 90:
        grade = "A+ 🌟"
    elif score >= 75:
        grade = "B 👍"
    elif score >= 60:
        grade = "C ⚠️"
    elif score >= 40:
        grade = "D 🔴"
    else:
        grade = "F ❌"

    return score, grade


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def detect_task(df: pd.DataFrame, target: str) -> tuple[str, str]:
    """Heuristically infer the ML task type from the target column.

    Decision rules (in order):
    1. Boolean target → binary classification.
    2. Non-numeric with 2 unique values → binary classification.
    3. Non-numeric with >2 unique values → multi-class classification.
    4. Numeric with exactly 2 distinct values → binary classification.
    5. Numeric with ≤20 distinct values AND cardinality ratio < 5% → multi-class.
    6. Everything else → regression.

    Args:
        df: The full DataFrame (target column must be present).
        target: Name of the target column.

    Returns:
        Tuple of (task_name, human_readable_reason).
    """
    y = df[target]
    n_unique = y.nunique(dropna=True)
    n_rows = len(y)

    if pd.api.types.is_bool_dtype(y):
        return "classification", (
            f"Target '{target}' is boolean, so this is a binary classification task."
        )

    if not pd.api.types.is_numeric_dtype(y):
        if n_unique == 2:
            return "classification", (
                f"Target '{target}' is non-numeric with 2 unique values, "
                f"so this is a binary classification task."
            )
        return "classification", (
            f"Target '{target}' is non-numeric with {n_unique} unique values, "
            f"so this is a multi-class classification task."
        )

    # numeric target — decide classification vs regression by cardinality ratio
    ratio = n_unique / max(n_rows, 1)
    if n_unique == 2:
        return "classification", (
            f"Target '{target}' is numeric but only takes 2 distinct values, "
            f"so this is treated as binary classification."
        )
    if n_unique <= 20 and ratio < 0.05:
        return "classification", (
            f"Target '{target}' is numeric with only {n_unique} distinct values "
            f"({ratio:.1%} of rows), which looks like encoded classes rather than "
            f"a continuous quantity, so this is treated as multi-class classification."
        )
    return "regression", (
        f"Target '{target}' is numeric and continuous ({n_unique} distinct values "
        f"across {n_rows} rows), so this is a regression task."
    )


def analyze(df: pd.DataFrame, target: Optional[str] = None) -> DatasetReport:
    """Run a comprehensive diagnostic pass on a DataFrame.

    Computes column taxonomy, missing value statistics, duplicate counts,
    cardinality analysis, outlier counts (IQR), skewness, kurtosis,
    correlation with the target, a data quality score, and (if *target*
    is provided) the inferred ML task type.

    Args:
        df: The dataset to analyse.
        target: Optional name of the target/label column.

    Returns:
        A fully-populated :class:`DatasetReport`.
    """
    numeric, categorical, datetime_cols, boolean, text = _split_columns(df)

    # ---- missing ----
    missing_counts: dict[str, int] = df.isna().sum().to_dict()
    missing_pct: dict[str, float] = {
        k: round(v / max(len(df), 1) * 100, 2) for k, v in missing_counts.items()
    }

    # ---- duplicates ----
    duplicate_rows = int(df.duplicated().sum())

    # ---- cardinality / variance ----
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    near_constant_cols = [
        c for c in df.columns
        if c not in constant_cols
        and df[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.95
    ]
    high_card_threshold = max(50, int(0.5 * len(df)))
    high_card_cols = [
        c for c in categorical
        if df[c].nunique(dropna=True) > high_card_threshold
    ]

    # ---- distribution ----
    skewness: dict[str, float] = {}
    kurtosis_vals: dict[str, float] = {}
    for col in numeric:
        try:
            skewness[col] = round(float(df[col].skew()), 4)
            kurtosis_vals[col] = round(float(df[col].kurtosis()), 4)
        except Exception:
            skewness[col] = 0.0
            kurtosis_vals[col] = 0.0

    # ---- outliers ----
    outlier_counts = _compute_outliers_iqr(df, numeric)

    # ---- correlations with target ----
    correlations_with_target: dict[str, float] = {}
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        for col in numeric:
            if col == target:
                continue
            try:
                corr = float(df[[col, target]].dropna().corr().iloc[0, 1])
                correlations_with_target[col] = round(corr, 4)
            except Exception:
                correlations_with_target[col] = 0.0

    # ---- quality score ----
    total_cells = len(df) * len(df.columns)
    total_missing = sum(missing_counts.values())
    missing_pct_total = total_missing / max(total_cells, 1) * 100
    quality_score, quality_grade = _compute_data_quality_score(
        n_rows=len(df),
        n_cols=len(df.columns),
        missing_pct_total=missing_pct_total,
        duplicate_rows=duplicate_rows,
        constant_cols=constant_cols,
        outlier_counts=outlier_counts,
        numeric_cols=numeric,
    )

    report = DatasetReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        numeric_cols=numeric,
        categorical_cols=categorical,
        datetime_cols=datetime_cols,
        boolean_cols=boolean,
        text_cols=text,
        missing_counts=missing_counts,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        constant_cols=constant_cols,
        near_constant_cols=near_constant_cols,
        high_cardinality_cols=high_card_cols,
        skewness=skewness,
        kurtosis=kurtosis_vals,
        outlier_counts=outlier_counts,
        correlations_with_target=correlations_with_target,
        data_quality_score=quality_score,
        data_quality_grade=quality_grade,
        target=target,
    )

    if target is not None and target in df.columns:
        task, reason = detect_task(df, target)
        report.task = task
        report.task_reason = reason
        if task == "classification":
            report.class_balance = (
                df[target].value_counts(normalize=True).round(3).to_dict()
            )

    return report
