"""
smartml.core.preprocessor
==========================
A pragmatic auto-cleaning pipeline. Every transformation it applies is
logged as a plain-English sentence so the Assistant can tell the user
exactly what happened to their data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from .analyzer import DatasetReport, analyze


@dataclass
class CleaningResult:
    df: pd.DataFrame
    steps: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    encoders: dict = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    impute_values: dict = field(default_factory=dict)


class Preprocessor:
    """Automatic dataset cleaning.

    Strategy (deliberately conservative / explainable, not exhaustive):
      1. Drop exact duplicate rows.
      2. Drop constant columns (zero information).
      3. Impute missing values: median for numeric, mode for categorical.
      4. Encode categoricals: one-hot if low cardinality, ordinal if high.
      5. Scale numeric features with StandardScaler.
    """

    HIGH_CARDINALITY_THRESHOLD = 15

    def clean(self, df: pd.DataFrame, target: Optional[str] = None) -> CleaningResult:
        steps: list[str] = []
        work = df.copy()

        # 1. duplicates
        n_dupes = int(work.duplicated().sum())
        if n_dupes:
            work = work.drop_duplicates().reset_index(drop=True)
            steps.append(f"Removed {n_dupes} duplicate row(s).")

        # 2. constant columns
        const_cols = [c for c in work.columns if c != target and work[c].nunique(dropna=False) <= 1]
        if const_cols:
            work = work.drop(columns=const_cols)
            steps.append(f"Dropped {len(const_cols)} constant column(s): {const_cols}.")

        report: DatasetReport = analyze(work, target=None)

        # 2b. Expand datetime columns into numeric parts
        for col in report.datetime_cols:
            if col == target:
                continue
            try:
                s = pd.to_datetime(work[col], errors="coerce")
                work[f"{col}_year"] = s.dt.year.fillna(0).astype(int)
                work[f"{col}_month"] = s.dt.month.fillna(0).astype(int)
                work[f"{col}_day"] = s.dt.day.fillna(0).astype(int)
                work[f"{col}_dayofweek"] = s.dt.dayofweek.fillna(0).astype(int)
                work = work.drop(columns=[col])
                steps.append(f"Expanded datetime column '{col}' into year, month, day, dayofweek components.")
            except Exception:
                work = work.drop(columns=[col])
                steps.append(f"Dropped unparseable datetime column '{col}'.")

        # 2c. Drop free text columns (high word count)
        for col in report.text_cols:
            if col == target:
                continue
            work = work.drop(columns=[col])
            steps.append(f"Dropped free-text column '{col}' to prevent unencoded text features.")

        # Re-profile after datetime expansion and text column drops
        report = analyze(work, target=None)
        numeric_cols = [c for c in report.numeric_cols if c != target]
        categorical_cols = [c for c in report.categorical_cols if c != target]
        # treat booleans as categorical for encoding purposes
        categorical_cols += [c for c in report.boolean_cols if c != target]

        # 3. missing value imputation
        impute_values: dict = {}
        for col in numeric_cols:
            n_missing = int(work[col].isna().sum())
            if n_missing:
                median = work[col].median()
                work[col] = work[col].fillna(median)
                impute_values[col] = median
                steps.append(
                    f"Imputed {n_missing} missing value(s) in numeric column "
                    f"'{col}' with the median ({median:.3g})."
                )
        for col in categorical_cols:
            n_missing = int(work[col].isna().sum())
            if n_missing:
                mode = work[col].mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) else "missing"
                work[col] = work[col].fillna(fill_value)
                impute_values[col] = fill_value
                steps.append(
                    f"Imputed {n_missing} missing value(s) in categorical column "
                    f"'{col}' with the most frequent value ('{fill_value}')."
                )

        if target is not None and target in work.columns:
            n_missing_target = int(work[target].isna().sum())
            if n_missing_target:
                before = len(work)
                work = work.dropna(subset=[target]).reset_index(drop=True)
                steps.append(
                    f"Dropped {before - len(work)} row(s) with missing target '{target}' "
                    f"(imputing the label would bias training)."
                )

        # 4. encoding
        encoders: dict = {}
        for col in categorical_cols:
            n_unique = work[col].nunique(dropna=True)
            if n_unique <= self.HIGH_CARDINALITY_THRESHOLD:
                dummies = pd.get_dummies(work[col], prefix=col, dummy_na=False)
                work = pd.concat([work.drop(columns=[col]), dummies], axis=1)
                steps.append(
                    f"One-hot encoded '{col}' ({n_unique} categories -> "
                    f"{dummies.shape[1]} binary columns)."
                )
                encoders[col] = {"type": "onehot", "categories": list(dummies.columns)}
            else:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                work[col] = enc.fit_transform(work[[col]].astype(str))
                steps.append(
                    f"Ordinal-encoded high-cardinality column '{col}' "
                    f"({n_unique} categories) to avoid a dimensionality explosion."
                )
                encoders[col] = {"type": "ordinal", "encoder": enc}

        # 5. scaling (numeric feature columns only, never the target)
        scaler = None
        scale_cols = [c for c in numeric_cols if c in work.columns]
        if scale_cols:
            scaler = StandardScaler()
            work[scale_cols] = scaler.fit_transform(work[scale_cols])
            steps.append(
                f"Standard-scaled {len(scale_cols)} numeric feature column(s) "
                f"(zero mean, unit variance)."
            )

        feature_names = [c for c in work.columns if c != target]

        return CleaningResult(
            df=work,
            steps=steps,
            feature_names=feature_names,
            encoders=encoders,
            scaler=scaler,
            numeric_cols=scale_cols,
            categorical_cols=categorical_cols,
            impute_values=impute_values,
        )

    @staticmethod
    def transform_new(df: pd.DataFrame, cleaning: "CleaningResult") -> pd.DataFrame:
        """Apply an *already-fitted* cleaning pipeline (encoders + scaler) to
        new, unseen data (e.g. a CSV passed to ``predict``). Never refits.
        """
        work = df.copy()

        # re-apply the exact fill values learned during training (median /
        # most-frequent-category) so prediction-time missing data doesn't
        # silently break the model.
        for col, fill_value in cleaning.impute_values.items():
            if col in work.columns and work[col].isna().any():
                work[col] = work[col].fillna(fill_value)

        for col, info in cleaning.encoders.items():
            if col not in work.columns:
                continue
            if info["type"] == "onehot":
                dummies = pd.get_dummies(work[col], prefix=col, dummy_na=False)
                # align to the exact dummy columns seen during training
                dummies = dummies.reindex(columns=info["categories"], fill_value=0)
                work = pd.concat([work.drop(columns=[col]), dummies], axis=1)
            else:  # ordinal
                enc = info["encoder"]
                work[col] = enc.transform(work[[col]].astype(str))

        if cleaning.scaler is not None and cleaning.numeric_cols:
            for c in cleaning.numeric_cols:
                if c not in work.columns:
                    work[c] = 0.0
            work[cleaning.numeric_cols] = cleaning.scaler.transform(work[cleaning.numeric_cols])

        # align final column set/order to what the model was trained on
        for col in cleaning.feature_names:
            if col not in work.columns:
                work[col] = 0
        work = work[cleaning.feature_names]
        return work
