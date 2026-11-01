"""
smartml.core.wrangler
====================
Interactive, undoable data wrangling with a 10-step history stack.
Strict safety guarantees:
  - Cannot drop the target column
  - Cannot filter out 100% of rows
  - Filter operations use a safe operator table without eval/exec
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
}


@dataclass
class WranglingAction:
    description: str
    previous_df: pd.DataFrame


class DataWrangler:
    """Manages an undoable data manipulation session.

    Args:
        df: Initial working DataFrame.
        target: Optional target column to protect against deletion.
        max_history: Number of undo steps preserved (default 10).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        max_history: int = 10,
    ) -> None:
        self.current_df = df.copy()
        self.target = target
        self.max_history = max_history
        self._history: list[WranglingAction] = []
        self.version: int = 0

    @property
    def history(self) -> list[WranglingAction]:
        """List of previous wrangling states in the undo stack."""
        return list(self._history)

    def _save_state(self, description: str) -> None:
        self._history.append(
            WranglingAction(description=description, previous_df=self.current_df.copy())
        )
        if len(self._history) > self.max_history:
            self._history.pop(0)
        self.version += 1

    @property
    def can_undo(self) -> bool:
        return len(self._history) > 0

    def undo(self) -> str:
        """Revert the latest wrangling action."""
        if not self._history:
            return "Nothing to undo."
        last_action = self._history.pop()
        self.current_df = last_action.previous_df
        self.version += 1
        return f"Undid: {last_action.description}. Restored {len(self.current_df)} rows and {len(self.current_df.columns)} columns."

    def drop_column(self, column: str) -> str:
        """Safely drop a column."""
        if column not in self.current_df.columns:
            return f"Column '{column}' does not exist in the dataset."
        if column == self.target:
            return f"Cannot drop target column '{column}'."
        if len(self.current_df.columns) <= 1:
            return "Cannot drop the last remaining column in the dataset."

        self._save_state(f"Drop column '{column}'")
        self.current_df = self.current_df.drop(columns=[column])
        return f"Dropped column '{column}'. {len(self.current_df.columns)} columns remaining."

    def rename_column(self, old_name: str, new_name: str) -> str:
        """Rename a column."""
        if old_name not in self.current_df.columns:
            return f"Column '{old_name}' not found."
        new_name = new_name.strip()
        if not new_name:
            return "New column name cannot be empty."
        if new_name in self.current_df.columns:
            return f"A column named '{new_name}' already exists."

        self._save_state(f"Rename '{old_name}' to '{new_name}'")
        self.current_df = self.current_df.rename(columns={old_name: new_name})
        if self.target == old_name:
            self.target = new_name
        return f"Renamed column '{old_name}' to '{new_name}'."

    def replace_value(self, column: str, old_val: Any, new_val: Any) -> str:
        """Replace specific values in a column safely."""
        if column not in self.current_df.columns:
            return f"Column '{column}' not found."

        s = self.current_df[column]
        # Match type
        if pd.api.types.is_numeric_dtype(s):
            try:
                old_val = float(old_val)
                new_val = float(new_val)
            except (ValueError, TypeError):
                return f"Column '{column}' is numeric, but replace values could not be parsed as numbers."
        else:
            old_val = str(old_val)
            new_val = str(new_val)

        count = int((s == old_val).sum())
        if count == 0:
            return f"No occurrences of '{old_val}' found in column '{column}'."

        self._save_state(f"Replace '{old_val}' with '{new_val}' in '{column}'")
        self.current_df[column] = s.replace(old_val, new_val)
        return f"Replaced {count} occurrence(s) of '{old_val}' with '{new_val}' in '{column}'."

    def fill_na(self, column: str, value: Any) -> str:
        """Fill missing values in a column."""
        if column not in self.current_df.columns:
            return f"Column '{column}' not found."

        s = self.current_df[column]
        missing_count = int(s.isna().sum())
        if missing_count == 0:
            return f"Column '{column}' has no missing values."

        if pd.api.types.is_numeric_dtype(s):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return f"Cannot fill numeric column '{column}' with non-numeric value '{value}'."

        self._save_state(f"Fill NA in '{column}' with '{value}'")
        self.current_df[column] = s.fillna(value)
        return f"Filled {missing_count} missing value(s) in '{column}' with '{value}'."

    def filter_rows(self, column: str, op_str: str, value: Any) -> str:
        """Filter rows using a strict operator table without eval/exec."""
        if column not in self.current_df.columns:
            return f"Column '{column}' not found."

        op_str = op_str.strip()
        if op_str not in OPERATORS and op_str != "contains":
            valid = ", ".join(list(OPERATORS.keys()) + ["contains"])
            return f"Unsupported operator '{op_str}'. Supported operators: {valid}"

        s = self.current_df[column]

        # Handle string containment
        if op_str == "contains":
            mask = s.astype(str).str.contains(str(value), case=False, na=False)
        else:
            # Handle numeric / comparison operators
            if pd.api.types.is_numeric_dtype(s):
                try:
                    num_val = float(value)
                    op_func = OPERATORS[op_str]
                    mask = op_func(s, num_val)
                except Exception as exc:
                    return f"Invalid numeric comparison on '{column}': {exc}"
            else:
                op_func = OPERATORS[op_str]
                mask = op_func(s.astype(str), str(value))

        n_kept = int(mask.sum())
        if n_kept == 0:
            return f"Filter would drop 100% of rows (0 rows match '{column} {op_str} {value}'). Operation refused."

        dropped = len(self.current_df) - n_kept
        self._save_state(f"Filter rows where {column} {op_str} {value}")
        self.current_df = self.current_df[mask].reset_index(drop=True)
        return f"Filtered rows: kept {n_kept} rows, dropped {dropped} row(s)."

    def remove_outliers(self, column: Optional[str] = None, factor: float = 1.5) -> str:
        """Remove IQR outliers on a specific column or across all numeric non-target columns."""
        if column is not None:
            if column not in self.current_df.columns:
                return f"Column '{column}' not found."
            cols = [column]
        else:
            cols = [
                c for c in self.current_df.select_dtypes(include=["number"]).columns
                if c != self.target
            ]
        if not cols:
            return "No numeric columns available for outlier removal."

        mask = pd.Series(True, index=self.current_df.index)
        for c in cols:
            s = self.current_df[c].dropna()
            if len(s) < 4:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            low = q1 - factor * iqr
            high = q3 + factor * iqr
            col_mask = self.current_df[c].isna() | ((self.current_df[c] >= low) & (self.current_df[c] <= high))
            mask &= col_mask

        n_kept = int(mask.sum())
        dropped = len(self.current_df) - n_kept
        if dropped == 0:
            return "Removed 0 outlier row(s) — no extreme values outside the IQR fence."
        if n_kept == 0:
            return "Outlier removal would drop 100% of rows. Operation refused."

        self._save_state(f"Remove outliers ({dropped} rows)")
        self.current_df = self.current_df[mask].reset_index(drop=True)
        return f"Removed {dropped} outlier row(s); {n_kept} rows remaining."

    def convert_column_type(self, column: str, target_type: str) -> str:
        """Convert a column to int, float, str/category, or datetime."""
        if column not in self.current_df.columns:
            return f"Column '{column}' not found."
        t = target_type.strip().lower()
        s = self.current_df[column]
        try:
            if t in ("int", "integer"):
                converted = pd.to_numeric(s, errors="raise").astype("int64")
            elif t in ("float", "numeric", "number", "double"):
                converted = pd.to_numeric(s, errors="raise").astype("float64")
            elif t in ("str", "string", "text", "category", "categorical"):
                converted = s.astype(str)
            elif t in ("date", "datetime", "timestamp"):
                converted = pd.to_datetime(s, errors="raise")
            else:
                return f"Unsupported type '{target_type}'. Choose int, float, string, or datetime."
        except Exception as exc:
            return f"Could not convert '{column}' to {target_type}: {exc}"

        if s.dtype == converted.dtype and s.equals(converted):
            return f"Column '{column}' is already of type {converted.dtype}."

        self._save_state(f"Convert '{column}' to {target_type}")
        self.current_df[column] = converted
        return f"Converted column '{column}' to {converted.dtype}."
