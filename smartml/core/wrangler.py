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

    @property
    def can_undo(self) -> bool:
        return len(self._history) > 0

    def undo(self) -> str:
        """Revert the latest wrangling action."""
        if not self._history:
            return "Nothing to undo."
        last_action = self._history.pop()
        self.current_df = last_action.previous_df
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
