"""
smartml.core.textutil
=====================
Phonetic, fuzzy, and substring matching utilities for resolving column names
and expressions safely without eval/exec.
"""

from __future__ import annotations

import difflib
import re
from typing import Optional, Sequence


def normalize_token(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def fuzzy_match_column(
    query: str,
    columns: Sequence[str],
    cutoff: float = 0.65,
) -> Optional[str]:
    """Find the best matching column name from a spoken or typed query.

    Matches in this order:
    1. Exact match (case-insensitive)
    2. Normalized alphanumeric match
    3. Token containment (query in column or column in query)
    4. Fuzzy string distance (difflib with cutoff threshold)

    Returns:
        The matched column name, or None if no confident match.
    """
    if not query or len(columns) == 0:
        return None

    cleaned_query = query.strip()
    lowered_query = cleaned_query.lower()

    # 1. Exact or case-insensitive
    for col in columns:
        if col.lower() == lowered_query:
            return col

    # 2. Normalized alphanumeric match
    norm_q = normalize_token(cleaned_query)
    for col in columns:
        if normalize_token(col) == norm_q:
            return col

    # 3. Substring / word boundary containment
    # E.g., query "sepal width" matches "sepal width (cm)"
    candidates = []
    for col in columns:
        col_low = col.lower()
        if lowered_query in col_low or col_low in lowered_query:
            candidates.append(col)
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Pick the candidate closest in length
        candidates.sort(key=lambda c: abs(len(c) - len(cleaned_query)))
        return candidates[0]

    # 4. Fuzzy ratio
    matches = difflib.get_close_matches(cleaned_query, columns, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    return None
