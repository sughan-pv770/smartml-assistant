"""
smartml.core.persistence
==========================
Model save/load with lightweight metadata (so a loaded model still knows
what feature columns and preprocessing it expects).
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib


@dataclass
class SavedBundle:
    """Serialisable container for a trained model plus its full pipeline context.

    Attributes:
        model: The fitted scikit-learn compatible estimator.
        feature_names: Ordered list of feature columns the model expects.
        target: Name of the target column.
        task: ``"classification"`` or ``"regression"``.
        encoders: Dict of encoding info produced by the preprocessor.
        scaler: Fitted StandardScaler (or None if not applied).
        saved_at: ISO-8601 timestamp string of when the model was saved.
        model_name: Human-readable name of the winning algorithm.
        impute_values: Dict mapping column names to fill values used during
            training imputation, so prediction-time NaNs are handled
            identically.
    """

    model: Any
    feature_names: list[str]
    target: str
    task: str
    encoders: dict
    scaler: Any
    saved_at: str
    model_name: str
    # NOTE: field(default_factory=dict) avoids the mutable-default-argument
    # antipattern that caused SavedBundle() to raise TypeError in Python 3.11+.
    impute_values: dict = field(default_factory=dict)


def save_model(
    path: str,
    model: Any,
    feature_names: list[str],
    target: str,
    task: str,
    encoders: dict,
    scaler: Any,
    model_name: str,
    impute_values: Optional[dict] = None,
) -> str:
    """Persist a trained model bundle to disk.

    Args:
        path: Destination file path. ``.joblib`` extension is appended if absent.
        model: Fitted estimator to save.
        feature_names: Ordered list of feature column names.
        target: Name of the target column.
        task: ``"classification"`` or ``"regression"``.
        encoders: Encoding metadata from the preprocessor.
        scaler: Fitted StandardScaler or None.
        model_name: Human-readable algorithm name.
        impute_values: Dict of column → fill-value used during training.

    Returns:
        The absolute path of the saved ``.joblib`` file.
    """
    if not path.endswith(".joblib"):
        path += ".joblib"
    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    bundle = SavedBundle(
        model=model,
        feature_names=feature_names,
        target=target,
        task=task,
        encoders=encoders,
        scaler=scaler,
        saved_at=dt.datetime.now().isoformat(timespec="seconds"),
        model_name=model_name,
        impute_values=impute_values or {},
    )
    joblib.dump(bundle, path)
    return path


def load_model(path: str) -> SavedBundle:
    """Load a previously saved model bundle from disk.

    Args:
        path: Path to the ``.joblib`` file. The extension is added
            automatically if absent.

    Returns:
        The de-serialised :class:`SavedBundle`.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
    """
    if not os.path.exists(path) and os.path.exists(path + ".joblib"):
        path += ".joblib"
    bundle: SavedBundle = joblib.load(path)
    return bundle
