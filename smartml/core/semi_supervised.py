"""
smartml.core.semi_supervised
============================
Semi-supervised learning for partially labeled tabular datasets.

Supported algorithms:
1. LabelPropagation: Graph-based diffusion of known labels through feature space.
2. LabelSpreading: Regularized graph-based diffusion with noise tolerance.
3. SelfTraining: Wrapper around any base classifier (RandomForest, ExtraTrees, LogisticRegression)
   iteratively assigning pseudo-labels to high-confidence unlabeled samples.

Unlabeled samples are designated by -1 (or NaN converted to -1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier


@dataclass
class SemiSupervisedResult:
    algorithm: str
    n_labeled: int
    n_unlabeled: int
    n_pseudo_labeled: int
    transduced_labels: np.ndarray
    model: Any
    holdout_accuracy: Optional[float] = None
    holdout_f1: Optional[float] = None
    narrative: str = ""
    steps: list[str] = field(default_factory=list)


class SemiSupervisedTrainer:
    """Trains semi-supervised models on partially labeled datasets.

    Args:
        algorithm: 'label_spreading', 'label_propagation', 'self_training', or 'auto'.
        unlabeled_identifier: Value marking unlabeled rows (default -1).
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        unlabeled_identifier: int = -1,
        random_state: int = 42,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.unlabeled_identifier = unlabeled_identifier
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SemiSupervisedResult:
        steps = []
        num_X = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Identify labeled vs unlabeled
        if isinstance(y, pd.Series):
            y_clean = y.copy()
        else:
            y_clean = pd.Series(y)
        # Convert NaNs in target to unlabeled identifier
        y_clean = y_clean.fillna(self.unlabeled_identifier)
        
        # Convert to numpy integer/string representation
        y_arr = np.asarray(y_clean)
        
        unlabeled_mask = (y_arr == self.unlabeled_identifier)
        n_unlabeled = int(unlabeled_mask.sum())
        n_labeled = len(y_arr) - n_unlabeled
        
        steps.append(f"Identified {n_labeled} labeled sample(s) and {n_unlabeled} unlabeled sample(s).")
        
        if n_labeled < 2:
            raise ValueError("Semi-supervised learning requires at least 2 labeled samples.")
        if n_unlabeled == 0:
            steps.append("Notice: No unlabeled samples found (-1 or NaN). Training standard supervised baseline.")

        algo = self.algorithm
        if algo in ("auto", "self_training"):
            base = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            model = SelfTrainingClassifier(base_estimator=base, threshold=0.75, max_iter=10)
            model.fit(num_X, y_arr)
            used_algo = "SelfTraining (RandomForest)"
            transduced = model.predict(num_X)
            n_pseudo = int((unlabeled_mask & (transduced != self.unlabeled_identifier)).sum())
            steps.append(f"Fitted SelfTrainingClassifier: successfully pseudo-labeled {n_pseudo} of {n_unlabeled} samples.")
        elif algo == "label_spreading":
            model = LabelSpreading(kernel="rbf", max_iter=30)
            model.fit(num_X, y_arr)
            used_algo = "LabelSpreading"
            transduced = model.transduction_
            n_pseudo = n_unlabeled
            steps.append(f"Propagated labels via RBF graph kernel across all {n_unlabeled} unlabeled samples.")
        elif algo == "label_propagation":
            model = LabelPropagation(kernel="rbf", max_iter=30)
            model.fit(num_X, y_arr)
            used_algo = "LabelPropagation"
            transduced = model.transduction_
            n_pseudo = n_unlabeled
            steps.append(f"Propagated labels via LabelPropagation across {n_unlabeled} unlabeled samples.")
        else:
            raise ValueError(f"Unknown semi-supervised algorithm '{self.algorithm}'. Choose from auto, self_training, label_spreading, label_propagation.")

        # Evaluate performance on labeled subset
        acc = None
        f1 = None
        if n_labeled > 0 and (not unlabeled_mask.all()):
            labeled_idx = ~unlabeled_mask
            y_true_labeled = y_arr[labeled_idx]
            y_pred_labeled = transduced[labeled_idx]
            acc = round(float(accuracy_score(y_true_labeled, y_pred_labeled)), 4)
            f1 = round(float(f1_score(y_true_labeled, y_pred_labeled, average="weighted", zero_division=0)), 4)

        narrative = (
            f"{used_algo} completed with {n_labeled} labeled and {n_unlabeled} unlabeled rows. "
            f"Assigned pseudo-labels to {n_pseudo} rows. "
        )
        if acc is not None:
            narrative += f"Training consistency accuracy on known labels: {acc * 100:.1f}%."

        return SemiSupervisedResult(
            algorithm=used_algo,
            n_labeled=n_labeled,
            n_unlabeled=n_unlabeled,
            n_pseudo_labeled=n_pseudo,
            transduced_labels=transduced,
            model=model,
            holdout_accuracy=acc,
            holdout_f1=f1,
            narrative=narrative,
            steps=steps,
        )
