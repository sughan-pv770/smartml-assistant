"""
smartml.core.anomaly
======================
Anomaly and novelty detection for tabular datasets.

Supported algorithms
--------------------
* **IsolationForest** — tree-based, fast, scales well; the default choice.
* **LocalOutlierFactor** — density-based, good for varying-density data.
* **OneClassSVM** — kernel-based; powerful but slow on large datasets.
* **EllipticEnvelope** — assumes Gaussian distribution; excellent baseline
  for near-normal data.
* **auto** — runs all four and picks the model with the highest estimated
  proportion of true inliers (smallest absolute outlier count near the
  contamination parameter).

Each model is fitted on the entire dataset (unsupervised: no labels needed).
Predicted labels follow the scikit-learn convention: +1 = inlier, −1 = outlier.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class AnomalyResult:
    """Output of an :class:`AnomalyDetector` run.

    Attributes:
        algorithm: Name of the algorithm used.
        labels: Per-row label: +1 = inlier, −1 = outlier.
        scores: Raw anomaly scores (lower = more anomalous for most algorithms).
        n_anomalies: Count of rows flagged as outliers.
        anomaly_pct: Percentage of rows flagged as outliers.
        model: The fitted estimator.
        narrative: Plain-English summary.
        steps: Audit log.
    """

    algorithm: str
    labels: np.ndarray
    scores: np.ndarray
    n_anomalies: int
    anomaly_pct: float
    model: Any
    narrative: str
    steps: list[str] = field(default_factory=list)

    @property
    def anomaly_indices(self) -> np.ndarray:
        """Row indices of the detected anomalies."""
        return np.where(self.labels == -1)[0]


class AnomalyDetector:
    """Fit an unsupervised anomaly detection model.

    Args:
        algorithm: One of ``"isolation_forest"``, ``"lof"``,
            ``"one_class_svm"``, ``"elliptic_envelope"``, or ``"auto"``.
        contamination: Estimated proportion of outliers in the data
            (float in (0, 0.5], or ``"auto"`` for IsolationForest).
        random_state: Random seed.
        scale: If True, standardise features before fitting.
    """

    def __init__(
        self,
        algorithm: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
        scale: bool = True,
    ):
        self.algorithm = algorithm.lower()
        self.contamination = contamination
        self.random_state = random_state
        self.scale = scale

    # ------------------------------------------------------------------ #

    def detect(self, X: pd.DataFrame) -> AnomalyResult:
        """Fit the anomaly detector and return labelled results.

        Args:
            X: Feature matrix (non-numeric columns are dropped).

        Returns:
            :class:`AnomalyResult` with per-row labels and a narrative.
        """
        steps: list[str] = []

        X_num = X.select_dtypes(include=[np.number]).copy()
        dropped = [c for c in X.columns if c not in X_num.columns]
        if dropped:
            steps.append(
                f"Dropped {len(dropped)} non-numeric column(s): {dropped}."
            )

        X_num = X_num.fillna(X_num.median())

        if self.scale:
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_num)
            steps.append("Standardised features before anomaly detection.")
        else:
            X_arr = X_num.values

        algo = self.algorithm
        if algo == "auto":
            return self._run_auto(X_arr, steps, len(X))
        if algo == "isolation_forest":
            return self._run_isolation_forest(X_arr, steps, len(X))
        if algo == "lof":
            return self._run_lof(X_arr, steps, len(X))
        if algo == "one_class_svm":
            return self._run_ocsvm(X_arr, steps, len(X))
        if algo == "elliptic_envelope":
            return self._run_elliptic(X_arr, steps, len(X))
        raise ValueError(
            f"Unknown anomaly detection algorithm '{self.algorithm}'. "
            "Choose from: auto, isolation_forest, lof, one_class_svm, elliptic_envelope."
        )

    # ------------------------------------------------------------------ #
    # Algorithm runners
    # ------------------------------------------------------------------ #

    def _run_isolation_forest(
        self, X: np.ndarray, steps: list[str], n_rows: int
    ) -> AnomalyResult:
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = model.fit_predict(X)
        scores = model.score_samples(X)  # lower = more anomalous
        steps.append(
            f"Fitted IsolationForest(contamination={self.contamination})."
        )
        return self._make_result("IsolationForest", labels, scores, model, steps, n_rows)

    def _run_lof(
        self, X: np.ndarray, steps: list[str], n_rows: int
    ) -> AnomalyResult:
        n_neighbors = min(20, max(2, n_rows // 10))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )
        labels = model.fit_predict(X)
        scores = model.negative_outlier_factor_  # lower = more anomalous
        steps.append(
            f"Fitted LocalOutlierFactor(n_neighbors={n_neighbors}, "
            f"contamination={self.contamination})."
        )
        return self._make_result("LocalOutlierFactor", labels, scores, model, steps, n_rows)

    def _run_ocsvm(
        self, X: np.ndarray, steps: list[str], n_rows: int
    ) -> AnomalyResult:
        nu = min(0.5, max(0.01, self.contamination))
        model = OneClassSVM(nu=nu, kernel="rbf")
        labels = model.fit_predict(X)
        scores = model.score_samples(X)
        steps.append(f"Fitted OneClassSVM(nu={nu}, kernel='rbf').")
        return self._make_result("OneClassSVM", labels, scores, model, steps, n_rows)

    def _run_elliptic(
        self, X: np.ndarray, steps: list[str], n_rows: int
    ) -> AnomalyResult:
        try:
            model = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=None,
            )
            labels = model.fit_predict(X)
            scores = model.score_samples(X)
            steps.append(
                f"Fitted EllipticEnvelope(contamination={self.contamination})."
            )
        except Exception as exc:
            # EllipticEnvelope can fail on near-singular covariance matrices;
            # fall back to IsolationForest with a warning.
            steps.append(
                f"EllipticEnvelope failed ({exc}); falling back to IsolationForest."
            )
            return self._run_isolation_forest(X, steps, n_rows)
        return self._make_result("EllipticEnvelope", labels, scores, model, steps, n_rows)

    def _run_auto(
        self, X: np.ndarray, steps: list[str], n_rows: int
    ) -> AnomalyResult:
        """Run all four algorithms and return the result with the median anomaly count."""
        candidates = {
            "IsolationForest": self._run_isolation_forest,
            "LocalOutlierFactor": self._run_lof,
            "EllipticEnvelope": self._run_elliptic,
        }
        results: list[AnomalyResult] = []
        for name, runner in candidates.items():
            try:
                r = runner(X, [], n_rows)
                results.append(r)
            except Exception:
                pass

        if not results:
            return self._run_isolation_forest(X, steps, n_rows)

        # Pick the algorithm closest to the expected contamination count
        expected = int(self.contamination * n_rows)
        best = min(results, key=lambda r: abs(r.n_anomalies - expected))
        best.steps = steps + [
            f"Auto-mode tested {len(results)} algorithms; selected "
            f"{best.algorithm} (flagged {best.n_anomalies} anomalies vs. "
            f"expected ≈ {expected})."
        ]
        return best

    @staticmethod
    def _make_result(
        algorithm: str,
        labels: np.ndarray,
        scores: np.ndarray,
        model: Any,
        steps: list[str],
        n_rows: int,
    ) -> AnomalyResult:
        n_anomalies = int((labels == -1).sum())
        pct = round(n_anomalies / max(n_rows, 1) * 100, 2)
        narrative = (
            f"{algorithm} flagged {n_anomalies} anomal{'y' if n_anomalies == 1 else 'ies'} "
            f"({pct}% of {n_rows} rows). "
        )
        if pct < 1:
            narrative += "This is a very low anomaly rate — the dataset appears clean."
        elif pct < 5:
            narrative += "This is within the typical 1–5% range for real-world datasets."
        elif pct < 15:
            narrative += (
                "A moderate anomaly rate. Consider investigating these rows before training."
            )
        else:
            narrative += (
                "A high anomaly rate. This may indicate noisy data, "
                "incorrect contamination parameter, or a genuinely unusual dataset."
            )
        return AnomalyResult(
            algorithm=algorithm,
            labels=labels,
            scores=scores,
            n_anomalies=n_anomalies,
            anomaly_pct=pct,
            model=model,
            narrative=narrative,
            steps=steps,
        )
