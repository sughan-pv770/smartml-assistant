"""
smartml.core.clustering
=========================
Unsupervised clustering support for tabular datasets.

Supported algorithms
--------------------
* **KMeans** — standard Euclidean centroid-based clustering; the number of
  clusters is auto-selected via silhouette score when *k* is not specified.
* **MiniBatchKMeans** — same as KMeans but much faster on large datasets.
* **DBSCAN** — density-based, discovers arbitrary shapes and automatically
  identifies noise points; eps is tuned on a small grid.
* **Agglomerative** — hierarchical, bottom-up linkage; works well when the
  number of clusters is known.
* **GaussianMixture** — probabilistic, soft assignments; selects the number
  of components via BIC when *k* is not specified.

Metrics
-------
* Silhouette score (higher is better, range −1 … 1)
* Davies-Bouldin index (lower is better)
* Calinski-Harabasz score (higher is better)
* Inertia / log-likelihood (algorithm-specific)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """Output of a :class:`ClusteringTrainer` run.

    Attributes:
        algorithm: Name of the clustering algorithm used.
        n_clusters: Number of distinct clusters found (excludes noise).
        labels: Cluster label per row (−1 = noise for DBSCAN).
        silhouette: Silhouette score (NaN if fewer than 2 clusters).
        davies_bouldin: Davies-Bouldin index (NaN if not applicable).
        calinski_harabasz: Calinski-Harabasz score (NaN if not applicable).
        model: The fitted estimator.
        narrative: Plain-English summary of the clustering result.
        steps: Audit log of decisions made during fitting.
    """

    algorithm: str
    n_clusters: int
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    model: Any
    narrative: str
    steps: list[str] = field(default_factory=list)


class ClusteringTrainer:
    """Fit and evaluate unsupervised clustering algorithms.

    Args:
        algorithm: One of ``"kmeans"``, ``"minibatch_kmeans"``,
            ``"dbscan"``, ``"agglomerative"``, ``"gmm"``,
            or ``"auto"`` (tries KMeans with auto-selected k).
        n_clusters: Number of clusters. When None, the trainer tries
            a range and picks the best silhouette score (KMeans / GMM).
        k_range: Tuple ``(min_k, max_k)`` for automatic k selection.
        random_state: Random seed.
        scale: If True, standardise features before clustering.
    """

    def __init__(
        self,
        algorithm: str = "auto",
        n_clusters: Optional[int] = None,
        k_range: tuple[int, int] = (2, 10),
        random_state: int = 42,
        scale: bool = True,
    ):
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.random_state = random_state
        self.scale = scale

    # ------------------------------------------------------------------ #

    def run(self, X: pd.DataFrame) -> ClusteringResult:
        """Fit the clustering model and evaluate it.

        Args:
            X: Numeric feature matrix (non-numeric columns are dropped with
               a warning).

        Returns:
            :class:`ClusteringResult` with labels, metrics, and a narrative.
        """
        steps: list[str] = []

        # Keep only numeric columns
        X_num = X.select_dtypes(include=[np.number]).copy()
        dropped = [c for c in X.columns if c not in X_num.columns]
        if dropped:
            steps.append(
                f"Dropped {len(dropped)} non-numeric column(s) before clustering: "
                f"{dropped}."
            )

        X_num = X_num.fillna(X_num.median())

        if self.scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_num)
            steps.append("Standardised all features (zero mean, unit variance) before clustering.")
        else:
            X_scaled = X_num.values

        algo = self.algorithm
        if algo in ("auto", "kmeans"):
            return self._run_kmeans(X_scaled, X_num, steps, mini=False)
        if algo == "minibatch_kmeans":
            return self._run_kmeans(X_scaled, X_num, steps, mini=True)
        if algo == "dbscan":
            return self._run_dbscan(X_scaled, steps)
        if algo == "agglomerative":
            return self._run_agglomerative(X_scaled, steps)
        if algo == "gmm":
            return self._run_gmm(X_scaled, steps)
        raise ValueError(
            f"Unknown clustering algorithm '{self.algorithm}'. "
            "Choose from: auto, kmeans, minibatch_kmeans, dbscan, agglomerative, gmm."
        )

    # ------------------------------------------------------------------ #
    # Algorithm runners
    # ------------------------------------------------------------------ #

    def _auto_k(self, X: np.ndarray, steps: list[str]) -> int:
        """Select the best number of clusters via silhouette score."""
        lo, hi = self.k_range
        hi = min(hi, len(X) // 2)
        best_k, best_sil = lo, -1.0
        for k in range(lo, hi + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X, labels, sample_size=min(2000, len(X)))
            if sil > best_sil:
                best_sil, best_k = sil, k
        steps.append(
            f"Auto-selected k={best_k} clusters (best silhouette={best_sil:.3f} "
            f"over k={lo}…{hi})."
        )
        return best_k

    def _metrics(self, X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
        """Compute silhouette, Davies-Bouldin, Calinski-Harabasz."""
        unique_labels = set(labels) - {-1}  # exclude DBSCAN noise
        if len(unique_labels) < 2:
            return float("nan"), float("nan"), float("nan")
        mask = labels != -1
        X_clean, labels_clean = X[mask], labels[mask]
        try:
            sil = float(silhouette_score(X_clean, labels_clean,
                                          sample_size=min(2000, len(X_clean))))
        except Exception:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(X_clean, labels_clean))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(X_clean, labels_clean))
        except Exception:
            ch = float("nan")
        return sil, db, ch

    @staticmethod
    def _narrative(algorithm: str, n_clusters: int, sil: float, n_rows: int) -> str:
        text = (
            f"{algorithm} identified {n_clusters} cluster(s) in the dataset "
            f"({n_rows} rows)."
        )
        if not np.isnan(sil):
            if sil > 0.7:
                quality = "excellent (well-separated, tight clusters)"
            elif sil > 0.5:
                quality = "good"
            elif sil > 0.25:
                quality = "fair (some overlap between clusters)"
            else:
                quality = "poor (clusters overlap significantly)"
            text += f" Silhouette score = {sil:.3f} — {quality}."
        return text

    def _run_kmeans(
        self,
        X: np.ndarray,
        X_orig: pd.DataFrame,
        steps: list[str],
        mini: bool,
    ) -> ClusteringResult:
        k = self.n_clusters or self._auto_k(X, steps)
        alg_name = "MiniBatchKMeans" if mini else "KMeans"
        Cls = MiniBatchKMeans if mini else KMeans
        model = Cls(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = model.fit_predict(X)
        sil, db, ch = self._metrics(X, labels)
        steps.append(f"Fitted {alg_name}(k={k}).")
        return ClusteringResult(
            algorithm=alg_name,
            n_clusters=k,
            labels=labels,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            model=model,
            narrative=self._narrative(alg_name, k, sil, len(X_orig)),
            steps=steps,
        )

    def _run_dbscan(
        self, X: np.ndarray, steps: list[str]
    ) -> ClusteringResult:
        # Simple eps grid search (0.3, 0.5, 0.8, 1.0, 1.5)
        best_labels = np.full(len(X), -1)
        best_sil = -2.0
        best_eps = 0.5
        for eps in (0.3, 0.5, 0.8, 1.0, 1.5):
            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(X)
            unique = set(labels) - {-1}
            if len(unique) < 2:
                continue
            sil = silhouette_score(
                X[labels != -1], labels[labels != -1],
                sample_size=min(2000, (labels != -1).sum()),
            )
            if sil > best_sil:
                best_sil, best_labels, best_eps = sil, labels, eps

        model = DBSCAN(eps=best_eps, min_samples=5).fit(X)
        labels = best_labels
        n_clusters = len(set(labels) - {-1})
        sil, db, ch = self._metrics(X, labels)
        steps.append(
            f"Fitted DBSCAN(eps={best_eps}, min_samples=5); "
            f"found {n_clusters} cluster(s) and "
            f"{int((labels == -1).sum())} noise point(s)."
        )
        return ClusteringResult(
            algorithm="DBSCAN",
            n_clusters=n_clusters,
            labels=labels,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            model=model,
            narrative=self._narrative("DBSCAN", n_clusters, sil, len(X)),
            steps=steps,
        )

    def _run_agglomerative(
        self, X: np.ndarray, steps: list[str]
    ) -> ClusteringResult:
        k = self.n_clusters or self._auto_k(X, steps)
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        sil, db, ch = self._metrics(X, labels)
        steps.append(f"Fitted AgglomerativeClustering(n_clusters={k}).")
        return ClusteringResult(
            algorithm="AgglomerativeClustering",
            n_clusters=k,
            labels=labels,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            model=model,
            narrative=self._narrative("Agglomerative clustering", k, sil, len(X)),
            steps=steps,
        )

    def _run_gmm(
        self, X: np.ndarray, steps: list[str]
    ) -> ClusteringResult:
        if self.n_clusters is not None:
            best_k = self.n_clusters
        else:
            lo, hi = self.k_range
            hi = min(hi, len(X) // 2)
            best_k, best_bic = lo, np.inf
            for k in range(lo, hi + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gmm = GaussianMixture(
                        n_components=k, random_state=self.random_state
                    )
                    gmm.fit(X)
                    bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic, best_k = bic, k
            steps.append(
                f"Auto-selected k={best_k} components for GMM via BIC "
                f"(best BIC={best_bic:.1f})."
            )

        model = GaussianMixture(
            n_components=best_k, random_state=self.random_state
        )
        model.fit(X)
        labels = model.predict(X)
        sil, db, ch = self._metrics(X, labels)
        steps.append(f"Fitted GaussianMixture(n_components={best_k}).")
        return ClusteringResult(
            algorithm="GaussianMixture",
            n_clusters=best_k,
            labels=labels,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            model=model,
            narrative=self._narrative("Gaussian Mixture Model", best_k, sil, len(X)),
            steps=steps,
        )
