"""
SmartML — Production-grade AutoML engine with a conversational Assistant.

Quick start
-----------
**One-line auto-pilot:**

    import smartml as sml

    result = sml.auto(df, target="Price")
    result.predict(new_df)
    result.report()
    result.explain()

**Task-specific shortcuts:**

    sml.regression(df, target="SalePrice")
    sml.classification(df, target="Species")
    sml.clustering(df)

**Data loading:**

    df = sml.load("data/train.csv")
    df = sml.load("report.parquet")

**Conversational assistant:**

    assistant = sml.Assistant(df, target="Price")
    assistant.ask("Clean my dataset")
    assistant.ask("Train the best regression model")
    assistant.ask("Why did you choose this algorithm?")
    assistant.ask("Show feature importance")
    assistant.ask("Generate a report")
    assistant.ask("Predict on this CSV file", file_path="new_data.csv")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from .assistant import Assistant, Session
from .core import (
    analyzer,
    preprocessor,
    trainer as trainer_mod,
    explainer,
    visualizer,
    reporter,
    persistence,
    feature_engineer,
    feature_selector,
    loaders,
    clustering as clustering_mod,
    anomaly,
    semi_supervised as semi_supervised_mod,
    reinforcement as reinforcement_mod,
    self_supervised as self_supervised_mod,
    forecaster as forecaster_mod,
    reducer as reducer_mod,
    stats as stats_mod,
    wrangler as wrangler_mod,
)

__version__ = "2.0.4"
__author__ = "SmartML Contributors"
__license__ = "MIT"

# Set up a library-level logger that is silent by default (NullHandler).
# Callers can configure logging at the application level.
logging.getLogger("smartml").addHandler(logging.NullHandler())

__all__ = [
    # High-level API for all ML Paradigms
    "auto",
    "regression",
    "classification",
    "clustering",
    "clustering_pipeline",
    "semi_supervised",
    "reinforcement",
    "self_supervised",
    "forecast",
    "load",
    # Core classes
    "Assistant",
    "Session",
    "SmartMLResult",
    # Submodules
    "analyzer",
    "preprocessor",
    "feature_engineer",
    "feature_selector",
    "trainer_mod",
    "explainer",
    "visualizer",
    "reporter",
    "persistence",
    "loaders",
    "clustering_mod",
    "anomaly",
    "semi_supervised_mod",
    "reinforcement_mod",
    "self_supervised_mod",
    "forecaster_mod",
    "reducer_mod",
    "stats_mod",
    "wrangler_mod",
]


# =========================================================================== #
# SmartMLResult — unified result object returned by high-level functions
# =========================================================================== #

class SmartMLResult:
    """Encapsulates a completed AutoML pipeline run.

    Returned by :func:`auto`, :func:`regression`, :func:`classification`, and
    :func:`clustering_pipeline`.  Provides a consistent interface for
    prediction, persistence, reporting, and explanation.

    Attributes:
        task: ML task type (``"classification"``, ``"regression"``,
              ``"clustering"``, etc.).
        training_result: :class:`~smartml.core.trainer.TrainingResult` (None
            for clustering).
        cleaning_result: :class:`~smartml.core.preprocessor.CleaningResult`.
        dataset_report: :class:`~smartml.core.analyzer.DatasetReport`.
        clustering_result: :class:`~smartml.core.clustering.ClusteringResult`
            (None for supervised tasks).
        output_dir: Directory where reports and plots are written.
        target: Target column name (None for unsupervised tasks).
    """

    def __init__(
        self,
        task: str,
        training_result: Optional[Any] = None,
        cleaning_result: Optional[Any] = None,
        dataset_report: Optional[Any] = None,
        clustering_result: Optional[Any] = None,
        output_dir: str = "smartml_output",
        target: Optional[str] = None,
    ):
        self.task = task
        self.training_result = training_result
        self.cleaning_result = cleaning_result
        self.dataset_report = dataset_report
        self.clustering_result = clustering_result
        self.output_dir = output_dir
        self.target = target

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data.

        The same preprocessing pipeline (imputation, encoding, scaling)
        that was applied during training is applied here, ensuring
        consistent predictions on unseen data.

        Args:
            data: New feature DataFrame (must contain the same columns as
                the training data, minus the target column).

        Returns:
            Series of predictions indexed to match *data*.

        Raises:
            RuntimeError: If the result is from a clustering pipeline
                (use ``clustering_result.labels`` instead).
        """
        if self.training_result is None:
            raise RuntimeError(
                "No trained model available. For clustering results, use "
                ".clustering_result.labels instead."
            )
        assert self.cleaning_result is not None
        X_new = preprocessor.Preprocessor.transform_new(data, self.cleaning_result)
        preds = self.training_result.best_model.predict(X_new)
        return pd.Series(preds, index=data.index, name="prediction")

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, path: str = "smartml_model") -> str:
        """Save the trained model bundle to disk.

        Args:
            path: Destination path (without extension; ``.joblib`` is added).

        Returns:
            Full path of the saved file.

        Raises:
            RuntimeError: If no trained model is available.
        """
        if self.training_result is None:
            raise RuntimeError("No trained model to save.")
        tr = self.training_result
        return persistence.save_model(
            path=path,
            model=tr.best_model,
            feature_names=tr.feature_names,
            target=self.target or "",
            task=tr.task,
            encoders=self.cleaning_result.encoders if self.cleaning_result else {},
            scaler=self.cleaning_result.scaler if self.cleaning_result else None,
            model_name=tr.best_name,
            impute_values=self.cleaning_result.impute_values if self.cleaning_result else {},
        )

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    def explain(self) -> str:
        """Return a natural-language feature importance explanation.

        Returns:
            Explanation string from the explainer module.

        Raises:
            RuntimeError: If no trained model is available.
        """
        if self.training_result is None:
            raise RuntimeError("No trained model to explain.")
        tr = self.training_result
        imp = explainer.explain_feature_importance(
            model=tr.best_model,
            X_test=tr.X_test,
            y_test=tr.y_test,
            feature_names=tr.feature_names,
            task=tr.task,
        )
        return imp.narrative

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def report(self, fmt: str = "html") -> str:
        """Generate an HTML or Markdown report.

        Args:
            fmt: ``"html"`` or ``"markdown"`` (``"md"`` also accepted).

        Returns:
            Path to the generated report file.
        """
        import os

        # Build a minimal session-like object the reporter understands
        class _FakeSession:
            def __init__(self, result: "SmartMLResult"):
                self.dataset_report = result.dataset_report
                self.cleaning_result = result.cleaning_result
                self.training_result = result.training_result
                self.target = result.target
                self.visualizations: dict = {}
                self.importance_result = None

        session = _FakeSession(self)

        os.makedirs(self.output_dir, exist_ok=True)
        if fmt in ("markdown", "md"):
            path = os.path.join(self.output_dir, "report.md")
            return reporter.generate_markdown_report(session, path)
        path = os.path.join(self.output_dir, "report.html")
        return reporter.generate_html_report(session, path)

    # ------------------------------------------------------------------ #
    # String representation
    # ------------------------------------------------------------------ #

    @property
    def summary(self) -> str:
        """Executive summary and leaderboard of the completed run."""
        if self.training_result:
            tr = self.training_result
            lines = [
                f"SmartML {self.task.capitalize()} Result:",
                f"  Best Model: {tr.best_name}",
                f"  Summary: {tr.best_reason}",
                f"  Leaderboard:"
            ]
            for i, entry in enumerate(tr.leaderboard, 1):
                marker = " <- best" if i == 1 else ""
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in list(entry.test_metrics.items())[:3])
                lines.append(f"    {i}. {entry.name}: CV={entry.cv_score:.4f}+/-{entry.cv_std:.4f} | {metrics_str}{marker}")
            return "\n".join(lines)
        if self.clustering_result:
            cr = self.clustering_result
            return f"SmartML Clustering Result:\n  Algorithm: {cr.algorithm}\n  Clusters: {cr.n_clusters}\n  Silhouette: {cr.silhouette:.4f}"
        return f"SmartML Result (task={self.task!r})"

    def __repr__(self) -> str:  # noqa: D105
        if self.training_result:
            return (
                f"SmartMLResult(task={self.task!r}, "
                f"best_model={self.training_result.best_name!r}, "
                f"cv_score={self.training_result.leaderboard[0].cv_score:.4f})"
            )
        if self.clustering_result:
            return (
                f"SmartMLResult(task='clustering', "
                f"algorithm={self.clustering_result.algorithm!r}, "
                f"n_clusters={self.clustering_result.n_clusters})"
            )
        return f"SmartMLResult(task={self.task!r})"


# =========================================================================== #
# Top-level convenience functions
# =========================================================================== #

def auto(
    data: pd.DataFrame,
    target: str,
    output_dir: str = "smartml_output",
    task: Optional[str] = None,
) -> SmartMLResult:
    """Full AutoML pipeline: detect task → clean → train → return result.

    Args:
        data: Input DataFrame.
        target: Name of the target column.
        output_dir: Directory for reports and saved models.
        task: Force a specific task (``"classification"`` or
            ``"regression"``). Auto-detected if None.

    Returns:
        :class:`SmartMLResult` ready for ``.predict()``, ``.save()``,
        ``.explain()``, and ``.report()``.

    Example::

        import smartml as sml
        from sklearn.datasets import load_iris
        import pandas as pd

        df = load_iris(as_frame=True).frame
        result = sml.auto(df, target="target")
        print(result)
    """
    logger = logging.getLogger("smartml")
    logger.info("SmartML.auto: analysing dataset (%d rows, %d cols)", *data.shape)

    report = analyzer.analyze(data, target=target)
    detected_task = task or report.task or "classification"

    logger.info("Task: %s — %s", detected_task, report.task_reason)

    pre = preprocessor.Preprocessor()
    cleaning = pre.clean(data, target=target)
    logger.info("Cleaning: %d step(s) applied.", len(cleaning.steps))

    X = cleaning.df.drop(columns=[target], errors="ignore")
    y = cleaning.df[target]

    tr = trainer_mod.Trainer(task=detected_task)
    training = tr.run(X, y)
    logger.info(
        "Training complete. Best model: %s (CV=%.4f)",
        training.best_name,
        training.leaderboard[0].cv_score,
    )

    return SmartMLResult(
        task=detected_task,
        training_result=training,
        cleaning_result=cleaning,
        dataset_report=report,
        output_dir=output_dir,
        target=target,
    )


def regression(
    data: pd.DataFrame,
    target: str,
    output_dir: str = "smartml_output",
) -> SmartMLResult:
    """Run a forced regression pipeline.

    Equivalent to ``auto(data, target, task="regression")``.

    Args:
        data: Input DataFrame.
        target: Name of the continuous target column.
        output_dir: Directory for outputs.

    Returns:
        :class:`SmartMLResult`.
    """
    return auto(data, target=target, output_dir=output_dir, task="regression")


def classification(
    data: pd.DataFrame,
    target: str,
    output_dir: str = "smartml_output",
) -> SmartMLResult:
    """Run a forced classification pipeline.

    Equivalent to ``auto(data, target, task="classification")``.

    Args:
        data: Input DataFrame.
        target: Name of the categorical/label target column.
        output_dir: Directory for outputs.

    Returns:
        :class:`SmartMLResult`.
    """
    return auto(data, target=target, output_dir=output_dir, task="classification")


def clustering_pipeline(
    data: pd.DataFrame,
    algorithm: str = "auto",
    n_clusters: Optional[int] = None,
    output_dir: str = "smartml_output",
) -> SmartMLResult:
    """Run an unsupervised clustering pipeline.

    Args:
        data: Input DataFrame (all columns are used as features).
        algorithm: Clustering algorithm name (``"auto"``, ``"kmeans"``,
            ``"dbscan"``, ``"agglomerative"``, ``"gmm"``).
        n_clusters: Number of clusters. Auto-selected if None.
        output_dir: Directory for outputs.

    Returns:
        :class:`SmartMLResult` with a populated ``clustering_result``.
    """
    logger = logging.getLogger("smartml")
    logger.info("SmartML.clustering: dataset (%d rows, %d cols)", *data.shape)

    ct = clustering_mod.ClusteringTrainer(
        algorithm=algorithm, n_clusters=n_clusters
    )
    result = ct.run(data)
    logger.info(
        "Clustering complete: %s found %d clusters (silhouette=%.3f).",
        result.algorithm, result.n_clusters, result.silhouette,
    )

    return SmartMLResult(
        task="clustering",
        clustering_result=result,
        output_dir=output_dir,
    )


def load(path: str, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from a file (CSV, Excel, JSON, Parquet, Feather, …).

    This is a thin wrapper around :func:`smartml.core.loaders.load` exposed
    at the package level for convenience.

    Args:
        path: Path to the data file.
        **kwargs: Forwarded to the underlying pandas reader.

    Returns:
        Loaded DataFrame.

    Example::

        import smartml as sml

        df = sml.load("data/titanic.csv")
        df = sml.load("sales_2024.parquet")
    """
    return loaders.load(path, **kwargs)


def clustering(
    data: pd.DataFrame,
    algorithm: str = "auto",
    n_clusters: Optional[int] = None,
    output_dir: str = "smartml_output",
) -> SmartMLResult:
    """Run an unsupervised clustering pipeline (alias of clustering_pipeline)."""
    return clustering_pipeline(data, algorithm=algorithm, n_clusters=n_clusters, output_dir=output_dir)


def semi_supervised(
    data: pd.DataFrame,
    target: str,
    algorithm: str = "auto",
    unlabeled_identifier: int = -1,
    random_state: int = 42,
) -> semi_supervised_mod.SemiSupervisedResult:
    """Run semi-supervised learning on partially labeled tabular data.

    Args:
        data: Input DataFrame.
        target: Target column containing labeled entries and unlabeled marks (-1 or NaN).
        algorithm: 'self_training', 'label_spreading', 'label_propagation', or 'auto'.
        unlabeled_identifier: Marker for unlabeled rows (default -1).
        random_state: Seed for reproducibility.
    """
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' not found.")
    X = data.drop(columns=[target], errors="ignore")
    y = data[target]
    trainer = semi_supervised_mod.SemiSupervisedTrainer(
        algorithm=algorithm, unlabeled_identifier=unlabeled_identifier, random_state=random_state
    )
    return trainer.fit(X, y)


def reinforcement(
    data: pd.DataFrame,
    action: str,
    reward: str,
    context_cols: Optional[list[str]] = None,
    strategy: str = "auto",
    **kwargs: Any,
) -> reinforcement_mod.BanditResult:
    """Run offline tabular contextual bandit / reinforcement decision optimization.

    Args:
        data: Logged interaction dataset.
        action: Column indicating action/arm taken.
        reward: Column indicating observed outcome/reward.
        context_cols: Optional list of state/feature columns for contextual bandits (LinUCB).
        strategy: 'linucb', 'thompson', 'epsilon_greedy', or 'auto'.
    """
    opt = reinforcement_mod.TabularBanditOptimizer(strategy=strategy, **kwargs)
    return opt.optimize(data, action_col=action, reward_col=reward, context_cols=context_cols)


def self_supervised(
    data: pd.DataFrame,
    target: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    **kwargs: Any,
) -> self_supervised_mod.SelfSupervisedResult:
    """Extract dense tabular latent representations using Masked/Denoising Autoencoders.

    Args:
        data: Tabular dataset.
        target: Optional target column to isolate and exclude from representations.
        embedding_dim: Latent representation size (default: half of feature count, min 2).
    """
    learner = self_supervised_mod.TabularSelfSupervisedLearner(embedding_dim=embedding_dim, **kwargs)
    return learner.fit_transform(data, target_col=target)


def forecast(
    data: pd.DataFrame,
    target: str,
    date_col: Optional[str] = None,
    horizon: int = 5,
    **kwargs: Any,
) -> forecaster_mod.ForecastResult:
    """Run multi-step recursive time series forecasting with automatic lag features.

    Args:
        data: Time series or sequential DataFrame.
        target: Numeric column to forecast.
        date_col: Optional date/timestamp column. Auto-detected if omitted.
        horizon: Number of future periods/steps to forecast.
    """
    fc = forecaster_mod.TabularForecaster(**kwargs)
    return fc.forecast(data, target_col=target, date_col=date_col, horizon=horizon)

