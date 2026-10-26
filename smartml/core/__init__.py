"""
smartml.core
=============
The analytical engine of SmartML.  Import the individual submodules directly
for programmatic use, or reach them via :class:`smartml.Assistant` for the
conversational interface.

Submodules
----------
analyzer
    Dataset profiling, type detection, task inference, data quality scoring.
preprocessor
    Automatic cleaning: deduplication, imputation, encoding, scaling.
feature_engineer
    Automatic feature generation (polynomial, date, frequency, interaction).
feature_selector
    Automatic feature selection (variance, correlation, MI, RFE, model-based).
trainer
    Multi-model comparison with cross-validation leaderboard.
explainer
    Feature importance (native or permutation) with narrative explanation.
visualizer
    Matplotlib-based plot generation (PNG, headless-safe).
reporter
    HTML / Markdown report assembly.
persistence
    Model bundle save / load (joblib).
loaders
    Smart data ingestion from CSV, Excel, JSON, Parquet, Feather, SQL.
clustering
    Unsupervised clustering (KMeans, DBSCAN, Agglomerative, GMM).
anomaly
    Anomaly / novelty detection (IsolationForest, LOF, OneClassSVM).
"""

from . import (
    analyzer,
    preprocessor,
    feature_engineer,
    feature_selector,
    trainer,
    explainer,
    visualizer,
    reporter,
    persistence,
    loaders,
    clustering,
    anomaly,
    wrangler,
    stats,
    reducer,
    tuner,
    forecaster,
    textutil,
    semi_supervised,
    reinforcement,
    self_supervised,
)

__all__ = [
    "analyzer",
    "preprocessor",
    "feature_engineer",
    "feature_selector",
    "trainer",
    "explainer",
    "visualizer",
    "reporter",
    "persistence",
    "loaders",
    "clustering",
    "anomaly",
    "wrangler",
    "stats",
    "reducer",
    "tuner",
    "forecaster",
    "textutil",
    "semi_supervised",
    "reinforcement",
    "self_supervised",
]
