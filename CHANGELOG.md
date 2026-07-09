# Changelog

All notable changes to SmartML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-30

### Added
- **Top-level API:** `sml.auto()`, `sml.regression()`, `sml.classification()`, `sml.clustering_pipeline()`, and `sml.load()`.
- **Feature Engineering Module:** Automatic generation of date, frequency, interaction, and ratio features.
- **Feature Selection Module:** Variance threshold, correlation filter, univariate `SelectKBest`, and RandomForest-based selection.
- **Clustering Module:** `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `AgglomerativeClustering`, and `GaussianMixture` with automatic `k` selection.
- **Anomaly Detection Module:** `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope`, and an `auto` mode.
- **Loaders Module:** Built-in support for loading CSV, TSV, Excel, JSON, Parquet, Feather, ORC, and SQL data.
- **More Models:** `XGBoost`, `LightGBM`, and `CatBoost` support (if installed). Plus `ExtraTrees`, `HistGradientBoosting`, `SVM`, `NaiveBayes`, and `KNN`.
- **More Visualizations:** ROC Curve, Precision-Recall Curve, Learning Curve, and Feature Distributions.
- **Robustness:** Added error handling for failed models during CV, returning the best successful one.

### Changed
- Replaced `print()` calls in the `Assistant` with proper `logging` for a production-ready library experience.
- Refactored `visualizer.py` to activate `matplotlib.use("Agg")` only inside plotting functions to prevent hijacking interactive environments (Jupyter, etc.).

### Fixed
- Fixed a Python 3.11+ `TypeError` in `SavedBundle` caused by a mutable dictionary default argument.
- Corrected test warnings by adding proper `pytest` config to `pyproject.toml`.

## [0.1.0] - 2026-06-25

### Added
- Initial release.
- Core engine for dataset profiling, auto-cleaning, model comparison, feature importance, and reporting.
- Conversational `Assistant` interface.
