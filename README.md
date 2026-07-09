# SmartML

**SmartML** is a production-grade, enterprise-level Machine Learning Automation Library for Python. It combines data preprocessing, feature engineering, feature selection, AutoML model training, explainable AI, visualization, and a conversational Assistant interface into a single unified framework.

## Installation

```bash
pip install smartml-assistant
```

To install with optional boosting algorithms (XGBoost, LightGBM, CatBoost) and visualization libraries (Plotly, Seaborn):

```bash
pip install "smartml-assistant[all]"
```

## Quick Start

### The One-Line Auto-Pilot
Provide your DataFrame and the target column, and SmartML will do the rest: profiling, cleaning, feature engineering, model comparison, training, and evaluation.

```python
import smartml as sml

# Load data (auto-detects CSV, Parquet, JSON, Excel, etc.)
df = sml.load("data/house_prices.csv")

# Run the complete AutoML pipeline
result = sml.auto(df, target="SalePrice", output_dir="my_project")

# Explain the best model
print(result.explain())

# Make predictions on new data
preds = result.predict(new_df)

# Save the model
result.save("my_model")

# Generate an HTML report
result.report(fmt="html")
```

### Unsupervised Clustering
```python
df = sml.load("data/customer_segments.csv")
result = sml.clustering_pipeline(df, algorithm="auto")
print(f"Found {result.clustering_result.n_clusters} clusters!")
```

### Conversational Assistant
Interact with your data using natural language:

```python
import smartml as sml

assistant = sml.Assistant(df, target="SalePrice")

assistant.ask("Clean my dataset")
assistant.ask("Engineer some new features")
assistant.ask("Train the best regression model")
assistant.ask("Why did you choose this algorithm?")
assistant.ask("Show feature importance")
assistant.ask("Generate a report")
```

## Features

- **Automated Data Cleaning:** Imputation, deduplication, low-cardinality one-hot encoding, and scaling.
- **Advanced Feature Engineering:** Polynomial interactions, date decomposition, frequency encoding, and pairwise ratios.
- **Smart Feature Selection:** Variance thresholding, correlation filtering, univariate selection, and model-based feature importance thresholding.
- **Multi-Model Leaderboard:** Automatically cross-validates Logistic/Linear Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM, CatBoost, and more.
- **Unsupervised Learning:** Full clustering (KMeans, DBSCAN, GMM, Agglomerative) and Anomaly Detection (IsolationForest, LOF, OneClassSVM).
- **Explainable AI:** Natural-language justifications for algorithm selection and feature importance.
- **Safe & Headless:** Safe for interactive environments. No global state side-effects.

## License
MIT License. See `LICENSE` for more information.
