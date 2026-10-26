# SmartML 🚀

[![PyPI version](https://img.shields.io/pypi/v/smartml-assistant.svg)](https://pypi.org/project/smartml-assistant/)
[![Python Versions](https://img.shields.io/pypi/pyversions/smartml-assistant.svg)](https://pypi.org/project/smartml-assistant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmartML** is a production-grade, enterprise-ready Multi-Paradigm AutoML and Data Science engine for Python.

It provides a unified, zero-boilerplate API spanning **Supervised Learning**, **Unsupervised Clustering**, **Semi-Supervised Learning**, **Contextual Reinforcement Learning**, **Self-Supervised Tabular Representation Learning**, **Recursive Time-Series Forecasting**, and an interactive conversational **Data Science Copilot**.

---

## 📦 Installation

```bash
pip install smartml-assistant
```

To install with full support for hardware-accelerated boosting (XGBoost, LightGBM, CatBoost), Bayesian tuning (Optuna), SHAP explanations, and interactive visualizations:

```bash
pip install "smartml-assistant[all]"
```

---

## ⚡ Quick Start: The 5 Paradigms in 1 Line

### 1. Ingestion (`sml.load`)
Auto-detects file format by extension (CSV, Excel, Parquet, JSON, Feather, ORC) and loads into a clean DataFrame:
```python
import smartml as sml

df = sml.load("dataset.csv")  # or .parquet, .xlsx, .json
```

---

### 2. Supervised Learning — AutoML (`sml.auto`)
Handles automated data cleaning, type inference, missing value imputation, categorical encoding, and cross-validates 9+ candidate algorithms (HistGradientBoosting, Gradient Boosting, Extra Trees, Random Forest, Logistic/Linear Regression, Ridge, SVM, Naive Bayes, KNN, XGBoost, LightGBM, CatBoost):

```python
import smartml as sml

# Train, tune, and evaluate all models in 1 line
result = sml.auto(df, target="Would_Recommend")

# 1. Print executive summary and candidate leaderboard
print(result.summary)

# 2. Plain-English explainable AI (feature importance breakdown)
print(result.explain())

# 3. Generate self-contained HTML report with charts and metrics
report_path = result.report(fmt="html")

# 4. Save model bundle to disk (.joblib with metadata & preprocessors)
result.save("my_production_model")

# 5. Make predictions on new incoming data
predictions = result.predict(new_df)
```

---

### 3. Unsupervised Learning — Auto-Clustering (`sml.clustering`)
Discovers natural groups, partitions, and density clusters without labels using K-Means, DBSCAN, or automatic silhouette-score optimization:

```python
# Cluster data and calculate Silhouette scores
cluster_res = sml.clustering(df, n_clusters=3)
print(cluster_res.summary)

# Access cluster assignments
df["Cluster_Label"] = cluster_res.clustering_result.labels
```

---

### 4. Semi-Supervised Learning (`sml.semi_supervised`)
When only a fraction of your data has labels and the rest are marked `-1` or `NaN`:

```python
# Pseudo-label unannotated records using Self-Training or Label Spreading
semi_res = sml.semi_supervised(df, target="Would_Recommend", algorithm="auto")

print("Newly pseudo-labeled rows:", semi_res.n_pseudo_labeled)
print("Holdout validation score:", semi_res.accuracy)
```

---

### 5. Reinforcement Learning — Decision Optimization (`sml.reinforcement`)
Applies offline tabular contextual bandits and Multi-Armed Bandit policies (**LinUCB**, **Thompson Sampling**, **Epsilon-Greedy**) to discover optimal policies from logged user interactions:

```python
# Select the best action based on user context and historical reward
bandit = sml.reinforcement(
    df,
    action="Primary_AI_Tool",
    reward="Overall_Satisfaction",
    context_cols=["User_Type", "Profession", "Monthly_Income"],
    strategy="auto"
)

print("Best action recommended:", bandit.best_action)
print("Estimated policy value:", bandit.expected_reward)
```

---

### 6. Self-Supervised Learning (`sml.self_supervised`)
Trains Denoising and Masked Autoencoders to learn dense, robust tabular representations for downstream tasks:

```python
# Extract low-dimensional latent embeddings
ssl_res = sml.self_supervised(df, target="Would_Recommend", embedding_dim=16)

# Dense representation matrix (shape: n_rows x 16)
dense_embeddings = ssl_res.embeddings
```

---

### 7. Recursive Time-Series Forecasting (`sml.forecast`)
Performs multi-step forecasting with automatic lag generation, rolling statistics, and trend decomposition:

```python
# Forecast future steps in 1 line
forecast_res = sml.forecast(df, target="Total_Hours_Per_Week", horizon=7)

print("Future forecast steps:")
print(forecast_res.predictions)
```

---

### 8. Conversational Data Science Copilot (`sml.Assistant`)
Interactive plain-English copilot with a **10-step safe undo/redo stack** for exploratory data analysis, data wrangling, and pipeline execution:

```python
ai = sml.Assistant(df, target="Would_Recommend")

# Natural language commands
print(ai.ask("show summary"))
print(ai.ask("remove outliers"))
print(ai.ask("undo"))
print(ai.ask("plot target distribution"))
print(ai.ask("do everything"))  # profiles, cleans, trains, explains, and builds HTML report
```

---

## 📊 Viewing the HTML Report in Jupyter / Google Colab

```python
from IPython.display import HTML, display

# Display the generated HTML report right inside your notebook
with open(result.report(fmt="html"), "r", encoding="utf-8") as f:
    display(HTML(f.read()))
```

---

## 🛠️ Complete Feature Matrix

| Paradigm / Feature | Algorithms / Capabilities |
| :--- | :--- |
| **Supervised AutoML** | HistGradientBoosting, Gradient Boosting, Extra Trees, Random Forest, Logistic/Linear Regression, Ridge, Huber, SVM, Naive Bayes, KNN, XGBoost, LightGBM, CatBoost |
| **Automated Preprocessing** | Deduplication, constant column pruning, median/mode imputation, datetime expansion, high-cardinality ordinal/one-hot encoding, standard scaling |
| **Unsupervised Learning** | K-Means, DBSCAN, Agglomerative, GMM, PCA dimensionality reduction, Isolation Forest anomaly detection |
| **Semi-Supervised** | Self-Training with confidence thresholding, Label Spreading, Label Propagation |
| **Reinforcement Learning** | Contextual LinUCB, Thompson Sampling, Epsilon-Greedy bandit policy evaluation |
| **Self-Supervised** | Denoising Autoencoder, Masked Autoencoder latent embeddings |
| **Time Series** | Recursive multi-step lag forecasting, rolling metrics, autoregression |
| **Data Science Assistant** | 10-step undo/redo state manager, IQR outlier handling, automated statistical profiling, interactive visualizations, HTML & Markdown report exporter |

---

## 📄 License
MIT License. Created by SmartML Contributors.
