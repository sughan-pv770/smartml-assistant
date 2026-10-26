import numpy as np
import pandas as pd
import pytest

from smartml.core.forecaster import TabularForecaster
from smartml.core.reducer import PCAReducer
from smartml.core.stats import compute_correlations, summarize_column
from smartml.core.tuner import ModelTuner


def test_stats_summary():
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    summary = summarize_column(df, "val")
    assert summary.mean == 55.0
    assert summary.median == 55.0
    assert summary.min == 10.0
    assert summary.max == 100.0
    assert summary.iqr == 45.0
    assert summary.skewness is not None


def test_stats_correlations():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "z": [5, 4, 3, 2, 1],
    })
    r_xy = compute_correlations(df, "x", "y")
    assert np.isclose(r_xy, 1.0)
    r_xz = compute_correlations(df, "x", "z")
    assert np.isclose(r_xz, -1.0)


def test_pca_reducer():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8],
        "f2": [2, 4, 6, 8, 10, 12, 14, 16],
        "f3": [3, 1, 4, 1, 5, 9, 2, 6],
    })
    reducer = PCAReducer(n_components=2)
    res = reducer.fit_transform(df)
    assert res.n_components == 2
    assert "PCA_1" in res.df_transformed.columns
    assert "PCA_2" in res.df_transformed.columns
    assert res.total_explained_variance > 0.8


def test_forecaster():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": [10 + i * 2 + (i % 2) for i in range(20)],
    })
    fc = TabularForecaster(lags=(1, 2), rolling_windows=(3,))
    res = fc.forecast(df, target_col="sales", date_col="date", horizon=3)
    assert len(res.predictions) == 3
    assert res.model_rmse >= 0.0


def test_model_tuner():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,
        "b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 3,
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3)

    tuner = ModelTuner(task="classification", n_iter=2, cv=2)
    res = tuner.tune("Logistic Regression", X, y, baseline_score=0.5)
    assert res.model_name == "Logistic Regression"
    assert res.best_score >= 0.0
