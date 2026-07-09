import numpy as np
import pandas as pd
from smartml.core.feature_selector import FeatureSelector

def test_variance_threshold():
    df = pd.DataFrame({
        "const": [1, 1, 1, 1, 1],
        "varies": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0]
    })
    X = df.drop(columns=["target"])
    y = df["target"]
    sel = FeatureSelector(variance_threshold=0.1, correlation_threshold=1.0, k_best=None, model_based=False)
    result = sel.select(X, y)
    assert "const" not in result.selected_features
    assert "varies" in result.selected_features

def test_correlation_filter():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 4, 6, 8, 10], # perfectly correlated with feature1
        "independent": [1, 0, 1, 0, 1],
        "target": [1, 2, 3, 4, 5]
    })
    X = df.drop(columns=["target"])
    y = df["target"]
    sel = FeatureSelector(correlation_threshold=0.9, k_best=None, model_based=False)
    result = sel.select(X, y)
    assert "independent" in result.selected_features
    # One of feature1 or feature2 should be dropped
    assert ("feature1" in result.selected_features) != ("feature2" in result.selected_features)

def test_univariate_selection():
    df = pd.DataFrame({
        "good": [1, 2, 3, 4, 5],
        "bad": [0, 0, 1, 0, 0],
        "target": [1, 2, 3, 4, 5]
    })
    X = df.drop(columns=["target"])
    y = df["target"]
    sel = FeatureSelector(k_best=1, model_based=False, correlation_threshold=1.0, variance_threshold=0.0)
    result = sel.select(X, y, task="regression")
    assert "good" in result.selected_features
    assert "bad" not in result.selected_features
