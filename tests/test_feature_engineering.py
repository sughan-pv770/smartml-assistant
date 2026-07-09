import numpy as np
import pandas as pd
from smartml.core.feature_engineer import FeatureEngineer

def test_feature_engineering_date_decomposition():
    df = pd.DataFrame({
        "date_col": pd.date_range("2024-01-01", periods=10, freq="D"),
        "target": range(10)
    })
    eng = FeatureEngineer(poly_degree=0)
    result = eng.fit_transform(df, target="target")
    assert "date_col_year" in result.df.columns
    assert "date_col_dayofweek" in result.df.columns
    assert "date_col_is_weekend" in result.df.columns

def test_feature_engineering_frequency_encoding():
    df = pd.DataFrame({
        "cat_col": ["a", "a", "a", "b", "c"],
        "target": [0, 1, 0, 1, 0]
    })
    eng = FeatureEngineer(poly_degree=0)
    result = eng.fit_transform(df, target="target")
    assert "cat_col_freq" in result.df.columns
    # 'a' appears 3/5 times = 0.6
    assert np.isclose(result.df.loc[0, "cat_col_freq"], 0.6)

def test_feature_engineering_polynomial_and_ratios():
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0]
    })
    eng = FeatureEngineer(poly_degree=2, max_ratio_pairs=5)
    result = eng.fit_transform(df, target="target")
    assert "num1_x_num2" in result.df.columns
    assert "num1_div_num2" in result.df.columns
    assert np.isclose(result.df.loc[0, "num1_x_num2"], 5)
    assert np.isclose(result.df.loc[0, "num1_div_num2"], 0.2)

def test_transform_unseen_data():
    df_train = pd.DataFrame({
        "cat": ["a", "a", "b"],
        "num": [1, 2, 3]
    })
    eng = FeatureEngineer(poly_degree=2)
    result = eng.fit_transform(df_train)
    
    df_test = pd.DataFrame({
        "cat": ["a", "c"], # 'c' is unseen
        "num": [4, 5]
    })
    transformed = eng.transform(df_test, result)
    assert "cat_freq" in transformed.columns
    # 'a' was 2/3 in train, 'c' was unseen so 0.0
    assert np.isclose(transformed.loc[0, "cat_freq"], 2/3)
    assert np.isclose(transformed.loc[1, "cat_freq"], 0.0)
