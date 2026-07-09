import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

@pytest.fixture
def classification_df():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "species"})
    df.loc[0:3, "sepal length (cm)"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # inject a duplicate
    return df

@pytest.fixture
def regression_df():
    data = load_diabetes(as_frame=True)
    df = data.frame.rename(columns={"target": "progression"})
    df.loc[0:3, "bmi"] = np.nan
    return df
