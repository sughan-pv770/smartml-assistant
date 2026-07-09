import os
import pandas as pd
import pytest
from smartml.core import loaders

def test_load_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = os.path.join(tmp_path, "test.csv")
    df.to_csv(path, index=False)
    loaded = loaders.load(path)
    pd.testing.assert_frame_equal(df, loaded)

def test_load_json(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = os.path.join(tmp_path, "test.json")
    df.to_json(path, orient="records")
    loaded = loaders.load(path)
    pd.testing.assert_frame_equal(df, loaded)

def test_load_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = os.path.join(tmp_path, "test.parquet")
    df.to_parquet(path)
    loaded = loaders.load(path)
    pd.testing.assert_frame_equal(df, loaded)

def test_load_unsupported(tmp_path):
    path = os.path.join(tmp_path, "test.unknown")
    with open(path, "w") as f:
        f.write("data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        loaders.load(path)
