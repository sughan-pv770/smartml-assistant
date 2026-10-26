import pandas as pd
import pytest

from smartml.core.wrangler import DataWrangler


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "city": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        "target": [0, 1, 0, 1, 1],
    })


def test_wrangler_drop_column(sample_df):
    w = DataWrangler(sample_df, target="target")
    msg = w.drop_column("city")
    assert "Dropped column 'city'" in msg
    assert "city" not in w.current_df.columns
    assert "target" in w.current_df.columns

    # Protect target column
    err = w.drop_column("target")
    assert "Cannot drop target column" in err


def test_wrangler_filter_rows(sample_df):
    w = DataWrangler(sample_df, target="target")
    msg = w.filter_rows("age", ">", 30)
    assert "kept 3 rows" in msg
    assert len(w.current_df) == 3

    # Refuse dropping 100% of rows
    refused = w.filter_rows("age", ">", 100)
    assert "refused" in refused.lower()
    assert len(w.current_df) == 3


def test_wrangler_undo(sample_df):
    w = DataWrangler(sample_df, target="target")
    orig_len = len(w.current_df)
    w.filter_rows("age", ">", 30)
    assert len(w.current_df) == 3
    undo_msg = w.undo()
    assert "Undid" in undo_msg
    assert len(w.current_df) == orig_len


def test_wrangler_replace_and_fill(sample_df):
    df_with_na = sample_df.copy()
    df_with_na.loc[0, "age"] = None
    w = DataWrangler(df_with_na, target="target")

    fill_msg = w.fill_na("age", 20)
    assert "Filled 1 missing" in fill_msg
    assert w.current_df.loc[0, "age"] == 20

    replace_msg = w.replace_value("city", "Paris", "Rome")
    assert "Replaced 1 occurrence" in replace_msg
    assert "Rome" in w.current_df["city"].values
