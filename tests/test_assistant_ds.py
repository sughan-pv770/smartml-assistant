import os
import pandas as pd
import pytest

import smartml as sml


def test_assistant_ds_wrangling_and_stats(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))

    # 1. Stats intent
    res_desc = assistant.ask("describe column sepal width (cm)")
    assert "Summary for 'sepal width (cm)'" in res_desc
    assert "Mean:" in res_desc

    # 2. Skewness
    res_skew = assistant.ask("check skewness of petal length (cm)")
    assert "Skewness=" in res_skew

    # 3. Filtering
    res_filter = assistant.ask("filter rows where sepal width (cm) is greater than 3.0")
    assert "Filtered rows" in res_filter

    # 4. Undo
    res_undo = assistant.ask("undo last step")
    assert "Undid" in res_undo

    # 5. Correlation
    res_corr = assistant.ask("correlation between sepal width (cm) and petal width (cm)")
    assert "Pearson correlation between" in res_corr

    # 6. PCA
    res_pca = assistant.ask("reduce to 2 components")
    assert "PCA complete" in res_pca


def test_assistant_do_everything(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    reply = assistant.ask("do everything")
    assert "Executing full AutoML pipeline" in reply
    assert "[x] Profiled dataset" in reply
    assert "[x] Preprocessed and cleaned data" in reply
    assert "[x] Trained and compared models" in reply
    assert "[x] Computed feature importance" in reply
    assert "[x] Generated full HTML report" in reply
    assert "[x] Saved best model bundle" in reply
    assert os.path.exists(os.path.join(str(tmp_path), "report.html"))
    assert os.path.exists(os.path.join(str(tmp_path), "model.joblib"))
