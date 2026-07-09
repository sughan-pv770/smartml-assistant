import os
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes
import smartml as sml

@pytest.fixture
def classification_data():
    data = load_iris(as_frame=True)
    return data.frame.rename(columns={"target": "species"})

@pytest.fixture
def regression_data():
    data = load_diabetes(as_frame=True)
    return data.frame.rename(columns={"target": "progression"})

def test_sml_auto_classification(classification_data, tmp_path):
    out_dir = str(tmp_path / "out")
    result = sml.auto(classification_data, target="species", output_dir=out_dir)
    assert result.task == "classification"
    assert result.training_result is not None
    assert result.dataset_report is not None
    assert result.cleaning_result is not None
    
    # test methods on result
    preds = result.predict(classification_data.drop(columns=["species"]))
    assert len(preds) == len(classification_data)
    
    explanation = result.explain()
    assert len(explanation) > 0
    
    model_path = result.save(os.path.join(out_dir, "my_model"))
    assert os.path.exists(model_path)
    
    report_path = result.report(fmt="html")
    assert os.path.exists(report_path)

def test_sml_clustering(classification_data, tmp_path):
    out_dir = str(tmp_path / "out")
    # drop target to simulate pure unsupervised
    df = classification_data.drop(columns=["species"])
    result = sml.clustering_pipeline(df, algorithm="kmeans", n_clusters=3, output_dir=out_dir)
    assert result.task == "clustering"
    assert result.clustering_result is not None
    assert result.clustering_result.n_clusters == 3
