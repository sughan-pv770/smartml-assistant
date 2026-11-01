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


def test_lazy_voice_imports_without_audio_libs():
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "for mod in ('sounddevice', 'pynput', 'faster_whisper', 'win32com', 'speech_recognition', 'psutil'):\n"
        "    sys.modules[mod] = None\n"
        "import smartml\n"
        "assert 'smartml.agent' not in sys.modules, 'smartml.agent eagerly imported'\n"
        "assert 'smartml.mial' not in sys.modules, 'smartml.mial eagerly imported'\n"
        "from smartml.core import analyzer\n"
        "import pandas as pd\n"
        "rep = analyzer.analyze(pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 0, 1]}), target='b')\n"
        "assert rep.data_quality_score >= 0\n"
        "cls = smartml.MialConfig\n"
        "assert cls().stt_engine == 'whisper'\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

