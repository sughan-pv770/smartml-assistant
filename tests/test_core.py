import os
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

from smartml.core import analyzer, preprocessor, trainer as trainer_mod, explainer, visualizer, reporter, persistence




# ---------------------------- analyzer ---------------------------- #

def test_detect_task_classification(classification_df):
    task, reason = analyzer.detect_task(classification_df, "species")
    assert task == "classification"
    assert "species" in reason


def test_detect_task_regression(regression_df):
    task, reason = analyzer.detect_task(regression_df, "progression")
    assert task == "regression"


def test_analyze_reports_missing_and_duplicates(classification_df):
    report = analyzer.analyze(classification_df, target="species")
    assert report.total_missing == 5  # 4 original NaNs + the duplicated row (a copy of row 0, which is NaN)
    # iris itself already contains one naturally duplicated row pair (rows 101/142),
    # plus the one we injected ourselves = 2 total.
    assert report.duplicate_rows == 2
    assert report.task == "classification"
    assert report.n_rows == len(classification_df)


# ---------------------------- preprocessor ---------------------------- #

def test_clean_removes_duplicates_and_imputes(classification_df):
    pre = preprocessor.Preprocessor()
    result = pre.clean(classification_df, target="species")
    assert result.df.duplicated().sum() == 0
    assert result.df.drop(columns=[c for c in ["species"] if c in result.df.columns]).isna().sum().sum() == 0
    assert any("duplicate" in s.lower() for s in result.steps)
    assert any("imputed" in s.lower() for s in result.steps)


def test_clean_one_hot_encodes_low_cardinality_categorical():
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "blue", "red"] * 4,
        "value": range(20),
        "target": [0, 1] * 10,
    })
    pre = preprocessor.Preprocessor()
    result = pre.clean(df, target="target")
    assert any(c.startswith("color_") for c in result.df.columns)
    assert "color" not in result.df.columns


def test_transform_new_aligns_columns(classification_df):
    pre = preprocessor.Preprocessor()
    result = pre.clean(classification_df, target="species")
    new_data = classification_df.drop(columns=["species"]).iloc[:3]
    transformed = preprocessor.Preprocessor.transform_new(new_data, result)
    expected_cols = [c for c in result.feature_names if c != "species"]
    assert list(transformed.columns) == expected_cols
    assert len(transformed) == 3


# ---------------------------- trainer ---------------------------- #

def test_trainer_classification(classification_df):
    pre = preprocessor.Preprocessor()
    cleaned = pre.clean(classification_df, target="species")
    X = cleaned.df.drop(columns=["species"])
    y = classification_df.loc[cleaned.df.index, "species"]

    tr = trainer_mod.Trainer(task="classification", cv=3)
    result = tr.run(X, y)

    assert result.task == "classification"
    assert len(result.leaderboard) >= 3
    assert result.best_model is not None
    assert "accuracy" in result.leaderboard[0].test_metrics
    assert result.best_reason  # non-empty explanation
    # leaderboard should be sorted descending by cv_score
    scores = [e.cv_score for e in result.leaderboard]
    assert scores == sorted(scores, reverse=True)


def test_trainer_regression(regression_df):
    pre = preprocessor.Preprocessor()
    cleaned = pre.clean(regression_df, target="progression")
    X = cleaned.df.drop(columns=["progression"]) if "progression" in cleaned.df.columns else cleaned.df
    y = regression_df.loc[cleaned.df.index, "progression"]

    tr = trainer_mod.Trainer(task="regression", cv=3)
    result = tr.run(X, y)

    assert result.task == "regression"
    assert "r2" in result.leaderboard[0].test_metrics


# ---------------------------- explainer ---------------------------- #

def test_explain_feature_importance(classification_df):
    pre = preprocessor.Preprocessor()
    cleaned = pre.clean(classification_df, target="species")
    X = cleaned.df.drop(columns=["species"])
    y = classification_df.loc[cleaned.df.index, "species"]
    tr = trainer_mod.Trainer(task="classification", cv=3)
    result = tr.run(X, y)

    imp = explainer.explain_feature_importance(
        result.best_model, result.X_test, result.y_test, result.feature_names, task="classification"
    )
    assert len(imp.importances) == len(result.feature_names)
    assert imp.narrative


# ---------------------------- visualizer / reporter / persistence ---------------------------- #

def test_visualizer_and_reporter_and_persistence(tmp_path, classification_df):
    out_dir = str(tmp_path / "out")
    pre = preprocessor.Preprocessor()
    cleaned = pre.clean(classification_df, target="species")
    X = cleaned.df.drop(columns=["species"])
    y = classification_df.loc[cleaned.df.index, "species"]
    tr = trainer_mod.Trainer(task="classification", cv=3)
    result = tr.run(X, y)

    heatmap_path = visualizer.plot_correlation_heatmap(classification_df, out_dir)
    assert os.path.exists(heatmap_path)

    cm_path = visualizer.plot_confusion_matrix(result.y_test, result.best_model.predict(result.X_test), out_dir)
    assert os.path.exists(cm_path)

    comp_path = visualizer.plot_model_comparison(result.leaderboard, out_dir)
    assert os.path.exists(comp_path)

    # persistence round trip
    save_path = persistence.save_model(
        path=str(tmp_path / "model"),
        model=result.best_model,
        feature_names=result.feature_names,
        target="species",
        task="classification",
        encoders=cleaned.encoders,
        scaler=cleaned.scaler,
        model_name=result.best_name,
    )
    assert os.path.exists(save_path)
    bundle = persistence.load_model(save_path)
    assert bundle.model_name == result.best_name
    preds = bundle.model.predict(result.X_test)
    assert len(preds) == len(result.y_test)
