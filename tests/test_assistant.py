import os
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_diabetes

import smartml as sml




def test_greeting_mentions_shape_and_no_target(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, output_dir=str(tmp_path))
    greeting = assistant.history[0]["text"]
    assert "151 rows" in greeting
    assert "no target set" in greeting.lower()


def test_set_target_via_text(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, output_dir=str(tmp_path))
    reply = assistant.ask("set target to species")
    assert "species" in reply
    assert assistant.session.target == "species"


def test_analyze_intent(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    reply = assistant.ask("analyze the dataset")
    assert "rows" in reply.lower()


def test_clean_intent(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    reply = assistant.ask("Clean my dataset")
    assert "Imputed" in reply or "already clean" in reply
    assert assistant.session.cleaning_result is not None


def test_full_classification_conversation(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))

    clean_reply = assistant.ask("Clean my dataset")
    assert assistant.session.cleaning_result is not None

    train_reply = assistant.ask("Train the best classification model")
    assert assistant.session.training_result is not None
    assert assistant.session.training_result.best_name in train_reply

    why_reply = assistant.ask("Why did you choose this algorithm?")
    assert why_reply == assistant.session.training_result.best_reason

    fi_reply = assistant.ask("Show feature importance")
    assert assistant.session.importance_result is not None
    assert "1." in fi_reply

    viz_reply = assistant.ask("Show me the confusion matrix")
    assert "Confusion Matrix" in assistant.session.visualizations

    report_reply = assistant.ask("Generate a report")
    assert os.path.exists(assistant.session.report_paths["html"])

    save_reply = assistant.ask("Save the model")
    assert os.path.exists(os.path.join(str(tmp_path), "model.joblib"))

    # predict on a small slice written to CSV, passed via file_path
    sample = classification_df.drop(columns=["species"]).iloc[:5]
    csv_path = os.path.join(str(tmp_path), "sample.csv")
    sample.to_csv(csv_path, index=False)
    predict_reply = assistant.ask("Predict on this CSV file", file_path=csv_path)
    assert "Predicted 5 row" in predict_reply
    assert os.path.exists(os.path.join(str(tmp_path), "predictions.csv"))


def test_regression_conversation(regression_df, tmp_path):
    assistant = sml.Assistant(regression_df, target="progression", output_dir=str(tmp_path))
    assistant.ask("Clean my dataset")
    train_reply = assistant.ask("Train the best regression model")
    assert assistant.session.training_result.task == "regression"
    assert "r2" in str(assistant.session.training_result.leaderboard[0].test_metrics)

    viz_reply = assistant.ask("Show me the residual plot")
    assert "Residual Plot" in assistant.session.visualizations


def test_explain_before_training_is_honest(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    reply = assistant.ask("Why did you choose this algorithm?")
    assert "haven't trained" in reply.lower()


def test_predict_without_data_asks_for_it(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    assistant.ask("Clean my dataset")
    assistant.ask("Train the best model")
    reply = assistant.ask("Predict on new data")
    assert "need data" in reply.lower()


def test_unknown_intent_offers_help(classification_df, tmp_path):
    assistant = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    reply = assistant.ask("sing me a song")
    assert "not sure" in reply.lower()


def test_save_load_predict_roundtrip(classification_df, tmp_path):
    a1 = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    a1.ask("Clean my dataset")
    a1.ask("Train the best model")
    a1.ask("Save the model")

    # fresh assistant/session, loads the persisted model instead of training
    a2 = sml.Assistant(classification_df, target="species", output_dir=str(tmp_path))
    load_reply = a2.ask("Load the model", path=os.path.join(str(tmp_path), "model.joblib"))
    assert "Loaded" in load_reply

    sample = classification_df.drop(columns=["species"]).iloc[:4]
    csv_path = os.path.join(str(tmp_path), "sample2.csv")
    sample.to_csv(csv_path, index=False)
    predict_reply = a2.ask("Predict on this CSV file", file_path=csv_path)
    assert "Predicted 4 row" in predict_reply
