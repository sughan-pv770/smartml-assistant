"""
Example: using the SmartML conversational Assistant.

Run with:  python examples/example_usage.py
"""

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
import smartml as sml


def main():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "species"})

    # --- Conversational API -------------------------------------------------
    assistant = sml.Assistant(df, target="species", output_dir="smartml_output")

    assistant.ask("Analyze the dataset")
    assistant.ask("Clean my dataset")
    assistant.ask("Train the best classification model")
    assistant.ask("Why did you choose this algorithm?")
    assistant.ask("Show feature importance")
    assistant.ask("Show me the confusion matrix")
    assistant.ask("Generate a report")
    assistant.ask("Save the model")

    print("\n--- Full conversation history ---")
    for turn in assistant.history:
        print(f"[{turn['role']}] {turn['text'][:120]}")

    # --- Equivalent programmatic API (no natural language needed) ----------
    print("\n--- Doing the same thing without the chat layer ---")
    from smartml.core import preprocessor, trainer, explainer

    cleaned = preprocessor.Preprocessor().clean(df, target="species")
    X = cleaned.df.drop(columns=["species"])
    y = df.loc[cleaned.df.index, "species"]

    result = trainer.Trainer(task="classification").run(X, y)
    print(f"Best model: {result.best_name}")
    print(result.best_reason)

    importance = explainer.explain_feature_importance(
        result.best_model, result.X_test, result.y_test, result.feature_names, task="classification"
    )
    print(importance.narrative)


if __name__ == "__main__":
    main()
