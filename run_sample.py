from sklearn.datasets import load_wine
import pandas as pd
from smartml.assistant import Assistant
import os
import sys

# Force UTF-8 encoding for Windows console (to print emojis without crashing)
sys.stdout.reconfigure(encoding='utf-8')

def main():
    # Load dataset
    print("Loading Wine dataset...")
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target_class'] = data.target

    # Initialize assistant
    assistant = Assistant(df, target="target_class")

    # Run steps
    print("\n--- Chatting with Assistant ---")
    print("\n[user] Analyze the dataset")
    print(f"[assistant] {assistant.ask('Analyze the dataset')}")
    
    print("\n[user] Clean my dataset and do feature engineering")
    print(f"[assistant] {assistant.ask('Clean my dataset and do feature engineering')}")
    
    print("\n[user] Train the best classification model")
    print(f"[assistant] {assistant.ask('Train the best classification model')}")
    
    print("\n[user] Why did you choose this algorithm?")
    print(f"[assistant] {assistant.ask('Why did you choose this algorithm?')}")
    
    print("\n[user] Show feature importance")
    print(f"[assistant] {assistant.ask('Show feature importance')}")
    
    print("\n[user] Generate a report")
    print(f"[assistant] {assistant.ask('Generate a report')}")

    print("\nScript finished successfully! Check smartml_output/report.html for the report.")

if __name__ == "__main__":
    main()
