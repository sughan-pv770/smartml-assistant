import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import smartml as sml
from smartml.core.feature_engineer import FeatureEngineer
from smartml.core.feature_selector import FeatureSelector
from smartml.core.anomaly import AnomalyDetector
from smartml.core.preprocessor import Preprocessor
from smartml.core.trainer import Trainer

def main():
    print("--- SmartML Advanced Programmatic Pipeline ---\n")
    
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "is_cancer"})
    target = "is_cancer"
    
    # 1. Anomaly Detection (Unsupervised)
    print("1. Running Anomaly Detection...")
    detector = AnomalyDetector(algorithm="isolation_forest", contamination=0.05)
    anomaly_res = detector.detect(df.drop(columns=[target]))
    print(f"   {anomaly_res.narrative}")
    
    # Filter out anomalies for training
    clean_df = df[anomaly_res.labels == 1].reset_index(drop=True)
    print(f"   Remaining rows for training: {len(clean_df)} (dropped {len(df) - len(clean_df)})\n")
    
    # 2. Preprocessing
    print("2. Preprocessing...")
    pre = Preprocessor()
    prep_res = pre.clean(clean_df, target=target)
    for step in prep_res.steps:
        print(f"   - {step}")
    print()
    
    # 3. Feature Engineering
    print("3. Feature Engineering...")
    eng = FeatureEngineer(poly_degree=2, max_ratio_pairs=5)
    eng_res = eng.fit_transform(prep_res.df, target=target)
    print(f"   Added {len(eng_res.new_features)} features.")
    for step in eng_res.steps:
        print(f"   - {step}")
    print()
        
    # 4. Feature Selection
    print("4. Feature Selection...")
    X = eng_res.df.drop(columns=[target])
    y = eng_res.df[target]
    
    sel = FeatureSelector(correlation_threshold=0.9, model_based=True)
    sel_res = sel.select(X, y, task="classification")
    print(f"   Kept {sel_res.n_selected}/{sel_res.n_original} features.")
    for step in sel_res.steps:
        print(f"   - {step}")
    print()
        
    # 5. Model Training
    print("5. Training Models...")
    X_final = X[sel_res.selected_features]
    
    trainer = Trainer(task="classification", cv=3)
    train_res = trainer.run(X_final, y)
    
    print(f"   Winner: {train_res.best_name}")
    print(f"   Reason: {train_res.best_reason}")
    print("\nLeaderboard:")
    for e in train_res.leaderboard[:3]:
        print(f"   - {e.name}: CV F1={e.cv_score:.4f} (Accuracy={e.test_metrics.get('accuracy', 0):.4f})")

if __name__ == "__main__":
    main()
