import pandas as pd
from sklearn.datasets import make_blobs
import smartml as sml

def main():
    print("--- SmartML Clustering Example ---\n")
    
    # 1. Generate some synthetic clustered data
    X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    
    # 2. Run the clustering pipeline
    # We let it auto-select the number of clusters (should find 4)
    print("Running auto-clustering...")
    result = sml.clustering_pipeline(df, algorithm="auto", output_dir="clustering_output")
    
    # 3. Print the results
    cr = result.clustering_result
    print(f"\nAlgorithm used: {cr.algorithm}")
    print(f"Number of clusters found: {cr.n_clusters}")
    print(f"Silhouette Score: {cr.silhouette:.3f}")
    print(f"Davies-Bouldin Index: {cr.davies_bouldin:.3f}")
    
    print("\nNarrative:")
    print(cr.narrative)
    
    print("\nSteps taken:")
    for step in cr.steps:
        print(f"  - {step}")

if __name__ == "__main__":
    main()
