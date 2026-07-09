import math
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from smartml.core.clustering import ClusteringTrainer

def test_kmeans_clustering():
    X = make_blobs(n_samples=100, centers=3, n_features=4, random_state=42)[0]
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    
    ct = ClusteringTrainer(algorithm="kmeans", n_clusters=3, random_state=42)
    result = ct.run(df)
    
    assert result.algorithm == "KMeans"
    assert result.n_clusters == 3
    assert not math.isnan(result.silhouette)
    assert len(result.labels) == 100

def test_auto_clustering():
    X = make_blobs(n_samples=100, centers=4, n_features=4, random_state=42)[0]
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    
    ct = ClusteringTrainer(algorithm="auto", random_state=42, k_range=(2, 6))
    result = ct.run(df)
    
    assert result.algorithm == "KMeans"
    assert result.n_clusters == 4
    assert not math.isnan(result.silhouette)

def test_dbscan_clustering():
    X = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)[0]
    df = pd.DataFrame(X, columns=["f1", "f2"])
    
    ct = ClusteringTrainer(algorithm="dbscan")
    result = ct.run(df)
    
    assert result.algorithm == "DBSCAN"
    assert result.n_clusters >= 1
