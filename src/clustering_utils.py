"""
clustering_utils.py — Reusable Clustering Pipeline Module

Functions for standardized K-Means clustering, K evaluation,
and PCA visualization.

Author: [Yun Deng]
Course: ECON 5200, Lab 22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Dict


def run_kmeans_pipeline(
    df: pd.DataFrame,
    features: List[str],
    k: int,
    random_state: int = 42
) -> Dict:
    """End-to-end K-Means pipeline.
    
    1. Extracts features from DataFrame
    2. Standardizes with StandardScaler
    3. Fits K-Means
    4. Returns labels, scaler, model, and silhouette score
    
    Args:
        df: DataFrame with feature columns
        features: List of column names to use
        k: Number of clusters
        random_state: Random seed for reproducibility
    
    Returns:
        dict with keys: 'labels', 'scaler', 'model', 'X_scaled',
                        'silhouette', 'inertia'
    """
    # YOUR IMPLEMENTATION HERE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    
    return {
        'labels': labels,
        'scaler': scaler,
        'model': kmeans,
        'X_scaled': X_scaled,
        'silhouette': sil,
        'inertia': kmeans.inertia_
    }

def evaluate_k_range(
    X: np.ndarray,
    k_range: range,
    random_state: int = 42
) -> pd.DataFrame:
    """Evaluate clustering quality across a range of K values.
    
    Computes WCSS (inertia) and silhouette score for each K.
    
    Args:
        X: Standardized feature matrix
        k_range: Range of K values to test (e.g., range(2, 11))
        random_state: Random seed
    
    Returns:
        DataFrame with columns: 'k', 'wcss', 'silhouette'
    """
    # YOUR IMPLEMENTATION HERE
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({'k': k, 'wcss': km.inertia_, 'silhouette': sil})
    return pd.DataFrame(results)



def plot_pca_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> None:
    """PCA 2D scatter plot with cluster coloring.
    
    Fits PCA(n_components=2), creates scatter plot colored by cluster,
    and annotates with explained variance ratios.
    
    Args:
        X: Standardized feature matrix
        labels: Cluster labels (array of integers)
        feature_names: List of original feature names
    """
    # YOUR IMPLEMENTATION HERE
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', alpha=0.7, s=40)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('K-Means Clusters — PCA Projection')
    plt.tight_layout()
    plt.show()


# --- Quick self-test ---
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X_test, _ = make_blobs(n_samples=200, centers=3, n_features=5, random_state=0)
    df_test = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(5)])
    
    result = run_kmeans_pipeline(df_test, [f'f{i}' for i in range(5)], k=3)
    print(f'Labels shape: {result["labels"].shape}')
    print(f'Silhouette: {result["silhouette"]:.4f}')
    
    eval_df = evaluate_k_range(result['X_scaled'], range(2, 8))
    print(eval_df)
    
    plot_pca_clusters(result['X_scaled'], result['labels'], [f'f{i}' for i in range(5)])
    print('Self-test passed.')
