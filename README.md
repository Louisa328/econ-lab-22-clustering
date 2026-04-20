# Lab 22: Unsupervised Learning — Clustering & Dimensionality Reduction

> **Course:** ECON 5200 · Causal Machine Learning & Applied Analytics  
> **Institution:** Northeastern University  
> **Author:** Yun Deng  

---

## Objective

Diagnose and repair a production-broken K-Means pipeline, then apply corrected unsupervised learning methods to cluster 250 World Bank economies and 2,000 synthetic fintech customers — benchmarking PCA against UMAP for dimensionality reduction and packaging the workflow into a reusable Python module.

---

## Repository Structure

```
lab-22-clustering/
├── notebooks/
│   └── lab_ch22_diagnostic_1.ipynb   # Main lab notebook
├── src/
│   └── clustering_utils.py           # Reusable pipeline module
├── figures/                          # Generated visualizations
└── README.md
```

---

## Methodology

### Part 1 — Pipeline Diagnosis

Identified four deliberate errors in a broken K-Means pipeline:

| # | Error Type | Description | Impact |
|---|-----------|-------------|--------|
| 1 | Preprocessing omission | K-Means applied to raw, unstandardized features | GDP per capita (range: $300–$120K) dominated all distance calculations; remaining 9 features contributed negligibly |
| 2 | API parameter error | `KMeans(k=4)` instead of `KMeans(n_clusters=4)` | Raises `TypeError`; common mistake when switching from R to Python |
| 3 | Method ordering error | PCA fitted on raw data, then standardized | PC1 captured >90% of variance — entirely proxying GDP scale, not true structure |
| 4 | Reproducibility error | `KMeans()` called without `random_state` | Results changed across runs due to random centroid initialization; non-reproducible research output |

### Part 2 — Corrected Pipeline

Built the canonical pipeline in correct order:

```
StandardScaler → KMeans(n_clusters=4, random_state=42) → PCA(n_components=2)
```

**Verification checkpoints passed:**
- Standardized feature means ≈ 0 ✓
- PC1 explained variance: **45.6%** (within expected 35–50% range) ✓
- Silhouette score: **0.2331** (within expected 0.15–0.40 range) ✓
- Cluster sizes: `[65, 17, 55, 113]` — no single dominant cluster ✓

### Part 3 — Customer Segmentation + PCA vs UMAP

Applied clustering to 2,000 synthetic fintech customers across 6 behavioral features (`avg_monthly_spend`, `txn_frequency`, `app_sessions`, `credit_utilization`, `products_held`, `digital_engagement`), mimicking Nubank-style customer archetypes.

- **K-Means (K=4) silhouette score: 0.2387**
- UMAP (`n_neighbors=15`, `min_dist=0.1`) produced tighter, more visually distinct cluster boundaries than PCA, reflecting its ability to capture nonlinear local neighborhood structure
- PCA projections showed overlapping boundaries; UMAP revealed cleaner segment separation consistent with the 4 latent data-generating centers

### Part 4 — `clustering_utils.py` Module

Packaged three reusable functions for production-grade unsupervised learning:

| Function | Description |
|----------|-------------|
| `run_kmeans_pipeline(df, features, k)` | End-to-end pipeline: standardize → fit KMeans → return labels, scaler, model, silhouette, inertia |
| `evaluate_k_range(X, k_range)` | Computes WCSS and silhouette score for K = 2–10; supports elbow + silhouette method selection |
| `plot_pca_clusters(X, labels, feature_names)` | PCA 2D scatter with cluster coloring and explained variance annotation |

Self-test passes on `make_blobs` synthetic data with 3 true clusters.

### Challenge — Hierarchical Clustering Comparison

Compared K-Means and Agglomerative (Ward linkage) clustering on the same WDI dataset:

| Method | Silhouette Score |
|--------|-----------------|
| K-Means (K=4) | **0.2331** |
| Agglomerative (K=4, Ward) | 0.2226 |

Cross-tabulation showed strong agreement between the two methods — the majority of countries were assigned to the same cluster, with minor divergence in mid-development economies. K-Means marginally outperformed Agglomerative by silhouette score, consistent with Ward linkage optimizing the same WCSS objective as K-Means.

---

## Key Findings

1. **Standardization is non-negotiable for K-Means.** Without `StandardScaler`, clustering on WDI data collapsed into GDP-only segmentation, ignoring health, education, trade, and digital access dimensions entirely.

2. **K = 4 is optimal** for both the WDI country dataset and the synthetic customer data, validated by silhouette score and consistent with known data structure.

3. **UMAP outperforms PCA for visual cluster separation** on behavioral customer data. PCA (linear) showed overlapping projections; UMAP (nonlinear, `n_neighbors=15`) produced clearly bounded segments.

4. **K-Means and Agglomerative clustering largely agree** on the WDI data (silhouette: 0.2331 vs 0.2226), suggesting the 4-cluster structure is robust across algorithmic assumptions.

5. **Reproducibility requires explicit `random_state`.** Without it, K-Means centroid initialization varies across runs, violating replication standards essential for academic and production settings.

---

## Tech Stack

`Python` · `scikit-learn` · `umap-learn` · `wbgapi` · `pandas` · `matplotlib` · `scipy`

---
