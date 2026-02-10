# Kalman Centroids: Adaptive Manifold Clustering

**A single-pass, statistical clustering engine for Graph Wiring that builds topology-aware Graph Laplacians.**

![Status](https://img.shields.io/badge/Status-Experimental-yellow)
![Performance](https://img.shields.io/badge/Complexity-O(N)-green)

## Overview

`kalman_centroids` replaces traditional multi-trial K-Means (Lloyd's algorithm) with a **Kalman Filter-based adaptive streaming algorithm**. Instead of treating centroids as static points in geometric space, this library models them as **Gaussian distributions** (Mean $\mu$ + Variance $\Sigma$) that evolve over time.

This approach solves two critical bottlenecks in the manifold pipeline:
1.  **Speed**: Eliminates the expensive "guess and check" `compute_optimal_k` heuristic.
2.  **Topology**: Constructs the Graph Laplacian using **Statistical Overlap** (Bhattacharyya Distance) rather than simple Euclidean proximity, preserving the true manifold structure.

## Quick Example
```rust
use kalman_centroids::KalmanClusterer;

fn main() {
    let rows: Vec<Vec<f64>> = load_embeddings(); 
    let max_k = 256; // Upper bound on number of clusters
    let n_items = rows.len();

    // Initialize adaptive clusterer
    let mut clusterer = KalmanClusterer::new(max_k, n_items);

    // Single-pass clustering with automatic K selection
    clusterer.fit(&rows); 
    
    println!("Auto-selected {} centroids", clusterer.centroids.len());
    
    // Access results
    for (i, centroid) in clusterer.centroids.iter().enumerate() {
        println!("Centroid {}: mean={:?}, var={:?}, count={}", 
                 i, centroid.mean, centroid.variance, centroid.count);
    }
}
```

## Key Features

- **🚀 Single-Pass Learning**: Adapts centroids dynamically as data streams in. No need to load the entire dataset into memory at once.
- **🧠 Automatic K-Selection**: Dynamically spawns new centroids when data points fall outside the statistical confidence interval (Mahalanobis distance) of existing clusters.
- **🔗 Statistical Wiring**: Connects the Graph Laplacian based on probability distribution overlap, ensuring spectral diffusion flows through data densities naturally.
- **📉 Variance Tracking**: Each centroid tracks its own uncertainty. High-variance clusters naturally bond with neighbors, while low-variance (sharp) clusters remain distinct.


## The Algorithm

### 1. Adaptive Updates (The "Brain")

Every centroid is a Kalman Filter. When a data point is assigned to a centroid, we perform a **Measurement Update**:

1. **Predict**: Increase variance slightly (Process Noise $Q$) to allow for drift.
2. **Gain**: Compute Kalman Gain based on current uncertainty.
3. **Correct**: Move mean $\mu$ towards data; shrink variance $\Sigma$.

\$ \mu_{new} = \mu_{old} + K(x - \mu_{old}) \$
\$ \Sigma_{new} = (1 - K)\Sigma_{old} \$

### 2. Decision Logic (The "Manager")

For each incoming point $x$:

1. Compute **Mahalanobis Distance** to all centroids:
\$ D_M(x) = \sqrt{ \sum \frac{(x_i - \mu_i)^2}{\sigma_i^2} } \$
2. **Split**: If $D_M > \text{Threshold}$ (typically $3\sigma$), the point is an outlier. Spawn a **new centroid**.
3. **Merge**: If $D_M \le \text{Threshold}$, update the nearest centroid's state.

### 3. Graph Wiring (The "Topology")

Instead of Euclidean distance, we build the Graph Laplacian using **Bhattacharyya Distance**. This connects centroids that *statistically overlap*, not just those that are close.

\$ D_B = \frac{1}{4} \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2} + \frac{1}{2} \ln \left( \frac{\sigma_1^2 + \sigma_2^2}{2 \sqrt{\sigma_1^2 \sigma_2^2}} \right) \$

**Result**: The Laplacian $L$ encodes the manifold invariant:
> **Manifold = Laplacian(Transpose(Centroids))**

## Performance vs. K-Means

| Feature | Standard K-Means | Kalman Centroids |
| :-- | :-- | :-- |
| **Complexity** | $O(N \times K \times \text{Iterations})$ | $O(N \times K)$ |
| **K-Selection** | Slow heuristic (Calinski-Harabasz) | **Automatic \& Dynamic** |
| **Metric** | Euclidean Only | **Mahalanobis (Adaptive)** |
| **Laplacian** | Geometric Proximity | **Statistical Overlap** |
| **Memory** | Batch (All Data) | **Streaming (Row by Row)** |
