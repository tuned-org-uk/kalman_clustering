use crate::centroids::KalmanCentroid;
use burn::tensor::backend::Backend;
use log::{debug, info, trace, warn};

/// Generic trait for distance-based graph construction
/// Allows plugging different statistical distance metrics
pub trait StatisticalDistance<B: Backend>: Send + Sync {
    /// Compute distance between two centroids
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32;

    /// Convert distance to edge weight
    fn distance_to_weight(&self, distance: f32) -> f32;

    /// Name of the distance metric (for logging)
    fn name(&self) -> &'static str;
}

/// Bhattacharyya distance for Gaussian centroids
pub struct BhattacharyyaDistance;

impl<B: Backend> StatisticalDistance<B> for BhattacharyyaDistance {
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32 {
        let dim = a.mean.dims()[0];

        trace!(
            "Computing Bhattacharyya distance: centroid_a.count={}, centroid_b.count={}, dim={}",
            a.count, b.count, dim
        );

        // Use the built-in method from KalmanCentroid
        let distance = a.bhattacharyya_distance(b);

        // Additional validation for logging
        if !distance.is_finite() {
            let mean_a = a.mean_to_vec();
            let mean_b = b.mean_to_vec();
            let var_a = a.variance_to_vec();
            let var_b = b.variance_to_vec();

            warn!(
                "Non-finite Bhattacharyya distance: {:.6}, checking dimensions...",
                distance
            );

            // Check individual dimensions
            for i in 0..dim.min(5) {
                // Log first 5 dims
                warn!(
                    "  dim {}: mean_diff={:.6}, var_a={:.6}, var_b={:.6}",
                    i,
                    mean_a[i] - mean_b[i],
                    var_a[i],
                    var_b[i]
                );
            }
        }

        trace!("Bhattacharyya distance computed: {:.6}", distance);
        distance
    }

    fn distance_to_weight(&self, distance: f32) -> f32 {
        let weight = (-distance).exp(); // Exponential kernel
        trace!(
            "Bhattacharyya distance {:.6} → weight {:.6}",
            distance, weight
        );
        weight
    }

    fn name(&self) -> &'static str {
        "Bhattacharyya"
    }
}

/// Kullback-Leibler divergence (asymmetric)
pub struct KLDivergence;

impl<B: Backend> StatisticalDistance<B> for KLDivergence {
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32 {
        let dim = a.mean.dims()[0];

        trace!(
            "Computing KL divergence: centroid_a.count={}, centroid_b.count={}, dim={}",
            a.count, b.count, dim
        );

        // Extract data for computation
        let mean_a = a.mean_to_vec();
        let mean_b = b.mean_to_vec();
        let var_a = a.variance_to_vec();
        let var_b = b.variance_to_vec();

        let mut kl: f32 = 0.0;
        for i in 0..dim {
            let var_ratio = var_a[i] / var_b[i].max(1e-9);
            let mean_diff = mean_b[i] - mean_a[i];

            // KL(a || b) for diagonal Gaussians
            // KL = 0.5 * (σ²_a/σ²_b + (μ_b - μ_a)²/σ²_b - 1 + ln(σ²_b/σ²_a))
            let term = 0.5
                * (var_ratio + (mean_diff * mean_diff) / var_b[i].max(1e-9) - 1.0
                    + (var_b[i] / var_a[i].max(1e-9)).ln());

            kl += term as f32;

            // Warn on numerical instabilities
            if !term.is_finite() {
                warn!(
                    "Non-finite KL term at dim {}: term={:.6}, var_ratio={:.6}, \
                     var_a={:.6}, var_b={:.6}, mean_diff={:.6}",
                    i, term, var_ratio, var_a[i], var_b[i], mean_diff
                );
            }
        }

        if !kl.is_finite() {
            warn!("Non-finite KL divergence computed: {:.6}", kl);
        }

        trace!("KL divergence computed: {:.6}", kl);
        kl
    }

    fn distance_to_weight(&self, distance: f32) -> f32 {
        let weight = 1.0 / (1.0 + distance); // Rational kernel
        trace!("KL divergence {:.6} → weight {:.6}", distance, weight);
        weight
    }

    fn name(&self) -> &'static str {
        "KL-Divergence"
    }
}

/// Symmetrized KL divergence (Jensen-Shannon divergence approximation)
pub struct SymmetrizedKL;

impl<B: Backend> StatisticalDistance<B> for SymmetrizedKL {
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32 {
        let kl_metric = KLDivergence;
        let kl_ab = kl_metric.distance(a, b);
        let kl_ba = kl_metric.distance(b, a);

        let symmetric_kl = (kl_ab + kl_ba) / 2.0;

        trace!(
            "Symmetrized KL: KL(a||b)={:.6}, KL(b||a)={:.6}, avg={:.6}",
            kl_ab, kl_ba, symmetric_kl
        );

        symmetric_kl
    }

    fn distance_to_weight(&self, distance: f32) -> f32 {
        let weight = (-distance).exp(); // Exponential kernel (like Bhattacharyya)
        trace!("Symmetrized KL {:.6} → weight {:.6}", distance, weight);
        weight
    }

    fn name(&self) -> &'static str {
        "Symmetrized-KL"
    }
}

/// Euclidean distance between centroid means (ignores variance)
pub struct EuclideanDistance;

impl<B: Backend> StatisticalDistance<B> for EuclideanDistance {
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32 {
        use burn::tensor::Tensor;

        // Compute ||mean_a - mean_b||₂
        let diff = a.mean.clone().sub(b.mean.clone());
        let dist_sq = diff.clone().mul(diff).sum();

        let dist_sq_val = dist_sq
            .into_data()
            .to_vec::<f32>()
            .expect("Failed to convert distance")[0];

        let distance = dist_sq_val.sqrt();

        trace!(
            "Euclidean distance computed: {:.6} (squared: {:.6})",
            distance, dist_sq_val
        );

        distance
    }

    fn distance_to_weight(&self, distance: f32) -> f32 {
        // Gaussian kernel with bandwidth σ=1.0
        let weight = (-0.5 * distance * distance).exp();
        trace!("Euclidean distance {:.6} → weight {:.6}", distance, weight);
        weight
    }

    fn name(&self) -> &'static str {
        "Euclidean"
    }
}

/// Mahalanobis-inspired distance: uses mean of both variances for normalization
pub struct MahalanobisLikeDistance;

impl<B: Backend> StatisticalDistance<B> for MahalanobisLikeDistance {
    fn distance(&self, a: &KalmanCentroid<B>, b: &KalmanCentroid<B>) -> f32 {
        let mean_a = a.mean_to_vec();
        let mean_b = b.mean_to_vec();
        let var_a = a.variance_to_vec();
        let var_b = b.variance_to_vec();

        let mut sum = 0.0;
        for i in 0..mean_a.len() {
            let mean_diff = mean_a[i] - mean_b[i];
            let avg_var = ((var_a[i] + var_b[i]) / 2.0).max(1e-9);

            sum += (mean_diff * mean_diff) / avg_var;
        }

        let distance = sum.sqrt() as f32;

        trace!(
            "Mahalanobis-like distance computed: {:.6} (squared: {:.6})",
            distance, sum
        );

        distance
    }

    fn distance_to_weight(&self, distance: f32) -> f32 {
        let weight = (-distance).exp();
        trace!(
            "Mahalanobis-like distance {:.6} → weight {:.6}",
            distance, weight
        );
        weight
    }

    fn name(&self) -> &'static str {
        "Mahalanobis-Like"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AutoBackend;
    type TestBackend = AutoBackend;

    // #[test]
    // fn test_bhattacharyya_distance() {
    //     let device = Default::default();

    //     let c1 = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.01, 0.1, &device);

    //     let c2 = KalmanCentroid::<TestBackend>::from_vec(&[1.0, 1.0], 0.01, 0.1, &device);

    //     let distance = BhattacharyyaDistance::distance(&c1, &c2);

    //     assert!(distance > 0.0);
    //     assert!(distance.is_finite());

    //     let weight: f64 = BhattacharyyaDistance::distance_to_weight(distance);
    //     assert!(weight > 0.0 && weight <= 1.0);
    // }

    #[test]
    fn test_kl_divergence() {
        let device = Default::default();

        let c1 = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.01, 0.1, &device);

        let c2 = KalmanCentroid::<TestBackend>::from_vec(&[1.0, 1.0], 0.01, 0.1, &device);

        let metric = KLDivergence;
        let distance = metric.distance(&c1, &c2);

        assert!(distance >= 0.0); // KL is non-negative
        assert!(distance.is_finite());
    }

    #[test]
    fn test_euclidean_distance() {
        let device = Default::default();

        let c1 = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.01, 0.1, &device);

        let c2 = KalmanCentroid::<TestBackend>::from_vec(&[3.0, 4.0], 0.01, 0.1, &device);

        let metric = EuclideanDistance;
        let distance = metric.distance(&c1, &c2);

        // Should be 5.0 (3-4-5 triangle)
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetrized_kl() {
        let device = Default::default();

        let c1 = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.01, 0.1, &device);

        let c2 = KalmanCentroid::<TestBackend>::from_vec(&[1.0, 1.0], 0.01, 0.1, &device);

        let metric = SymmetrizedKL;
        let dist_ab = metric.distance(&c1, &c2);
        let dist_ba = metric.distance(&c2, &c1);

        // Should be symmetric
        assert!((dist_ab - dist_ba).abs() < 1e-9);
    }
}
