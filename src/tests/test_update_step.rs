use crate::KalmanClusterer;
use crate::backend::AutoBackend;
use crate::centroids::KalmanCentroid;
use crate::init;
use burn::tensor::{Tensor, TensorData};

use log::{debug, info};

type TestBackend = AutoBackend;

// ========================================================================
// PART 1: Low-Level Kalman Filter Mechanics (The "Brain")
// ========================================================================

#[test]
fn test_centroid_update_reduces_variance() {
    init();
    info!("Testing Kalman variance reduction property");

    let device = Default::default();
    let initial_mean = vec![0.0f32, 0.0];
    // High initial uncertainty (variance = 10.0 from ::new)
    let mut centroid = KalmanCentroid::<TestBackend>::from_vec(&initial_mean, 0.1, 0.5, &device);

    let initial_var = centroid.variance_to_vec();
    assert!(
        (initial_var[0] - 10.0).abs() < 1e-3,
        "Initial variance should be high"
    );

    // Update with a point exactly at the mean (pure variance reduction test)
    // Ideally, if data matches our expectation, confidence should grow (variance drops)
    centroid.update_from_vec(&[0.0, 0.0]);

    let new_var = centroid.variance_to_vec();
    debug!("Variance update: {} -> {}", initial_var[0], new_var[0]);

    assert!(
        new_var[0] < initial_var[0],
        "Variance must decrease after observation"
    );

    // Confidence is inverse variance, so it should increase
    assert!(
        centroid.confidence > 0.1,
        "Confidence must increase as variance drops"
    );
}

#[test]
fn test_centroid_mean_drift() {
    init();
    info!("Testing Kalman mean adaptation (Drift)");

    let device = Default::default();
    let mut centroid = KalmanCentroid::<TestBackend>::from_vec(&[0.0], 1e-4, 1.0, &device);

    // Feed a series of points at 10.0
    // The mean should asymptotically approach 10.0, but not jump instantly
    for i in 1..=5 {
        centroid.update_from_vec(&[10.0]);
        let mean = centroid.mean_to_vec();
        let var = centroid.variance_to_vec();
        debug!("Step {}: Mean = {:.4}, Var = {:.4}", i, mean[0], var[0]);
    }

    let final_mean = centroid.mean_to_vec();
    assert!(
        final_mean[0] > 0.0 && final_mean[0] < 10.0,
        "Mean should move towards data but retain memory"
    );

    // Verify it moved significantly
    assert!(
        final_mean[0] > 5.0,
        "Mean should have adapted significantly after 5 steps"
    );
}

#[test]
fn test_process_noise_sets_nonzero_steady_state_variance() {
    init();

    let device = Default::default();
    let q = 2.0;
    let r = 0.1;
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&[0.0], q, r, &device);

    // Drive with repeated identical measurements.
    for _ in 0..5000 {
        c.update_from_vec(&[0.0]);
    }

    let variance = c.variance_to_vec();
    let p = variance[0];
    debug!("Converged Variance with Q={q}, R={r}: {p:.6}");

    assert!(p.is_finite(), "Variance must be finite");
    assert!(p > 0.0, "Variance must remain positive when Q>0");

    // Expected steady-state variance for the update used in centroids.rs
    let p_star = (-q + (q * q + 4.0 * r * q).sqrt()) / 2.0;
    debug!("Expected steady-state P*: {:.6}", p_star);

    // Should converge near P*
    let tol = 1e-2; // Relaxed for f32
    assert!(
        (p - p_star).abs() < tol,
        "Variance should converge to steady-state: got {p:.6}, expected {p_star:.6}"
    );

    // And it should not collapse toward zero (numerical floor check)
    assert!(
        p > 1e-6,
        "Variance should not collapse to ~0 when Q>0 (got {p})"
    );
}

// ========================================================================
// PART 2: Clusterer Decision Logic (The "Manager")
// ========================================================================

#[test]
fn test_clusterer_creates_new_centroid_for_outlier() {
    init();
    info!("Testing Clusterer Split Logic (New Centroid Creation)");

    let device = Default::default();
    // Setup: Max 5 clusters
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 100, device);

    // 1. Feed a point at [0,0] -> Creates Centroid 0
    clusterer.fit(&vec![vec![0.0, 0.0]]);
    assert_eq!(clusterer.centroids.len(), 1);
    let first_mean = clusterer.centroids[0].mean_to_vec();

    // 2. Feed a point FAR away at [100, 100]
    // Given defaults (var=10), Mahalanobis dist will be large (~30 sigma)
    // This should trigger the split threshold
    clusterer.fit(&vec![vec![100.0, 100.0]]);

    assert_eq!(
        clusterer.centroids.len(),
        2,
        "Should have created a second centroid for the outlier"
    );

    // Verify centroids are distinct
    let new_first_mean = clusterer.centroids[0].mean_to_vec();
    let second_mean = clusterer.centroids[1].mean_to_vec();

    assert_eq!(
        new_first_mean, first_mean,
        "First centroid should be unaffected"
    );
    assert_eq!(
        second_mean,
        vec![100.0, 100.0],
        "Second centroid acts as new anchor"
    );
}

#[test]
fn test_clusterer_assigns_and_updates_for_inlier() {
    init();
    info!("Testing Clusterer Merge/Update Logic (Inlier Assimilation)");

    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 100, device);

    // 1. Initialize with [0,0]
    clusterer.fit(&vec![vec![0.0, 0.0]]);
    let initial_variance = clusterer.centroids[0].variance_to_vec()[0];

    // 2. Feed a point NEARBY at [0.5, 0.5]
    // This should be within the ~3-sigma threshold
    clusterer.fit(&vec![vec![0.5, 0.5]]);

    assert_eq!(
        clusterer.centroids.len(),
        1,
        "Should NOT create new centroid for inlier"
    );

    // 3. Verify the single centroid UPDATED its state
    let updated_centroid = &clusterer.centroids[0];
    let updated_mean = updated_centroid.mean_to_vec();
    let updated_variance = updated_centroid.variance_to_vec();

    // Mean should have shifted towards [0.5, 0.5]
    assert!(updated_mean[0] > 0.0, "Centroid mean should shift right");

    // Variance should have decreased (learned from more data)
    assert!(
        updated_variance[0] < initial_variance,
        "Uncertainty should drop"
    );

    // Count should be 2
    assert_eq!(updated_centroid.count, 2, "Centroid count should increment");
}

#[test]
fn test_clusterer_capacity_limit_behavior() {
    init();
    info!("Testing Clusterer Soft Assignment at Max Capacity");

    let device = Default::default();
    let max_k = 2;
    let mut clusterer = KalmanClusterer::<TestBackend>::new(max_k, 100, device);

    // 1. Create 2 distinct clusters
    clusterer.fit(&vec![vec![0.0, 0.0]]); // Cluster 0
    clusterer.fit(&vec![vec![100.0, 100.0]]); // Cluster 1
    assert_eq!(clusterer.centroids.len(), 2);

    // 2. Introduce a 3rd distinct point [500, 500]
    // Normally this would split, but we are at max_k = 2
    clusterer.fit(&vec![vec![500.0, 500.0]]);

    assert_eq!(clusterer.centroids.len(), 2, "Should not exceed max_k");

    // It should be assigned to the "nearest" existing one (likely Cluster 1 at 100,100)
    // And importantly: It should NOT wreck the statistics of Cluster 1 if soft-assigned
    // (Implementation detail check: does your code update the centroid on soft assignment?
    // The provided code snippet said: "// Don't update centroid (freeze it as outlier boundary)")

    let c1_mean = clusterer.centroids[1].mean_to_vec();
    assert_eq!(
        c1_mean,
        vec![100.0, 100.0],
        "Frozen centroid should not move when soft-assigned (outlier protection)"
    );
}

#[test]
fn test_mahalanobis_stability() {
    init();
    info!("Testing Numerical Stability of Mahalanobis");

    let device = Default::default();
    let mut centroid = KalmanCentroid::<TestBackend>::from_vec(&[0.0], 0.0, 0.0, &device);

    // Force extremely small variance (but not as extreme for f32)
    centroid.variance = Tensor::from_data(TensorData::from(&[1e-12f32][..]), &device);

    // Distance should not panic or return NaN, should be clamped by epsilon
    let dist = centroid.mahalanobis_distance_sq_from_vec(&[1.0]);

    assert!(
        dist.is_finite(),
        "Distance must be finite even with near-zero variance"
    );
    assert!(dist > 1000.0, "Distance should be huge for small variance");
}

#[test]
fn test_process_noise_steady_state_variance_2d_all_dims_match() {
    init();

    let device = Default::default();
    let q = 0.25;
    let r = 0.05;

    // Same measurement for both dims, repeated.
    let z = [0.0f32, 0.0];

    let mut c = KalmanCentroid::<TestBackend>::from_vec(&z, q, r, &device);

    for _ in 0..8000 {
        c.update_from_vec(&z);
    }

    // Scalar expected steady-state for THIS update rule:
    // P = (-Q + sqrt(Q^2 + 4 R Q))/2
    let p_star = (-q + (q * q + 4.0 * r * q).sqrt()) / 2.0;

    let tol = 1e-3; // Relaxed for f32

    let variance = c.variance_to_vec();
    assert_eq!(variance.len(), 2);

    for (i, &p) in variance.iter().enumerate() {
        assert!(p.is_finite(), "dim {i}: variance must be finite");
        assert!(p > 0.0, "dim {i}: variance must be positive");
        assert!(
            (p - p_star).abs() < tol,
            "dim {i}: variance should converge to P*: got {p:.6}, expected {p_star:.6}"
        );
    }

    // Also assert the two dims converge to the same value (symmetry check).
    assert!(
        (variance[0] - variance[1]).abs() < tol as f32,
        "2D symmetry: variances should match across dims"
    );
}

#[test]
fn test_process_noise_steady_state_variance_8d_all_dims_match() {
    init();

    let device = Default::default();
    let q = 2.0;
    let r = 0.1;

    let dim = 8;
    let z: Vec<f32> = vec![0.0; dim];

    let mut c = KalmanCentroid::<TestBackend>::from_vec(&z, q, r, &device);

    for _ in 0..12000 {
        c.update_from_vec(&z);
    }

    let p_star = (-q + (q * q + 4.0 * r * q).sqrt()) / 2.0;

    // Slightly looser tolerance for longer loops / platform differences and f32.
    let tol = 5e-3;

    let variance = c.variance_to_vec();
    assert_eq!(variance.len(), dim);

    // 1) Each dimension converges to P*
    for (i, &p) in variance.iter().enumerate() {
        assert!(p.is_finite(), "dim {i}: variance must be finite");
        assert!(p > 0.0, "dim {i}: variance must be positive");
        assert!(
            (p - p_star).abs() < tol,
            "dim {i}: variance should converge to P*: got {p:.6}, expected {p_star:.6}"
        );
    }

    // 2) All dimensions agree with each other (strong symmetry assertion)
    let p0 = variance[0];
    for (i, &p) in variance.iter().enumerate().skip(1) {
        assert!(
            (p - p0).abs() < tol as f32,
            "8D symmetry: dim {i} variance differs: {p:.6} vs {p0:.6}"
        );
    }
}

#[test]
fn test_clusterer_incremental_fit() {
    init();
    info!("Testing incremental fit behavior");

    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(10, 100, device);

    // First batch
    clusterer.fit(&vec![vec![0.0, 0.0], vec![0.1, 0.1]]);
    let count_after_first = clusterer.centroids.len();

    assert!(count_after_first >= 1, "Should have at least one centroid");

    // Second batch - should be able to fit again
    clusterer.fit(&vec![vec![0.2, 0.2], vec![0.3, 0.3]]);

    // Should still have centroids
    assert!(
        clusterer.centroids.len() >= 1,
        "Should maintain centroids after second fit"
    );
}

#[test]
fn test_confidence_increases_with_observations() {
    init();

    let device = Default::default();
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 1e-4, 1e-2, &device);

    let initial_confidence = c.confidence;

    // Add several observations
    for _ in 0..10 {
        c.update_from_vec(&[0.0, 0.0]);
    }

    let final_confidence = c.confidence;

    assert!(
        final_confidence > initial_confidence,
        "Confidence should increase with more observations: {} -> {}",
        initial_confidence,
        final_confidence
    );
}
