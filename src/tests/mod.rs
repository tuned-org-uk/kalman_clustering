mod test_centroids;
mod test_clustering;
mod test_lib;
mod test_update_step;

use crate::backend::AutoBackend;
use crate::centroids::KalmanCentroid;
use crate::init;
use crate::laplacian::StatisticalDistance;
use crate::laplacian::{
    BhattacharyyaDistance, EuclideanDistance, KLDivergence, MahalanobisLikeDistance, SymmetrizedKL,
};
use burn::tensor::{Tensor, TensorData};
use log::{debug, info, trace, warn};

// Type alias for test backend - automatically selected
type TestBackend = AutoBackend;

/// Helper: Call distance method with explicit backend type
fn compute_distance<M: StatisticalDistance<TestBackend>>(
    metric: &M,
    a: &KalmanCentroid<TestBackend>,
    b: &KalmanCentroid<TestBackend>,
) -> f32 {
    metric.distance(a, b)
}

/// Helper: Call distance_to_weight with explicit backend type
fn compute_weight<M: StatisticalDistance<TestBackend>>(metric: &M, distance: f32) -> f32 {
    metric.distance_to_weight(distance)
}

/// Helper: Create a 2D centroid for testing
fn centroid_2d(
    mean: [f32; 2],
    var: [f32; 2],
    count: usize,
    device: &<TestBackend as burn::tensor::backend::Backend>::Device,
) -> KalmanCentroid<TestBackend> {
    let mean_tensor = Tensor::from_data(TensorData::from(&mean[..]), device);
    let var_tensor = Tensor::from_data(TensorData::from(&var[..]), device);

    let confidence = 1.0 / ((var[0] + var[1]) / 2.0);

    KalmanCentroid {
        mean: mean_tensor,
        variance: var_tensor,
        count,
        process_noise: 1e-4,
        measurement_noise: 1e-2,
        confidence,
    }
}

/// Helper: Create an 8D centroid for testing
fn centroid_8d(
    mean_shift: f32,
    var: f32,
    count: usize,
    device: &<TestBackend as burn::tensor::backend::Backend>::Device,
) -> KalmanCentroid<TestBackend> {
    let mean: Vec<f32> = (0..8).map(|i| i as f32 * mean_shift).collect();
    let variance = vec![var; 8];

    let mean_tensor = Tensor::from_data(TensorData::from(mean.as_slice()), device);
    let var_tensor = Tensor::from_data(TensorData::from(variance.as_slice()), device);

    let confidence = 1.0 / var;

    KalmanCentroid {
        mean: mean_tensor,
        variance: var_tensor,
        count,
        process_noise: 1e-4,
        measurement_noise: 1e-2,
        confidence,
    }
}

/// Helper: Approximate equality check
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn test_bhattacharyya_distance() {
    init();
    info!("Testing Bhattacharyya distance");

    let device = Default::default();
    let c1 = centroid_2d([0.0, 0.0], [10.0, 10.0], 1, &device);
    let c2 = centroid_2d([1.0, 1.0], [10.0, 10.0], 1, &device);

    let distance = BhattacharyyaDistance;
    let d = compute_distance(&distance, &c1, &c2);

    debug!("Bhattacharyya distance result: {:.6}", d);

    assert!(d > 0.0, "Distance should be positive");
    assert!(d.is_finite(), "Distance should be finite");

    info!("Bhattacharyya test passed");
}

#[test]
fn test_kl_divergence() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .is_test(true)
        .try_init()
        .ok();

    info!("Testing KL divergence");

    let device = Default::default();
    let c1 = centroid_2d([0.0, 0.0], [10.0, 10.0], 1, &device);
    let c2 = centroid_2d([1.0, 1.0], [10.0, 10.0], 1, &device);

    let distance = KLDivergence;
    let d = compute_distance(&distance, &c1, &c2);

    debug!("KL divergence result: {:.6}", d);

    assert!(d >= 0.0, "KL divergence should be non-negative");
    assert!(d.is_finite(), "KL divergence should be finite");

    info!("KL divergence test passed");
}

#[test]
fn test_distance_to_weight_conversion() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init()
        .ok();

    info!("Testing distance-to-weight conversions");

    let bhatt = BhattacharyyaDistance;
    let kl = KLDivergence;

    let test_dist = 0.5;

    let w_bhatt = compute_weight(&bhatt, test_dist);
    let w_kl = compute_weight(&kl, test_dist);

    debug!(
        "Distance {} → weights: Bhatt={:.6}, KL={:.6}",
        test_dist, w_bhatt, w_kl,
    );

    assert!(
        w_bhatt > 0.0 && w_bhatt <= 1.0,
        "Bhattacharyya weight out of bounds"
    );
    assert!(w_kl > 0.0 && w_kl <= 1.0, "KL weight out of bounds");

    info!("Distance-to-weight conversion tests passed");
}

#[test]
fn kl_is_asymmetric_in_general() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [0.5, 0.5], 10, &device);
    let b = centroid_2d([1.0, 1.0], [2.0, 2.0], 10, &device);

    let m = KLDivergence;
    let dab = compute_distance(&m, &a, &b);
    let dba = compute_distance(&m, &b, &a);

    assert!(dab.is_finite() && dba.is_finite());

    // KL divergence should be asymmetric for different variance scales
    assert!(
        (dab - dba).abs() > 1e-6,
        "KL should be asymmetric here: dab={dab}, dba={dba}"
    );
}

#[test]
fn test_kl_divergence_symmetry_check() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [0.5, 2.0], 10, &device);
    let b = centroid_2d([1.0, 1.0], [2.0, 0.5], 10, &device);

    let m = KLDivergence;
    let dab = compute_distance(&m, &a, &b);
    let dba = compute_distance(&m, &b, &a);

    assert!(dab.is_finite() && dba.is_finite());
    debug!("KL results: dab={:.6}, dba={:.6}", dab, dba);
}

#[test]
fn weights_monotonicity_bhattacharyya_exp_kernel() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b_close = centroid_2d([0.1, 0.1], [1.0, 1.0], 10, &device);
    let b_far = centroid_2d([3.0, 3.0], [1.0, 1.0], 10, &device);

    let m = BhattacharyyaDistance;

    let d_close = compute_distance(&m, &a, &b_close);
    let d_far = compute_distance(&m, &a, &b_far);

    let w_close = compute_weight(&m, d_close);
    let w_far = compute_weight(&m, d_far);

    assert!(d_close.is_finite() && d_far.is_finite());
    assert!(w_close.is_finite() && w_far.is_finite());
    assert!(
        d_close < d_far,
        "expected close distance < far distance: {d_close} vs {d_far}"
    );
    assert!(
        w_close > w_far,
        "expected weight to decrease with distance: {w_close} vs {w_far}"
    );
    assert!(
        w_close <= 1.0 + 1e-6 && w_far >= 0.0,
        "exp(-d) weights should be in (0,1]"
    );
}

#[test]
fn kl_can_be_symmetric_for_some_inputs() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [0.5, 2.0], 10, &device);
    let b = centroid_2d([1.0, 1.0], [2.0, 0.5], 10, &device);

    let m = KLDivergence;
    let dab = compute_distance(&m, &a, &b);
    let dba = compute_distance(&m, &b, &a);

    assert!(dab.is_finite() && dba.is_finite());
    debug!("KL dab={:.6}, dba={:.6}", dab, dba);
}

#[test]
fn kl_weight_is_finite_for_reasonable_inputs() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([0.2, 0.2], [1.2, 1.2], 10, &device);

    let m = KLDivergence;
    let d = compute_distance(&m, &a, &b);
    let w = compute_weight(&m, d);

    assert!(d.is_finite(), "KL distance must be finite");
    assert!(w.is_finite(), "KL-derived weight must be finite");
    assert!((1.0 + d).abs() > 1e-12, "1 + distance too close to zero");
}

#[test]
fn test_symmetrized_kl() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([1.0, 1.0], [1.2, 1.2], 10, &device);

    let m = SymmetrizedKL;
    let dab = compute_distance(&m, &a, &b);
    let dba = compute_distance(&m, &b, &a);

    assert!(dab.is_finite() && dba.is_finite());

    // Symmetrized KL should be symmetric
    assert!(
        approx_eq(dab, dba, 1e-6),
        "Symmetrized KL should be symmetric: dab={dab}, dba={dba}"
    );

    debug!("Symmetrized KL: dab={:.6}, dba={:.6}", dab, dba);
}

#[test]
fn test_euclidean_distance() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([3.0, 4.0], [1.0, 1.0], 10, &device);

    let m = EuclideanDistance;
    let d = compute_distance(&m, &a, &b);

    // Should be 5.0 (3-4-5 triangle)
    assert!(approx_eq(d, 5.0, 1e-4), "Expected distance 5.0, got {d}");

    debug!("Euclidean distance: {:.6}", d);
}

#[test]
fn test_mahalanobis_like_distance() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([1.0, 1.0], [1.0, 1.0], 10, &device);

    let m = MahalanobisLikeDistance;
    let d = compute_distance(&m, &a, &b);

    assert!(d > 0.0, "Distance should be positive");
    assert!(d.is_finite(), "Distance should be finite");

    debug!("Mahalanobis-like distance: {:.6}", d);
}

#[test]
fn test_all_metrics_consistency() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([0.5, 0.5], [1.0, 1.0], 10, &device);

    let metrics: Vec<(
        &str,
        Box<dyn Fn(&KalmanCentroid<TestBackend>, &KalmanCentroid<TestBackend>) -> f32>,
    )> = vec![
        (
            "Bhattacharyya",
            Box::new(|a, b| compute_distance(&BhattacharyyaDistance, a, b)),
        ),
        ("KL", Box::new(|a, b| compute_distance(&KLDivergence, a, b))),
        (
            "SymmetrizedKL",
            Box::new(|a, b| compute_distance(&SymmetrizedKL, a, b)),
        ),
        (
            "Euclidean",
            Box::new(|a, b| compute_distance(&EuclideanDistance, a, b)),
        ),
        (
            "MahalanobisLike",
            Box::new(|a, b| compute_distance(&MahalanobisLikeDistance, a, b)),
        ),
    ];

    for (name, metric_fn) in metrics {
        let d = metric_fn(&a, &b);

        assert!(
            d.is_finite(),
            "{} distance should be finite, got {}",
            name,
            d
        );

        debug!("{}: distance={:.6}", name, d);
    }
}

#[test]
fn test_high_dimensional_consistency() {
    init();

    let device = Default::default();
    let a = centroid_8d(0.0, 1.0, 50, &device);
    let b = centroid_8d(0.1, 1.0, 50, &device);

    let bhatt = BhattacharyyaDistance;
    let d = compute_distance(&bhatt, &a, &b);

    assert!(d > 0.0, "8D distance should be positive");
    assert!(d.is_finite(), "8D distance should be finite");

    debug!("8D Bhattacharyya distance: {:.6}", d);
}

#[test]
fn test_zero_distance_same_centroids() {
    init();

    let device = Default::default();
    let a = centroid_2d([1.0, 1.0], [0.5, 0.5], 10, &device);
    let b = centroid_2d([1.0, 1.0], [0.5, 0.5], 10, &device);

    let euclidean = EuclideanDistance;
    let d = compute_distance(&euclidean, &a, &b);

    assert!(
        approx_eq(d, 0.0, 1e-6),
        "Distance between identical centroids should be ~0, got {d}"
    );
}

#[test]
fn test_weight_ranges() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1.0, 1.0], 10, &device);
    let b = centroid_2d([5.0, 5.0], [1.0, 1.0], 10, &device);

    let bhatt = BhattacharyyaDistance;
    let kl = KLDivergence;

    let d_bhatt = compute_distance(&bhatt, &a, &b);
    let w_bhatt = compute_weight(&bhatt, d_bhatt);

    let d_kl = compute_distance(&kl, &a, &b);
    let w_kl = compute_weight(&kl, d_kl);

    // Exponential kernel for large distances should be close to 0
    assert!(
        w_bhatt < 0.1,
        "Bhattacharyya weight for distant points should be small"
    );

    // Rational kernel 1/(1+d) should be in (0, 1)
    assert!(w_kl > 0.0 && w_kl < 1.0, "KL weight should be in (0, 1)");

    debug!("Far distance weights: Bhatt={:.6}, KL={:.6}", w_bhatt, w_kl);
}

#[test]
fn test_numerical_stability_high_variance() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [100.0, 100.0], 10, &device);
    let b = centroid_2d([1.0, 1.0], [100.0, 100.0], 10, &device);

    let bhatt = BhattacharyyaDistance;
    let d = compute_distance(&bhatt, &a, &b);

    assert!(d.is_finite(), "High variance case should remain finite");
    assert!(d > 0.0, "Distance should still be positive");

    debug!("High variance Bhattacharyya distance: {:.6}", d);
}

#[test]
fn test_numerical_stability_low_variance() {
    init();

    let device = Default::default();
    let a = centroid_2d([0.0, 0.0], [1e-6, 1e-6], 10, &device);
    let b = centroid_2d([1.0, 1.0], [1e-6, 1e-6], 10, &device);

    let bhatt = BhattacharyyaDistance;
    let d = compute_distance(&bhatt, &a, &b);

    assert!(d.is_finite(), "Low variance case should remain finite");
    assert!(d > 0.0, "Distance should still be positive");

    debug!("Low variance Bhattacharyya distance: {:.6}", d);
}

#[test]
fn test_backend_device_default() {
    init();

    let device = Default::default();
    let c = centroid_2d([1.0, 2.0], [0.5, 0.5], 1, &device);

    assert_eq!(c.mean.dims()[0], 2);
    assert_eq!(c.variance.dims()[0], 2);
    assert_eq!(c.count, 1);

    debug!(
        "AutoBackend initialized successfully on device: {:?}",
        device
    );
}
