use crate::backend::AutoBackend;
use crate::centroids::KalmanCentroid;
use burn::tensor::{Tensor, TensorData};

type TestBackend = AutoBackend;

fn assert_all_finite(xs: &[f32]) {
    assert!(
        xs.iter().all(|v| v.is_finite()),
        "expected all finite, got {xs:?}"
    );
}

#[test]
fn new_initializes_expected_state() {
    let device = Default::default();
    let x0 = vec![1.0f32, -2.0, 3.5];
    let c = KalmanCentroid::<TestBackend>::from_vec(&x0, 1e-4, 1e-2, &device);

    let mean = c.mean_to_vec();
    let variance = c.variance_to_vec();

    assert_eq!(mean, x0);
    assert_eq!(variance.len(), x0.len());
    assert!(variance.iter().all(|&v| (v - 10.0).abs() < 1e-3));
    assert_eq!(c.count, 1);
    assert!((c.process_noise - 1e-4).abs() < 1e-9);
    assert!((c.measurement_noise - 1e-2).abs() < 1e-9);
    assert!((c.confidence - 0.1).abs() < 1e-2);

    assert_all_finite(&mean);
    assert_all_finite(&variance);
    assert!(c.confidence.is_finite());
}

#[test]
fn update_moves_mean_toward_measurement_and_shrinks_variance() {
    let device = Default::default();
    let x0 = vec![0.0f32, 0.0];
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&x0, 1e-6, 1e-2, &device);

    let p_before = c.variance_to_vec();
    let mean_before = c.mean_to_vec();
    let count_before = c.count;

    let z = vec![1.0f32, -1.0];
    c.update_from_vec(&z);

    let mean_after = c.mean_to_vec();
    let p_after = c.variance_to_vec();

    // Mean should move toward measurement (not necessarily equal, but closer than before).
    for i in 0..2 {
        let dist_before = (z[i] - mean_before[i]).abs();
        let dist_after = (z[i] - mean_after[i]).abs();
        assert!(
            dist_after < dist_before,
            "mean did not move toward measurement on dim {i}: before={dist_before} after={dist_after}"
        );
    }

    // Variance should shrink (posterior variance < predicted variance).
    // Given p_pred = p + Q, and posterior = (1-K)*p_pred with 0<K<1 => posterior < p_pred.
    // With small Q, posterior should also be <= prior in this configuration.
    for i in 0..2 {
        assert!(
            p_after[i] < (p_before[i] + c.process_noise as f32),
            "variance did not shrink vs predicted on dim {i}"
        );
        assert!(
            p_after[i] <= p_before[i],
            "variance unexpectedly increased vs prior on dim {i}"
        );
    }

    assert_eq!(c.count, count_before + 1);
    assert!(c.confidence > 0.0);
    assert!(c.confidence.is_finite());

    let final_mean = c.mean_to_vec();
    let final_var = c.variance_to_vec();
    assert_all_finite(&final_mean);
    assert_all_finite(&final_var);
}

#[test]
fn repeated_updates_converge_mean_and_increase_confidence() {
    let device = Default::default();
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0, 0.0], 1e-8, 1e-3, &device);
    let z = vec![2.0f32, -1.0, 0.5];

    let mut prev_err = f32::INFINITY;
    let mut prev_conf = c.confidence;

    for _ in 0..50 {
        c.update_from_vec(&z);

        let mean = c.mean_to_vec();
        let err: f32 = mean.iter().zip(&z).map(|(m, zi)| (zi - m).abs()).sum();

        assert!(err.is_finite());
        assert!(
            err <= prev_err + 1e-6,
            "error did not monotonically decrease"
        );
        prev_err = err;

        assert!(c.confidence.is_finite());
        assert!(
            c.confidence >= prev_conf - 1e-6,
            "confidence did not increase"
        );
        prev_conf = c.confidence;
    }

    // Should end close-ish to measurement (tolerance is loose to avoid test brittleness).
    let final_mean = c.mean_to_vec();
    for i in 0..3 {
        assert!((final_mean[i] - z[i]).abs() < 1e-2);
    }
}

#[test]
fn mahalanobis_distance_sq_is_zero_at_mean_and_non_negative() {
    let device = Default::default();
    let c = KalmanCentroid::<TestBackend>::from_vec(&[1.0, 2.0], 1e-4, 1e-2, &device);

    let d0 = c.mahalanobis_distance_sq_from_vec(&[1.0, 2.0]);
    assert!((d0 - 0.0).abs() < 1e-6);

    let d1 = c.mahalanobis_distance_sq_from_vec(&[2.0, 0.0]);
    assert!(d1 >= 0.0);
    assert!(d1.is_finite());
}

#[test]
fn mahalanobis_uses_variance_scaling() {
    let device = Default::default();
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.0, 1.0, &device);

    // Make dim0 very certain (small variance), dim1 uncertain (large variance)
    let new_variance = Tensor::from_data(TensorData::from(&[1e-3f32, 1000.0][..]), &device);
    c.variance = new_variance;

    // Same absolute deviation in both dims: x = [1, 1]
    let d = c.mahalanobis_distance_sq_from_vec(&[1.0, 1.0]);

    // Contribution from dim0 should dominate due to tiny variance.
    let d0 = (1.0f32 * 1.0) / 1e-3;
    let d1 = (1.0f32 * 1.0) / 1000.0;
    let expected = d0 + d1;
    assert!((d - expected).abs() / expected < 1e-3);
}

#[test]
fn bhattacharyya_is_symmetric_and_zero_for_identical_centroids() {
    let device = Default::default();
    let mut a = KalmanCentroid::<TestBackend>::from_vec(&[0.5, -0.5], 1e-4, 1e-2, &device);
    let mut b = KalmanCentroid::<TestBackend>::from_vec(&[0.5, -0.5], 1e-4, 1e-2, &device);

    // Force identical variances as well
    let var = Tensor::from_data(TensorData::from(&[0.2f32, 0.3][..]), &device);
    a.variance = var.clone();
    b.variance = var;

    let d_ab = a.bhattacharyya_distance(&b);
    let d_ba = b.bhattacharyya_distance(&a);

    assert!(d_ab.is_finite());
    assert!(d_ba.is_finite());
    assert!((d_ab - d_ba).abs() < 1e-6, "expected symmetry");
    assert!(d_ab.abs() < 1e-6, "expected ~0 for identical distributions");
}

#[test]
fn bhattacharyya_increases_with_mean_separation() {
    let device = Default::default();
    let mut a = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 1e-4, 1e-2, &device);
    let mut b = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 1e-4, 1e-2, &device);

    let var = Tensor::from_data(TensorData::from(&[1.0f32, 1.0][..]), &device);
    a.variance = var.clone();
    b.variance = var;

    let d0 = a.bhattacharyya_distance(&b);

    b.mean = Tensor::from_data(TensorData::from(&[0.1f32, 0.1][..]), &device);
    let d_small = a.bhattacharyya_distance(&b);

    b.mean = Tensor::from_data(TensorData::from(&[2.0f32, 2.0][..]), &device);
    let d_big = a.bhattacharyya_distance(&b);

    assert!(d0 >= 0.0);
    assert!(d_small > d0);
    assert!(d_big > d_small);
}

#[test]
fn numerical_stability_tiny_variance_does_not_nan() {
    let device = Default::default();
    let mut a = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 1e-4, 1e-2, &device);
    let mut b = KalmanCentroid::<TestBackend>::from_vec(&[1.0, -1.0], 1e-4, 1e-2, &device);

    // Use small but not pathological variances (appropriate for f32)
    a.variance = Tensor::from_data(TensorData::from(&[1e-6f32, 1e-7][..]), &device);
    b.variance = Tensor::from_data(TensorData::from(&[1e-7f32, 1e-6][..]), &device);

    let d_m = a.mahalanobis_distance_sq_from_vec(&[1.0, 1.0]);
    let d_b = a.bhattacharyya_distance(&b);

    assert!(
        d_m.is_finite(),
        "Mahalanobis distance should be finite, got {}",
        d_m
    );
    assert!(
        d_b.is_finite(),
        "Bhattacharyya distance should be finite, got {}",
        d_b
    );
    assert!(d_m >= 0.0);
    assert!(d_b >= 0.0);
}

#[test]
#[should_panic(expected = "Measurement dimension mismatch")]
fn update_panics_on_dimension_mismatch() {
    let device = Default::default();
    let mut c = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 1e-4, 1e-2, &device);
    c.update_from_vec(&[1.0]); // wrong dim
}
