use crate::KalmanClusterer;
use crate::backend::AutoBackend;

type TestBackend = AutoBackend;

fn make_two_blob_dataset(n_per: usize) -> Vec<Vec<f32>> {
    // Deterministic "two blobs" in 2D, no RNG required.
    // Blob A around (0,0), Blob B around (10,10).
    let mut rows = Vec::with_capacity(2 * n_per);
    for i in 0..n_per {
        let t = i as f32 / (n_per as f32);
        rows.push(vec![0.2 * t, -0.2 * t]); // A: small spread
        rows.push(vec![10.0 + 0.2 * t, 10.0 - 0.2 * t]); // B: small spread
    }
    rows
}

fn make_three_blob_dataset(n_per: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(3 * n_per);
    for i in 0..n_per {
        let t = i as f32 / (n_per as f32);
        rows.push(vec![0.0 + 0.1 * t, 0.0 - 0.1 * t]);
        rows.push(vec![10.0 + 0.1 * t, 10.0 - 0.1 * t]);
        rows.push(vec![-10.0 + 0.1 * t, 10.0 + 0.1 * t]);
    }
    rows
}

#[test]
fn kalman_clusterer_two_blobs_creates_multiple_centroids_and_assigns_all() {
    let rows = make_two_blob_dataset(200);
    let n = rows.len();

    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(64, n, device);

    // Make splitting easier than default (default may still work, this just hardens the test).
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    assert!(
        kc.centroids.len() >= 2,
        "Expected at least 2 centroids for two blobs, got {}",
        kc.centroids.len()
    );

    assert_eq!(
        kc.assignments.len(),
        n,
        "Assignments length mismatch: {} vs {}",
        kc.assignments.len(),
        n
    );

    for (i, a) in kc.assignments.iter().enumerate() {
        let idx = a.expect("Every row should be assigned in this dataset");
        assert!(
            idx < kc.centroids.len(),
            "Assignment out of bounds at row {}: {} >= {}",
            i,
            idx,
            kc.centroids.len()
        );
    }

    for (cidx, c) in kc.centroids.iter().enumerate() {
        let mean = c.mean_to_vec();
        let variance = c.variance_to_vec();

        assert_eq!(mean.len(), 2, "Centroid {} mean dimension mismatch", cidx);
        assert_eq!(
            variance.len(),
            2,
            "Centroid {} variance dimension mismatch",
            cidx
        );
        assert!(
            mean.iter().all(|x| x.is_finite()),
            "Centroid {} mean has NaN/inf",
            cidx
        );
        assert!(
            variance.iter().all(|x| x.is_finite() && *x > 0.0),
            "Centroid {} variance has invalid values",
            cidx
        );
        assert!(c.count >= 1, "Centroid {} count must be >=1", cidx);
    }
}

#[test]
fn kalman_clusterer_export_centroids_shape_is_correct() {
    let rows = make_three_blob_dataset(120);
    let n = rows.len();

    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(128, n, device);
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    let centroids = kc.export_centroids();
    let k = centroids.len();
    let f = if k > 0 { centroids[0].len() } else { 0 };

    assert_eq!(f, 2, "Expected 2 features in exported centroids, got {}", f);
    assert_eq!(
        k,
        kc.centroids.len(),
        "Exported centroids count should match number of centroids"
    );
    assert!(
        k >= 3,
        "Expected at least 3 centroids for three blobs, got {}",
        k
    );

    // Basic finiteness check on exported centroids.
    for (i, centroid) in centroids.iter().enumerate() {
        for (j, &v) in centroid.iter().enumerate() {
            assert!(
                v.is_finite(),
                "export_centroids contains NaN/inf at ({},{})",
                i,
                j
            );
        }
    }
}

#[test]
fn kalman_clusterer_export_centroids_flat_shape_is_correct() {
    let rows = make_three_blob_dataset(120);
    let n = rows.len();

    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(128, n, device);
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    let (flat, k, f) = kc.export_centroids_flat();

    assert_eq!(f, 2, "Expected 2 features in exported centroids, got {}", f);
    assert_eq!(
        k,
        kc.centroids.len(),
        "Flat matrix rows should match number of centroids"
    );
    assert!(
        k >= 3,
        "Expected at least 3 centroids for three blobs, got {}",
        k
    );

    assert_eq!(
        flat.len(),
        k * f,
        "Flat vector length should be k * f: {} != {} * {}",
        flat.len(),
        k,
        f
    );

    // Basic finiteness check on exported flat matrix.
    for (idx, &v) in flat.iter().enumerate() {
        assert!(
            v.is_finite(),
            "export_centroids_flat contains NaN/inf at index {}",
            idx
        );
    }
}

#[test]
fn kalman_clusterer_respects_max_k_cap() {
    // Many far-apart points in 2D; with a low split threshold, we try to create lots of clusters.
    let mut rows = Vec::new();
    for i in 0..200usize {
        rows.push(vec![i as f32 * 10.0, 0.0]);
    }

    let n = rows.len();
    let max_k = 8;
    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(max_k, n, device);
    kc.split_threshold_mahal = 1e-6; // force new cluster creation pressure

    kc.fit(&rows);

    assert!(
        kc.centroids.len() <= max_k,
        "Expected centroids <= max_k, got {} > {}",
        kc.centroids.len(),
        max_k
    );

    // Even when max_k hit, we still assign all points.
    assert!(kc.assignments.iter().all(|a| a.is_some()));
}

#[test]
fn kalman_clusterer_cluster_statistics() {
    let rows = make_two_blob_dataset(100);
    let n = rows.len();

    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(64, n, device);
    kc.split_threshold_mahal = 1.0;
    kc.merge_threshold_bhatt = 0.05;

    kc.fit(&rows);

    let sizes = kc.cluster_sizes();
    let confidences = kc.cluster_confidences();

    assert_eq!(sizes.len(), kc.centroids.len());
    assert_eq!(confidences.len(), kc.centroids.len());

    // Total size should equal number of points
    let total_size: usize = sizes.iter().sum();
    assert_eq!(
        total_size, n,
        "Total cluster sizes should equal number of points"
    );

    // All confidences should be positive and finite
    for (i, &conf) in confidences.iter().enumerate() {
        assert!(
            conf > 0.0 && conf.is_finite(),
            "Cluster {} confidence should be positive and finite, got {}",
            i,
            conf
        );
    }

    // All sizes should be at least 1
    for (i, &size) in sizes.iter().enumerate() {
        assert!(
            size >= 1,
            "Cluster {} should have at least 1 point, got {}",
            i,
            size
        );
    }
}

#[test]
fn kalman_clusterer_single_point() {
    let rows = vec![vec![1.0f32, 2.0]];
    let n = rows.len();

    let device = Default::default();
    let mut kc = KalmanClusterer::<TestBackend>::new(10, n, device);

    kc.fit(&rows);

    assert_eq!(
        kc.centroids.len(),
        1,
        "Single point should create 1 centroid"
    );
    assert_eq!(
        kc.assignments[0],
        Some(0),
        "Point should be assigned to centroid 0"
    );

    let centroid_mean = kc.centroids[0].mean_to_vec();
    assert_eq!(centroid_mean.len(), 2);
    assert!((centroid_mean[0] - 1.0).abs() < 1e-6);
    assert!((centroid_mean[1] - 2.0).abs() < 1e-6);
}

#[test]
fn kalman_clusterer_empty_initialization() {
    let device = Default::default();
    let kc = KalmanClusterer::<TestBackend>::new(10, 100, device);

    assert_eq!(
        kc.centroids.len(),
        0,
        "New clusterer should have no centroids"
    );
    assert_eq!(
        kc.assignments.len(),
        100,
        "Assignments vector should be sized correctly"
    );
    assert!(
        kc.assignments.iter().all(|a| a.is_none()),
        "All assignments should be None initially"
    );
}
