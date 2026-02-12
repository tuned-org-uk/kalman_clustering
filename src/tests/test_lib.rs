use crate::KalmanClusterer;
use crate::backend::AutoBackend;

use burn::tensor::backend::Backend;

type TestBackend = AutoBackend;

#[test]
fn test_kalman_clusterer_basic() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(10, 100, device);

    // Generate simple 2-cluster data
    let mut data = Vec::new();

    // Cluster 1: around (0, 0)
    for _ in 0..50 {
        data.push(vec![
            0.0 + rand::random::<f32>() * 0.1,
            0.0 + rand::random::<f32>() * 0.1,
        ]);
    }

    // Cluster 2: around (5, 5)
    for _ in 0..50 {
        data.push(vec![
            5.0 + rand::random::<f32>() * 0.1,
            5.0 + rand::random::<f32>() * 0.1,
        ]);
    }

    clusterer.fit(&data);

    // Should find 2 clusters
    assert!(
        clusterer.centroids.len() >= 2,
        "Expected at least 2 clusters, found {}",
        clusterer.centroids.len()
    );
    assert!(
        clusterer.centroids.len() <= 10,
        "Expected at most 10 clusters, found {}",
        clusterer.centroids.len()
    );
}

#[test]
fn test_export_centroids() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 10, device);

    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
    ];

    clusterer.fit(&data);

    let centroids = clusterer.export_centroids();
    assert_eq!(
        centroids.len(),
        clusterer.centroids.len(),
        "Exported centroid count should match internal count"
    );

    for (i, centroid_vec) in centroids.iter().enumerate() {
        assert_eq!(
            centroid_vec.len(),
            2,
            "Centroid {} should have 2 dimensions",
            i
        );

        // Verify all values are finite
        for (j, &val) in centroid_vec.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Centroid {} dimension {} should be finite",
                i,
                j
            );
        }
    }
}

#[test]
fn test_cluster_statistics() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 10, device);

    let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.05, 0.05]];

    clusterer.fit(&data);

    let sizes = clusterer.cluster_sizes();
    let confidences = clusterer.cluster_confidences();

    assert_eq!(
        sizes.len(),
        clusterer.centroids.len(),
        "Sizes vector length should match centroid count"
    );
    assert_eq!(
        confidences.len(),
        clusterer.centroids.len(),
        "Confidences vector length should match centroid count"
    );

    let total_size: usize = sizes.iter().sum();
    assert_eq!(
        total_size, 3,
        "Total cluster sizes should equal number of data points"
    );

    // Verify all confidences are positive and finite
    for (i, &conf) in confidences.iter().enumerate() {
        assert!(
            conf > 0.0 && conf.is_finite(),
            "Cluster {} confidence should be positive and finite, got {}",
            i,
            conf
        );
    }

    // Verify all sizes are at least 1
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
fn test_export_centroids_flat() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 10, device);

    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![5.0, 5.0],
        vec![5.1, 5.1],
    ];

    clusterer.fit(&data);

    let (flat, n, f) = clusterer.export_centroids_flat();

    assert_eq!(
        n,
        clusterer.centroids.len(),
        "Number of rows should match centroid count"
    );
    assert_eq!(f, 2, "Number of features should be 2");
    assert_eq!(flat.len(), n * f, "Flat vector length should be n * f");

    // Verify all values are finite
    for (i, &val) in flat.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Flat centroid value at index {} should be finite",
            i
        );
    }
}

#[test]
fn test_empty_clusterer() {
    let device = Default::default();
    let clusterer = KalmanClusterer::<TestBackend>::new(5, 10, device);

    assert_eq!(
        clusterer.centroids.len(),
        0,
        "New clusterer should have no centroids"
    );
    assert_eq!(
        clusterer.assignments.len(),
        10,
        "Assignments should match specified size"
    );

    let centroids = clusterer.export_centroids();
    assert_eq!(
        centroids.len(),
        0,
        "Empty clusterer should export empty centroids"
    );

    let sizes = clusterer.cluster_sizes();
    let confidences = clusterer.cluster_confidences();
    assert_eq!(sizes.len(), 0, "Empty clusterer should have no sizes");
    assert_eq!(
        confidences.len(),
        0,
        "Empty clusterer should have no confidences"
    );
}

#[test]
fn test_single_point_clustering() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 1, device);

    let data = vec![vec![1.0, 2.0]];

    clusterer.fit(&data);

    assert_eq!(
        clusterer.centroids.len(),
        1,
        "Single point should create exactly 1 centroid"
    );
    assert_eq!(
        clusterer.assignments[0],
        Some(0),
        "Single point should be assigned to centroid 0"
    );

    let centroids = clusterer.export_centroids();
    assert_eq!(centroids.len(), 1);
    assert_eq!(centroids[0].len(), 2);

    // Verify centroid is at the point location
    assert!((centroids[0][0] - 1.0).abs() < 1e-6);
    assert!((centroids[0][1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_deterministic_clustering() {
    let device: <AutoBackend as Backend>::Device = Default::default();

    // Use deterministic data
    let data = vec![
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![10.0, 10.0],
        vec![10.0, 10.0],
    ];

    let mut clusterer1 = KalmanClusterer::<TestBackend>::new(5, 4, device.clone());
    clusterer1.fit(&data);

    let mut clusterer2 = KalmanClusterer::<TestBackend>::new(5, 4, device);
    clusterer2.fit(&data);

    // Should produce the same number of clusters
    assert_eq!(
        clusterer1.centroids.len(),
        clusterer2.centroids.len(),
        "Deterministic data should produce consistent cluster counts"
    );
}

#[test]
fn test_high_dimensional_clustering() {
    let device = Default::default();
    let mut clusterer = KalmanClusterer::<TestBackend>::new(5, 10, device);

    // 5D data
    let data = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.1, 0.1, 0.1, 0.1, 0.1],
        vec![5.0, 5.0, 5.0, 5.0, 5.0],
        vec![5.1, 5.1, 5.1, 5.1, 5.1],
    ];

    clusterer.fit(&data);

    let centroids = clusterer.export_centroids();

    // Verify dimensionality is preserved
    for (i, centroid) in centroids.iter().enumerate() {
        assert_eq!(centroid.len(), 5, "Centroid {} should have 5 dimensions", i);
    }
}
