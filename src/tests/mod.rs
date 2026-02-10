use crate::centroids::KalmanCentroid;
use crate::laplacian::StatisticalDistance;
use crate::laplacian::{BhattacharyyaDistance, HellingerDistance, KLDivergence};
use log::{debug, info, trace, warn};

#[test]
fn test_bhattacharyya_distance() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .is_test(true)
        .try_init()
        .ok();

    info!("Testing Bhattacharyya distance");

    let c1 = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    let c2 = KalmanCentroid::new(&[1.0, 1.0], 1e-4, 1e-2);

    let distance = BhattacharyyaDistance;
    let d = distance.distance(&c1, &c2);

    debug!("Bhattacharyya distance result: {:.6}", d);

    assert!(d > 0.0, "Distance should be positive");
    assert!(d.is_finite(), "Distance should be finite");

    info!("Bhattacharyya test passed");
}

#[test]
fn test_hellinger_bounded() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .is_test(true)
        .try_init()
        .ok();

    info!("Testing Hellinger distance bounds");

    let c1 = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    let c2 = KalmanCentroid::new(&[10.0, 10.0], 1e-4, 1e-2);

    let distance = HellingerDistance;
    let d = distance.distance(&c1, &c2);

    debug!("Hellinger distance result: {:.6}", d);

    assert!(
        d >= 0.0 && d <= 1.0,
        "Hellinger should be in [0,1], got {:.6}",
        d
    );

    info!("Hellinger test passed");
}

#[test]
fn test_kl_divergence() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .is_test(true)
        .try_init()
        .ok();

    info!("Testing KL divergence");

    let c1 = KalmanCentroid::new(&[0.0, 0.0], 1e-4, 1e-2);
    let c2 = KalmanCentroid::new(&[1.0, 1.0], 1e-4, 1e-2);

    let distance = KLDivergence;
    let d = distance.distance(&c1, &c2);

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
    let hellinger = HellingerDistance;

    let test_dist = 0.5;

    let w_bhatt = bhatt.distance_to_weight(test_dist);
    let w_kl = kl.distance_to_weight(test_dist);
    let w_hellinger = hellinger.distance_to_weight(test_dist);

    debug!(
        "Distance {} → weights: Bhatt={:.6}, KL={:.6}, Hellinger={:.6}",
        test_dist, w_bhatt, w_kl, w_hellinger
    );

    assert!(
        w_bhatt > 0.0 && w_bhatt <= 1.0,
        "Bhattacharyya weight out of bounds"
    );
    assert!(w_kl > 0.0 && w_kl <= 1.0, "KL weight out of bounds");
    assert!(
        w_hellinger >= 0.0 && w_hellinger <= 1.0,
        "Hellinger weight out of bounds"
    );

    info!("Distance-to-weight conversion tests passed");
}
