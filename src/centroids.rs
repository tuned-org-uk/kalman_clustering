/// Kalman-tracked centroid with adaptive covariance
#[derive(Clone, Debug)]
pub struct KalmanCentroid {
    /// Mean state vector μ (F-dimensional)
    pub mean: Vec<f64>,

    /// Diagonal covariance P (F-dimensional variance per feature)
    pub variance: Vec<f64>,

    /// Number of points assigned (for weight in graph construction)
    pub count: usize,

    /// Process noise Q: increases variance over time (drift tolerance)
    pub process_noise: f64,

    /// Measurement noise R: expected embedding noise
    pub measurement_noise: f64,

    /// Confidence: 1/(mean(variance)), used for auto-pruning
    pub confidence: f64,
}

impl KalmanCentroid {
    /// Initialize from first data point with high initial uncertainty
    pub fn new(initial_point: &[f64], process_noise: f64, measurement_noise: f64) -> Self {
        let dim = initial_point.len();
        Self {
            mean: initial_point.to_vec(),
            variance: vec![10.0; dim], // High initial uncertainty
            count: 1,
            process_noise,
            measurement_noise,
            confidence: 0.1, // Low confidence initially
        }
    }

    /// Kalman update: predict (add process noise) + correct (measurement update)
    pub fn update(&mut self, measurement: &[f64]) {
        assert_eq!(measurement.len(), self.mean.len());

        for i in 0..self.mean.len() {
            // 1. Prediction: add process noise to account for drift
            let p_pred = self.variance[i] + self.process_noise;

            // 2. Correction: compute Kalman gain
            let kalman_gain = p_pred / (p_pred + self.measurement_noise);

            // 3. Update mean with weighted innovation
            self.mean[i] += kalman_gain * (measurement[i] - self.mean[i]);

            // 4. Update covariance (shrinks with each observation)
            self.variance[i] = (1.0 - kalman_gain) * p_pred;
        }

        self.count += 1;

        // Recompute confidence: inverse average variance
        self.confidence = 1.0 / (self.variance.iter().sum::<f64>() / self.mean.len() as f64);
        self.confidence = self.confidence.min(100.0); // Cap to avoid numerical issues
    }

    /// Mahalanobis distance: distance normalized by variance
    /// Returns squared distance: Σ((x_i - μ_i)² / σ_i²)
    pub fn mahalanobis_distance_sq(&self, point: &[f64]) -> f64 {
        point
            .iter()
            .zip(&self.mean)
            .zip(&self.variance)
            .map(|((x, mu), var)| {
                let diff = x - mu;
                (diff * diff) / var.max(1e-9) // Numerical stability
            })
            .sum()
    }

    /// Bhattacharyya distance to another Gaussian centroid
    /// Measures statistical overlap for graph edge weighting
    pub fn bhattacharyya_distance(&self, other: &KalmanCentroid) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.mean.len() {
            let sigma_avg = (self.variance[i] + other.variance[i]) / 2.0;
            let mean_diff = self.mean[i] - other.mean[i];

            // Simplified Bhattacharyya for diagonal Gaussians
            let term1 = 0.25 * (mean_diff * mean_diff) / sigma_avg.max(1e-9);
            let term2 = 0.5 * ((self.variance[i] * other.variance[i]).sqrt() / sigma_avg).ln();

            sum += term1 + term2;
        }
        sum
    }
}
