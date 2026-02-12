use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Kalman-tracked centroid with adaptive covariance (Burn 0.21)
#[derive(Clone, Debug)]
pub struct KalmanCentroid<B: Backend> {
    /// Mean state vector μ (F-dimensional)
    pub mean: Tensor<B, 1>,

    /// Diagonal covariance P (F-dimensional variance per feature)
    pub variance: Tensor<B, 1>,

    /// Number of points assigned (for weight in graph construction)
    pub count: usize,

    /// Process noise Q: increases variance over time (drift tolerance)
    pub process_noise: f32,

    /// Measurement noise R: expected embedding noise
    pub measurement_noise: f32,

    /// Confidence: 1/(mean(variance)), used for auto-pruning
    pub confidence: f32,
}

impl<B: Backend> KalmanCentroid<B> {
    /// Initialize from first data point with high initial uncertainty
    pub fn new(initial_point: Tensor<B, 1>, process_noise: f32, measurement_noise: f32) -> Self {
        let device = initial_point.device();
        let dim = initial_point.dims()[0];

        // High initial uncertainty
        let variance = Tensor::ones([dim], &device).mul_scalar(10.0);

        Self {
            mean: initial_point,
            variance,
            count: 1,
            process_noise,
            measurement_noise,
            confidence: 0.1, // Low confidence initially
        }
    }

    /// Create from Vec<f64> (convenience constructor)
    pub fn from_vec(
        initial_point: &[f32],
        process_noise: f32,
        measurement_noise: f32,
        device: &B::Device,
    ) -> Self {
        let data = TensorData::from(initial_point);
        let tensor = Tensor::from_data(data, device);
        Self::new(tensor, process_noise, measurement_noise)
    }

    /// Kalman update: predict (add process noise) + correct (measurement update)
    pub fn update(&mut self, measurement: Tensor<B, 1>) {
        assert_eq!(
            measurement.dims()[0],
            self.mean.dims()[0],
            "Measurement dimension mismatch"
        );

        // 1. Prediction: add process noise to account for drift
        let p_pred = self.variance.clone().add_scalar(self.process_noise);

        // 2. Correction: compute Kalman gain
        // K = P_pred / (P_pred + R)
        let kalman_gain = p_pred
            .clone()
            .div(p_pred.clone().add_scalar(self.measurement_noise));

        // 3. Update mean with weighted innovation
        // μ = μ + K * (z - μ)
        let innovation = measurement.sub(self.mean.clone());
        self.mean = self.mean.clone().add(kalman_gain.clone().mul(innovation));

        // 4. Update covariance (shrinks with each observation)
        // P = (1 - K) * P_pred
        let ones = Tensor::ones_like(&kalman_gain);
        self.variance = ones.sub(kalman_gain).mul(p_pred);

        self.count += 1;

        // Recompute confidence: inverse average variance
        let variance_data = self.variance.clone().into_data();
        let variance_vec: Vec<f32> = variance_data.to_vec().expect("Failed to convert variance");
        let mean_variance = variance_vec.iter().sum::<f32>() / variance_vec.len() as f32;

        self.confidence = (1.0 / mean_variance).min(100.0); // Cap to avoid numerical issues
    }

    /// Convenience update from Vec<f64>
    pub fn update_from_vec(&mut self, measurement: &[f32]) {
        let device = self.mean.device();
        let data = TensorData::from(measurement);
        let tensor = Tensor::from_data(data, &device);
        self.update(tensor);
    }

    /// Mahalanobis distance: distance normalized by variance
    /// Returns squared distance: Σ((x_i - μ_i)² / σ_i²)
    pub fn mahalanobis_distance_sq(&self, point: Tensor<B, 1>) -> f32 {
        // Numerical stability: max(variance, 1e-9)
        let variance_stable = self.variance.clone().clamp_min(1e-9);

        // Compute: (point - mean)² / variance
        let diff = point.sub(self.mean.clone());
        let diff_sq = diff.clone().mul(diff);
        let normalized = diff_sq.div(variance_stable);

        // Sum all elements
        let result_data = normalized.sum().into_data();
        result_data
            .to_vec::<f32>()
            .expect("Failed to convert result")[0]
    }

    /// Convenience Mahalanobis distance from Vec<f64>
    pub fn mahalanobis_distance_sq_from_vec(&self, point: &[f32]) -> f32 {
        let device = self.mean.device();
        let data = TensorData::from(point);
        let tensor = Tensor::from_data(data, &device);
        self.mahalanobis_distance_sq(tensor)
    }

    /// Bhattacharyya distance to another Gaussian centroid
    /// Measures statistical overlap for graph edge weighting
    pub fn bhattacharyya_distance(&self, other: &KalmanCentroid<B>) -> f32 {
        // σ_avg = (σ₁ + σ₂) / 2
        let sigma_avg = self
            .variance
            .clone()
            .add(other.variance.clone())
            .div_scalar(2.0)
            .clamp_min(1e-9); // Numerical stability

        // Mean difference: (μ₁ - μ₂)²
        let mean_diff = self.mean.clone().sub(other.mean.clone());
        let mean_diff_sq = mean_diff.clone().mul(mean_diff);

        // Term 1: 0.25 * (μ₁ - μ₂)² / σ_avg
        let term1 = mean_diff_sq.div(sigma_avg.clone()).mul_scalar(0.25);

        // Term 2: 0.5 * ln(√(σ₁ * σ₂) / σ_avg)
        let variance_product = self.variance.clone().mul(other.variance.clone());
        let sqrt_product = variance_product.sqrt();
        let ratio = sqrt_product.div(sigma_avg);
        let term2 = ratio.log().mul_scalar(0.5);

        // Sum both terms
        let result = term1.add(term2).sum();
        let result_data = result.into_data();
        result_data
            .to_vec::<f32>()
            .expect("Failed to convert result")[0]
    }

    /// Export mean to Vec<f64>
    pub fn mean_to_vec(&self) -> Vec<f32> {
        let data = self.mean.clone().into_data();
        data.to_vec().expect("Failed to convert mean")
    }

    /// Export variance to Vec<f64>
    pub fn variance_to_vec(&self) -> Vec<f32> {
        let data = self.variance.clone().into_data();
        data.to_vec().expect("Failed to convert variance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AutoBackend;
    type TestBackend = AutoBackend;

    #[test]
    fn test_kalman_centroid_creation() {
        let device = Default::default();
        let initial = vec![1.0, 2.0, 3.0];
        let centroid = KalmanCentroid::<TestBackend>::from_vec(&initial, 0.01, 0.1, &device);

        assert_eq!(centroid.count, 1);
        assert_eq!(centroid.mean_to_vec(), initial);
    }

    #[test]
    fn test_kalman_update() {
        let device = Default::default();
        let initial = vec![0.0, 0.0];
        let mut centroid = KalmanCentroid::<TestBackend>::from_vec(&initial, 0.01, 0.1, &device);

        centroid.update_from_vec(&[1.0, 1.0]);
        centroid.update_from_vec(&[1.0, 1.0]);

        let mean = centroid.mean_to_vec();
        assert!(mean[0] > 0.5); // Should have moved toward measurements
        assert_eq!(centroid.count, 3);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let device = Default::default();
        let centroid = KalmanCentroid::<TestBackend>::from_vec(&[0.0, 0.0], 0.01, 0.1, &device);

        let dist = centroid.mahalanobis_distance_sq_from_vec(&[1.0, 1.0]);
        assert!(dist > 0.0);
    }
}
