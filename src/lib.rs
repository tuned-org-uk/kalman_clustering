pub mod backend;
pub mod centroids;
pub mod laplacian;

#[cfg(test)]
mod tests;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}

use burn::tensor::backend::Backend;
use log::{debug, info, trace};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use centroids::*;

/// Kalman-based incremental clustering with automatic K selection
pub struct KalmanClusterer<B: Backend> {
    /// Active centroids (grows dynamically up to max_k)
    pub centroids: Vec<KalmanCentroid<B>>,

    /// Backend device for tensor operations
    pub device: B::Device,

    /// Maximum allowed clusters
    pub max_k: usize,

    /// Split threshold: Mahalanobis distance for creating new clusters
    /// Auto-calibrated based on data statistics
    pub split_threshold_mahal: f32,

    /// Merge threshold: Bhattacharyya distance for merging overlapping clusters
    pub merge_threshold_bhatt: f32,

    /// Process noise (Q): how much drift to tolerate
    pub process_noise: f32,

    /// Measurement noise (R): expected sensor noise
    pub measurement_noise: f32,

    /// Assignments: which centroid each item belongs to
    pub assignments: Vec<Option<usize>>,
}

impl<B: Backend> KalmanClusterer<B> {
    pub fn new(max_k: usize, nrows: usize, device: B::Device) -> Self {
        Self {
            centroids: Vec::with_capacity(max_k),
            device,
            max_k,
            split_threshold_mahal: 9.0, // χ²(F) ≈ 3σ for F-dim Gaussian
            merge_threshold_bhatt: 0.5, // Low Bhattacharyya = high overlap
            process_noise: 1e-4,        // Small drift allowance
            measurement_noise: 1e-2,    // Typical embedding noise
            assignments: vec![None; nrows],
        }
    }

    /// Single-pass adaptive clustering with auto K-selection
    pub fn fit(&mut self, rows: &[Vec<f32>]) {
        info!(
            "Kalman clustering: {} rows, max_k={}",
            rows.len(),
            self.max_k
        );

        // Auto-calibrate split threshold from first 100 samples
        if rows.len() >= 100 {
            self.auto_calibrate_threshold(&rows[..100]);
        }

        for (idx, row) in rows.iter().enumerate() {
            if idx % 1000 == 0 {
                trace!("Processing row {}/{}", idx, rows.len());
            }

            self.process_point(idx, row);

            // Periodically merge overlapping clusters (every 500 points)
            if idx % 500 == 0 && idx > 0 {
                self.merge_overlapping_centroids();
            }
        }

        // Final merge pass
        self.merge_overlapping_centroids();

        info!(
            "Kalman clustering complete: {} centroids from {} rows",
            self.centroids.len(),
            rows.len()
        );
    }

    /// Process single point: assign or create new centroid
    fn process_point(&mut self, idx: usize, point: &[f32]) {
        if self.centroids.is_empty() {
            // First point: create first centroid
            self.centroids.push(KalmanCentroid::from_vec(
                point,
                self.process_noise,
                self.measurement_noise,
                &self.device,
            ));
            self.assignments[idx] = Some(0);
            return;
        }

        // Find nearest centroid by Mahalanobis distance
        let (best_idx, best_dist_sq) = self.find_nearest_mahalanobis(point);

        // Decision: assign vs. create new
        if best_dist_sq < self.split_threshold_mahal {
            // ASSIGN to existing cluster
            self.centroids[best_idx].update_from_vec(point);
            self.assignments[idx] = Some(best_idx);
        } else if self.centroids.len() < self.max_k {
            // CREATE new cluster
            let new_idx = self.centroids.len();
            self.centroids.push(KalmanCentroid::from_vec(
                point,
                self.process_noise,
                self.measurement_noise,
                &self.device,
            ));
            self.assignments[idx] = Some(new_idx);

            debug!(
                "Created centroid {} (Mahalanobis dist={:.4} > threshold={:.4})",
                new_idx,
                best_dist_sq.sqrt(),
                self.split_threshold_mahal.sqrt()
            );
        } else {
            // SOFT ASSIGN to nearest (capacity reached)
            // Don't update centroid (freeze it as outlier boundary)
            self.assignments[idx] = Some(best_idx);
        }
    }

    /// Find nearest centroid using Mahalanobis distance
    fn find_nearest_mahalanobis(&self, point: &[f32]) -> (usize, f32) {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.mahalanobis_distance_sq_from_vec(point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .unwrap_or((0, f32::INFINITY))
    }

    /// Auto-calibrate split threshold from initial sample variance
    fn auto_calibrate_threshold(&mut self, sample: &[Vec<f32>]) {
        let dim = sample[0].len();

        // Compute global variance across sample
        let mean: Vec<f32> = (0..dim)
            .map(|j| sample.iter().map(|row| row[j]).sum::<f32>() / sample.len() as f32)
            .collect();

        let variance: Vec<f32> = (0..dim)
            .map(|j| {
                sample
                    .iter()
                    .map(|row| (row[j] - mean[j]).powi(2))
                    .sum::<f32>()
                    / sample.len() as f32
            })
            .collect();

        let mean_variance = variance.iter().sum::<f32>() / dim as f32;

        // Set threshold to 3σ in standardized space (99% confidence interval)
        // For F dimensions with unit variance: χ²(F, 0.99) ≈ F + 2√(2F)
        self.split_threshold_mahal = (dim as f32 + 2.0 * (2.0 * dim as f32).sqrt()) / mean_variance;

        info!(
            "Auto-calibrated split threshold: {:.4} (dim={}, mean_var={:.6})",
            self.split_threshold_mahal, dim, mean_variance
        );
    }

    /// Merge centroids with high statistical overlap (Bhattacharyya < threshold)
    fn merge_overlapping_centroids(&mut self) {
        let n = self.centroids.len();
        if n < 2 {
            return;
        }

        // Compute pairwise Bhattacharyya distances (parallel)
        // Note: We need to extract data for parallel processing since Backend is not always Send
        let centroid_data: Vec<(Vec<f32>, Vec<f32>)> = self
            .centroids
            .iter()
            .map(|c| (c.mean_to_vec(), c.variance_to_vec()))
            .collect();

        let distances: Vec<(usize, usize, f32)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..n)
                    .map(|j| {
                        // Compute Bhattacharyya distance on CPU side
                        let dist = compute_bhattacharyya_cpu(
                            &centroid_data[i].0,
                            &centroid_data[i].1,
                            &centroid_data[j].0,
                            &centroid_data[j].1,
                        );
                        (i, j, dist)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Find pairs to merge (distance < threshold)
        let mut to_merge: Vec<(usize, usize)> = distances
            .iter()
            .filter(|(_, _, d)| *d < self.merge_threshold_bhatt)
            .map(|(i, j, _)| (*i, *j))
            .collect();

        if to_merge.is_empty() {
            return;
        }

        // Sort by distance (merge closest pairs first)
        to_merge.sort_by(|a, b| {
            let d_a = distances.iter().find(|(i, j, _)| (*i, *j) == *a).unwrap().2;
            let d_b = distances.iter().find(|(i, j, _)| (*i, *j) == *b).unwrap().2;
            d_a.partial_cmp(&d_b).unwrap()
        });

        debug!("Merging {} overlapping centroid pairs", to_merge.len());

        // Merge pairs (weighted average by count)
        let mut merged_indices = std::collections::HashSet::new();
        for (i, j) in to_merge {
            if merged_indices.contains(&i) || merged_indices.contains(&j) {
                continue; // Already merged
            }

            let count_i = self.centroids[i].count;
            let count_j = self.centroids[j].count;
            let total_count = count_i + count_j;
            let w_i = count_i as f32 / total_count as f32;
            let w_j = count_j as f32 / total_count as f32;

            // Extract current values
            let mean_i = self.centroids[i].mean_to_vec();
            let mean_j = self.centroids[j].mean_to_vec();
            let var_i = self.centroids[i].variance_to_vec();
            let var_j = self.centroids[j].variance_to_vec();

            // Weighted mean of means
            let merged_mean: Vec<f32> = mean_i
                .iter()
                .zip(&mean_j)
                .map(|(m_i, m_j)| w_i * m_i + w_j * m_j)
                .collect();

            // Weighted variance (pooled)
            let merged_variance: Vec<f32> = var_i
                .iter()
                .zip(&var_j)
                .map(|(v_i, v_j)| w_i * v_i + w_j * v_j)
                .collect();

            // Create new merged centroid
            let merged_centroid = create_centroid_from_vecs(
                merged_mean,
                merged_variance,
                total_count,
                self.process_noise,
                self.measurement_noise,
                &self.device,
            );

            // Update centroid i with merged state
            self.centroids[i] = merged_centroid;

            // Reassign all points from j to i
            for assignment in &mut self.assignments {
                if *assignment == Some(j) {
                    *assignment = Some(i);
                }
            }

            merged_indices.insert(j);
        }

        // Remove merged centroids and update assignments
        if !merged_indices.is_empty() {
            let mut remove_list: Vec<usize> = merged_indices.into_iter().collect();
            remove_list.sort_unstable_by(|a, b| b.cmp(a)); // Remove from end to avoid index shifts

            for &idx in &remove_list {
                self.centroids.remove(idx);

                // Shift assignments
                for assignment in &mut self.assignments {
                    if let Some(a) = assignment {
                        if *a > idx {
                            *a -= 1;
                        }
                    }
                }
            }

            info!(
                "Merged {} centroids, now have {}",
                remove_list.len(),
                self.centroids.len()
            );
        }
    }

    /// Export centroids as Vec<Vec<f64>> for downstream processing
    pub fn export_centroids(&self) -> Vec<Vec<f32>> {
        self.centroids.iter().map(|c| c.mean_to_vec()).collect()
    }

    /// Export centroids as flat matrix (row-major)
    pub fn export_centroids_flat(&self) -> (Vec<f32>, usize, usize) {
        let n = self.centroids.len();
        if n == 0 {
            return (Vec::new(), 0, 0);
        }

        let f = self.centroids[0].mean.dims()[0];
        let flat: Vec<f32> = self
            .centroids
            .iter()
            .flat_map(|c| c.mean_to_vec())
            .collect();

        (flat, n, f)
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.centroids.iter().map(|c| c.count).collect()
    }

    /// Get cluster confidences
    pub fn cluster_confidences(&self) -> Vec<f32> {
        self.centroids.iter().map(|c| c.confidence).collect()
    }
}

/// Helper: Compute Bhattacharyya distance on CPU (for parallel merging)
fn compute_bhattacharyya_cpu(mean_a: &[f32], var_a: &[f32], mean_b: &[f32], var_b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..mean_a.len() {
        let sigma_avg = ((var_a[i] + var_b[i]) / 2.0).max(1e-9);
        let mean_diff = mean_a[i] - mean_b[i];

        let term1 = 0.25 * (mean_diff * mean_diff) / sigma_avg;
        let term2 = 0.5 * ((var_a[i] * var_b[i]).sqrt() / sigma_avg).ln();

        sum += term1 + term2;
    }
    sum
}

/// Helper: Create a KalmanCentroid from pre-computed vectors
fn create_centroid_from_vecs<B: Backend>(
    mean: Vec<f32>,
    variance: Vec<f32>,
    count: usize,
    process_noise: f32,
    measurement_noise: f32,
    device: &B::Device,
) -> KalmanCentroid<B> {
    use burn::tensor::{Tensor, TensorData};

    let mean_tensor = Tensor::from_data(TensorData::from(mean.as_slice()), device);
    let variance_tensor = Tensor::from_data(TensorData::from(variance.as_slice()), device);

    // Compute confidence
    let mean_variance = variance.iter().sum::<f32>() / variance.len() as f32;
    let confidence = (1.0 / mean_variance).min(100.0);

    KalmanCentroid {
        mean: mean_tensor,
        variance: variance_tensor,
        count,
        process_noise,
        measurement_noise,
        confidence,
    }
}
