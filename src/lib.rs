pub mod centroids;
pub mod laplacian;

#[cfg(test)]
mod tests;

use log::{debug, info, trace};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

use centroids::*;

/// Kalman-based incremental clustering with automatic K selection
pub struct KalmanClusterer {
    /// Active centroids (grows dynamically up to max_k)
    pub centroids: Vec<KalmanCentroid>,

    /// Maximum allowed clusters
    pub max_k: usize,

    /// Split threshold: Mahalanobis distance for creating new clusters
    /// Auto-calibrated based on data statistics
    pub split_threshold_mahal: f64,

    /// Merge threshold: Bhattacharyya distance for merging overlapping clusters
    pub merge_threshold_bhatt: f64,

    /// Process noise (Q): how much drift to tolerate
    pub process_noise: f64,

    /// Measurement noise (R): expected sensor noise
    pub measurement_noise: f64,

    /// Assignments: which centroid each item belongs to
    pub assignments: Vec<Option<usize>>,
}

impl KalmanClusterer {
    pub fn new(max_k: usize, nrows: usize) -> Self {
        Self {
            centroids: Vec::with_capacity(max_k),
            max_k,
            split_threshold_mahal: 9.0, // χ²(F) ≈ 3σ for F-dim Gaussian
            merge_threshold_bhatt: 0.5, // Low Bhattacharyya = high overlap
            process_noise: 1e-4,        // Small drift allowance
            measurement_noise: 1e-2,    // Typical embedding noise
            assignments: vec![None; nrows],
        }
    }

    /// Single-pass adaptive clustering with auto K-selection
    pub fn fit(&mut self, rows: &[Vec<f64>]) {
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
    fn process_point(&mut self, idx: usize, point: &[f64]) {
        if self.centroids.is_empty() {
            // First point: create first centroid
            self.centroids.push(KalmanCentroid::new(
                point,
                self.process_noise,
                self.measurement_noise,
            ));
            self.assignments[idx] = Some(0);
            return;
        }

        // Find nearest centroid by Mahalanobis distance
        let (best_idx, best_dist_sq) = self.find_nearest_mahalanobis(point);

        // Decision: assign vs. create new
        if best_dist_sq < self.split_threshold_mahal {
            // ASSIGN to existing cluster
            self.centroids[best_idx].update(point);
            self.assignments[idx] = Some(best_idx);
        } else if self.centroids.len() < self.max_k {
            // CREATE new cluster
            let new_idx = self.centroids.len();
            self.centroids.push(KalmanCentroid::new(
                point,
                self.process_noise,
                self.measurement_noise,
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
    fn find_nearest_mahalanobis(&self, point: &[f64]) -> (usize, f64) {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.mahalanobis_distance_sq(point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .unwrap_or((0, f64::INFINITY))
    }

    /// Auto-calibrate split threshold from initial sample variance
    fn auto_calibrate_threshold(&mut self, sample: &[Vec<f64>]) {
        let dim = sample[0].len();

        // Compute global variance across sample
        let mean: Vec<f64> = (0..dim)
            .map(|j| sample.iter().map(|row| row[j]).sum::<f64>() / sample.len() as f64)
            .collect();

        let variance: Vec<f64> = (0..dim)
            .map(|j| {
                sample
                    .iter()
                    .map(|row| (row[j] - mean[j]).powi(2))
                    .sum::<f64>()
                    / sample.len() as f64
            })
            .collect();

        let mean_variance = variance.iter().sum::<f64>() / dim as f64;

        // Set threshold to 3σ in standardized space (99% confidence interval)
        // For F dimensions with unit variance: χ²(F, 0.99) ≈ F + 2√(2F)
        self.split_threshold_mahal = (dim as f64 + 2.0 * (2.0 * dim as f64).sqrt()) / mean_variance;

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
        let distances: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..n)
                    .map(|j| {
                        let dist = self.centroids[i].bhattacharyya_distance(&self.centroids[j]);
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

            let c_i = &self.centroids[i];
            let c_j = &self.centroids[j];
            let total_count = c_i.count + c_j.count;
            let w_i = c_i.count as f64 / total_count as f64;
            let w_j = c_j.count as f64 / total_count as f64;

            // Weighted mean of means
            let merged_mean: Vec<f64> = c_i
                .mean
                .iter()
                .zip(&c_j.mean)
                .map(|(m_i, m_j)| w_i * m_i + w_j * m_j)
                .collect();

            // Weighted variance (pooled)
            let merged_variance: Vec<f64> = c_i
                .variance
                .iter()
                .zip(&c_j.variance)
                .map(|(v_i, v_j)| w_i * v_i + w_j * v_j)
                .collect();

            // Update centroid i with merged state
            self.centroids[i].mean = merged_mean;
            self.centroids[i].variance = merged_variance;
            self.centroids[i].count = total_count;

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

    /// Export centroids as DenseMatrix for ArrowSpace
    pub fn export_centroids(&self) -> DenseMatrix<f64> {
        let n = self.centroids.len();
        let f = self.centroids[0].mean.len();
        let flat: Vec<f64> = self.centroids.iter().flat_map(|c| c.mean.clone()).collect();
        DenseMatrix::from_iterator(flat.into_iter(), n, f, 1)
    }
}
