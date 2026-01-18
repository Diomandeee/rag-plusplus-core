//! Outcome Statistics with Welford's Online Algorithm
//!
//! Provides numerically stable running statistics for outcome dimensions.
//!
//! # Invariants
//!
//! - INV-002: Variance is always non-negative
//! - All operations are O(dim) where dim is the outcome dimension
//!
//! # Example
//!
//! ```
//! use rag_plusplus_core::OutcomeStats;
//!
//! let mut stats = OutcomeStats::new(3);
//! stats.update(&[0.8, 0.9, 0.7]);
//! stats.update(&[0.85, 0.88, 0.75]);
//!
//! assert_eq!(stats.count(), 2);
//! assert!(stats.mean().is_some());
//! ```

/// Running statistics using Welford's online algorithm.
///
/// Maintains count, mean, and M2 (sum of squared differences) for
/// numerically stable variance computation.
#[derive(Debug, Clone)]
pub struct OutcomeStats {
    /// Number of observations
    count: u64,
    /// Running mean per dimension
    mean: Vec<f32>,
    /// Sum of squared differences from mean (for variance)
    m2: Vec<f32>,
    /// Minimum observed values
    min: Vec<f32>,
    /// Maximum observed values
    max: Vec<f32>,
}

impl OutcomeStats {
    /// Create new empty statistics for given dimension.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            min: vec![f32::INFINITY; dim],
            max: vec![f32::NEG_INFINITY; dim],
        }
    }

    /// Update statistics with a new observation (Welford's algorithm).
    ///
    /// # Panics
    ///
    /// Panics if `outcome.len() != self.dim()`.
    pub fn update(&mut self, outcome: &[f32]) {
        assert_eq!(
            outcome.len(),
            self.dim(),
            "Outcome dimension mismatch: expected {}, got {}",
            self.dim(),
            outcome.len()
        );

        self.count += 1;
        let n = self.count as f32;

        for i in 0..self.dim() {
            let x = outcome[i];

            // Welford's update
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x - self.mean[i];
            self.m2[i] += delta * delta2;

            // Min/max tracking
            self.min[i] = self.min[i].min(x);
            self.max[i] = self.max[i].max(x);
        }
    }

    /// Merge two statistics objects (parallel Welford).
    ///
    /// Useful for combining statistics computed in parallel.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        if self.count == 0 {
            return other.clone();
        }
        if other.count == 0 {
            return self.clone();
        }

        assert_eq!(self.dim(), other.dim(), "Dimension mismatch in merge");

        let combined_count = self.count + other.count;
        let mut combined_mean = vec![0.0; self.dim()];
        let mut combined_m2 = vec![0.0; self.dim()];
        let mut combined_min = vec![0.0; self.dim()];
        let mut combined_max = vec![0.0; self.dim()];

        for i in 0..self.dim() {
            let delta = other.mean[i] - self.mean[i];
            combined_mean[i] = self.mean[i]
                + delta * (other.count as f32 / combined_count as f32);
            combined_m2[i] = self.m2[i]
                + other.m2[i]
                + delta * delta
                    * (self.count as f32 * other.count as f32 / combined_count as f32);
            combined_min[i] = self.min[i].min(other.min[i]);
            combined_max[i] = self.max[i].max(other.max[i]);
        }

        Self {
            count: combined_count,
            mean: combined_mean,
            m2: combined_m2,
            min: combined_min,
            max: combined_max,
        }
    }

    /// Update with a single scalar value (1D convenience method).
    pub fn update_scalar(&mut self, value: f64) {
        self.update(&[value as f32]);
    }

    /// Number of observations.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Get the scalar mean (for 1D stats).
    #[must_use]
    pub fn mean_scalar(&self) -> Option<f64> {
        self.mean().map(|m| m[0] as f64)
    }

    /// Get the scalar variance (for 1D stats).
    #[must_use]
    pub fn variance_scalar(&self) -> Option<f64> {
        self.variance().map(|v| v[0] as f64)
    }

    /// Get the scalar std (for 1D stats).
    #[must_use]
    pub fn std_scalar(&self) -> Option<f64> {
        self.std().map(|s| s[0] as f64)
    }

    /// Dimension of outcome vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Current mean estimate (None if no observations).
    #[must_use]
    pub fn mean(&self) -> Option<&[f32]> {
        if self.count > 0 {
            Some(&self.mean)
        } else {
            None
        }
    }

    /// Population variance (None if < 2 observations).
    #[must_use]
    pub fn variance(&self) -> Option<Vec<f32>> {
        if self.count < 2 {
            return None;
        }
        Some(self.m2.iter().map(|m| m / self.count as f32).collect())
    }

    /// Population standard deviation (None if < 2 observations).
    #[must_use]
    pub fn std(&self) -> Option<Vec<f32>> {
        self.variance().map(|v| v.iter().map(|x| x.sqrt()).collect())
    }

    /// Sample variance with Bessel's correction (None if < 2 observations).
    #[must_use]
    pub fn sample_variance(&self) -> Option<Vec<f32>> {
        if self.count < 2 {
            return None;
        }
        Some(
            self.m2
                .iter()
                .map(|m| m / (self.count - 1) as f32)
                .collect(),
        )
    }

    /// Minimum observed values (None if no observations).
    #[must_use]
    pub fn min(&self) -> Option<&[f32]> {
        if self.count > 0 {
            Some(&self.min)
        } else {
            None
        }
    }

    /// Maximum observed values (None if no observations).
    #[must_use]
    pub fn max(&self) -> Option<&[f32]> {
        if self.count > 0 {
            Some(&self.max)
        } else {
            None
        }
    }

    /// Compute confidence interval for the mean.
    ///
    /// Uses t-distribution for small samples (< 30), normal for large.
    /// Returns (lower, upper) bounds.
    #[must_use]
    pub fn confidence_interval(&self, confidence: f32) -> Option<(Vec<f32>, Vec<f32>)> {
        if self.count < 2 {
            return None;
        }

        let std = self.std()?;
        let std_err: Vec<f32> = std.iter().map(|s| s / (self.count as f32).sqrt()).collect();

        // Approximate t-value (use 1.96 for 95% CI with large n)
        let t_val = if self.count < 30 {
            // Rough approximation for small samples
            2.0 + 1.0 / (self.count as f32).sqrt()
        } else {
            // Normal approximation
            match confidence {
                c if (c - 0.90).abs() < 0.01 => 1.645,
                c if (c - 0.95).abs() < 0.01 => 1.96,
                c if (c - 0.99).abs() < 0.01 => 2.576,
                _ => 1.96, // Default to 95%
            }
        };

        let lower: Vec<f32> = self
            .mean
            .iter()
            .zip(&std_err)
            .map(|(m, se)| m - t_val * se)
            .collect();
        let upper: Vec<f32> = self
            .mean
            .iter()
            .zip(&std_err)
            .map(|(m, se)| m + t_val * se)
            .collect();

        Some((lower, upper))
    }
}

impl Default for OutcomeStats {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_stats() {
        let stats = OutcomeStats::new(3);
        assert_eq!(stats.count(), 0);
        assert!(stats.mean().is_none());
        assert!(stats.variance().is_none());
    }

    #[test]
    fn test_single_update() {
        let mut stats = OutcomeStats::new(3);
        stats.update(&[1.0, 2.0, 3.0]);

        assert_eq!(stats.count(), 1);
        assert_eq!(stats.mean(), Some([1.0, 2.0, 3.0].as_slice()));
        assert!(stats.variance().is_none()); // Need 2+ observations
    }

    #[test]
    fn test_multiple_updates() {
        let mut stats = OutcomeStats::new(2);
        stats.update(&[1.0, 2.0]);
        stats.update(&[3.0, 4.0]);
        stats.update(&[5.0, 6.0]);

        assert_eq!(stats.count(), 3);
        let mean = stats.mean().unwrap();
        assert!((mean[0] - 3.0).abs() < 1e-6);
        assert!((mean[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge() {
        let mut stats1 = OutcomeStats::new(2);
        stats1.update(&[1.0, 2.0]);
        stats1.update(&[2.0, 3.0]);

        let mut stats2 = OutcomeStats::new(2);
        stats2.update(&[3.0, 4.0]);
        stats2.update(&[4.0, 5.0]);

        let merged = stats1.merge(&stats2);
        assert_eq!(merged.count(), 4);

        let mean = merged.mean().unwrap();
        assert!((mean[0] - 2.5).abs() < 1e-6);
        assert!((mean[1] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with large values that would overflow naive algorithm
        let mut stats = OutcomeStats::new(1);
        let base = 1e9_f32;

        for i in 0..1000 {
            stats.update(&[base + (i as f32) * 0.001]);
        }

        let mean = stats.mean().unwrap()[0];
        assert!((mean - base).abs() < 1.0); // Should be close to base
        
        let var = stats.variance().unwrap()[0];
        assert!(var >= 0.0); // Variance must be non-negative (INV-002)
    }

    #[test]
    fn test_min_max() {
        let mut stats = OutcomeStats::new(2);
        stats.update(&[1.0, 5.0]);
        stats.update(&[3.0, 2.0]);
        stats.update(&[2.0, 8.0]);

        assert_eq!(stats.min(), Some([1.0, 2.0].as_slice()));
        assert_eq!(stats.max(), Some([3.0, 8.0].as_slice()));
    }
}
