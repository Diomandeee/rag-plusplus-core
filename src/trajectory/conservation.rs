//! Conservation Metrics for Bounded Forgetting
//!
//! Provides metrics to validate that memory operations preserve important
//! properties. Inspired by RCP's conservation laws.
//!
//! # Conservation Laws
//!
//! | Law | Formula | Meaning |
//! |-----|---------|---------|
//! | Magnitude | ‖C‖ = const | Context doesn't disappear |
//! | Energy | E = ½Σᵢⱼ aᵢaⱼ cos(eᵢ, eⱼ) | Attention capacity conserved |
//! | Information | H = -Σ aᵢ log(aᵢ) | Shannon entropy of attention |
//!
//! # Usage
//!
//! ```
//! use rag_plusplus_core::trajectory::ConservationMetrics;
//!
//! // Example embeddings (3 vectors of dimension 4)
//! let embeddings_before: Vec<&[f32]> = vec![
//!     &[1.0, 0.0, 0.0, 0.0],
//!     &[0.0, 1.0, 0.0, 0.0],
//!     &[0.0, 0.0, 1.0, 0.0],
//! ];
//! let attention_before = vec![0.5, 0.3, 0.2];
//!
//! let embeddings_after: Vec<&[f32]> = vec![
//!     &[0.9, 0.1, 0.0, 0.0],
//!     &[0.1, 0.9, 0.0, 0.0],
//!     &[0.0, 0.0, 1.0, 0.0],
//! ];
//! let attention_after = vec![0.5, 0.3, 0.2];
//!
//! // Compute metrics before and after an operation
//! let before = ConservationMetrics::compute(&embeddings_before, &attention_before);
//! let after = ConservationMetrics::compute(&embeddings_after, &attention_after);
//!
//! // Check if conservation is preserved
//! if before.is_conserved(&after, 0.1) {
//!     println!("Operation preserved conservation laws");
//! } else {
//!     println!("Warning: Conservation violation detected");
//! }
//! ```

use crate::distance::{cosine_similarity_fast, norm_fast};

/// Conservation metrics for a set of embeddings with attention weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservationMetrics {
    /// Total weighted magnitude: Σᵢ aᵢ ‖eᵢ‖
    pub magnitude: f32,
    /// Attention energy: ½Σᵢⱼ aᵢaⱼ cos(eᵢ, eⱼ)
    pub energy: f32,
    /// Shannon entropy: -Σᵢ aᵢ log(aᵢ)
    pub information: f32,
}

impl ConservationMetrics {
    /// Compute conservation metrics for a set of embeddings and attention weights.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Slice of embedding slices
    /// * `attention` - Attention weights for each embedding (must sum to 1)
    ///
    /// # Returns
    ///
    /// ConservationMetrics with magnitude, energy, and information.
    pub fn compute(embeddings: &[&[f32]], attention: &[f32]) -> Self {
        assert_eq!(embeddings.len(), attention.len(), "Embeddings and attention must have same length");

        if embeddings.is_empty() {
            return Self {
                magnitude: 0.0,
                energy: 0.0,
                information: 0.0,
            };
        }

        // Magnitude: Σᵢ aᵢ ‖eᵢ‖
        let magnitude: f32 = embeddings.iter()
            .zip(attention.iter())
            .map(|(e, a)| a * norm_fast(e))
            .sum();

        // Energy: ½Σᵢⱼ aᵢaⱼ cos(eᵢ, eⱼ)
        let mut energy = 0.0_f32;
        for (i, ei) in embeddings.iter().enumerate() {
            for (j, ej) in embeddings.iter().enumerate() {
                energy += attention[i] * attention[j] * cosine_similarity_fast(ei, ej);
            }
        }
        energy *= 0.5;

        // Information: -Σᵢ aᵢ log(aᵢ)
        let information: f32 = -attention.iter()
            .filter(|&&a| a > 1e-10)
            .map(|a| a * a.ln())
            .sum::<f32>();

        Self {
            magnitude,
            energy,
            information,
        }
    }

    /// Compute metrics from owned vectors.
    pub fn from_vecs(embeddings: &[Vec<f32>], attention: &[f32]) -> Self {
        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        Self::compute(&refs, attention)
    }

    /// Check if conservation is preserved within tolerance.
    ///
    /// # Arguments
    ///
    /// * `other` - Metrics after an operation
    /// * `tolerance` - Maximum allowed difference
    ///
    /// # Returns
    ///
    /// True if magnitude and energy are conserved within tolerance.
    #[inline]
    pub fn is_conserved(&self, other: &Self, tolerance: f32) -> bool {
        (self.magnitude - other.magnitude).abs() < tolerance
            && (self.energy - other.energy).abs() < tolerance
    }

    /// Check if all three metrics are conserved.
    #[inline]
    pub fn is_fully_conserved(&self, other: &Self, tolerance: f32) -> bool {
        self.is_conserved(other, tolerance)
            && (self.information - other.information).abs() < tolerance
    }

    /// Compute the conservation violation (distance from conservation).
    pub fn violation(&self, other: &Self) -> ConservationViolation {
        ConservationViolation {
            magnitude_delta: (self.magnitude - other.magnitude).abs(),
            energy_delta: (self.energy - other.energy).abs(),
            information_delta: (self.information - other.information).abs(),
        }
    }

    /// Create metrics for a uniform attention distribution.
    pub fn uniform(embeddings: &[&[f32]]) -> Self {
        if embeddings.is_empty() {
            return Self {
                magnitude: 0.0,
                energy: 0.0,
                information: 0.0,
            };
        }

        let n = embeddings.len();
        let attention: Vec<f32> = vec![1.0 / n as f32; n];
        Self::compute(embeddings, &attention)
    }

    /// Maximum possible entropy for n items.
    #[inline]
    pub fn max_entropy(n: usize) -> f32 {
        if n <= 1 {
            0.0
        } else {
            (n as f32).ln()
        }
    }

    /// Normalized entropy (0 = concentrated, 1 = uniform).
    #[inline]
    pub fn normalized_entropy(&self, n: usize) -> f32 {
        let max = Self::max_entropy(n);
        if max > 0.0 {
            self.information / max
        } else {
            0.0
        }
    }
}

impl Default for ConservationMetrics {
    fn default() -> Self {
        Self {
            magnitude: 0.0,
            energy: 0.0,
            information: 0.0,
        }
    }
}

/// Details of a conservation violation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservationViolation {
    /// Absolute difference in magnitude
    pub magnitude_delta: f32,
    /// Absolute difference in energy
    pub energy_delta: f32,
    /// Absolute difference in information
    pub information_delta: f32,
}

impl ConservationViolation {
    /// Total violation magnitude (L1 norm).
    #[inline]
    pub fn total(&self) -> f32 {
        self.magnitude_delta + self.energy_delta + self.information_delta
    }

    /// Maximum single violation.
    #[inline]
    pub fn max(&self) -> f32 {
        self.magnitude_delta
            .max(self.energy_delta)
            .max(self.information_delta)
    }

    /// Check if violation is within tolerance.
    #[inline]
    pub fn is_acceptable(&self, tolerance: f32) -> bool {
        self.max() < tolerance
    }
}

/// Configuration for conservation checking.
#[derive(Debug, Clone)]
pub struct ConservationConfig {
    /// Tolerance for magnitude conservation
    pub magnitude_tolerance: f32,
    /// Tolerance for energy conservation
    pub energy_tolerance: f32,
    /// Tolerance for information conservation
    pub information_tolerance: f32,
    /// Whether to enforce conservation (fail on violation)
    pub strict: bool,
}

impl Default for ConservationConfig {
    fn default() -> Self {
        Self {
            magnitude_tolerance: 0.01,
            energy_tolerance: 0.01,
            information_tolerance: 0.1, // Information can vary more
            strict: false,
        }
    }
}

impl ConservationConfig {
    /// Create a strict configuration.
    pub fn strict() -> Self {
        Self {
            magnitude_tolerance: 0.001,
            energy_tolerance: 0.001,
            information_tolerance: 0.01,
            strict: true,
        }
    }

    /// Check if violation is acceptable according to config.
    pub fn is_acceptable(&self, violation: &ConservationViolation) -> bool {
        violation.magnitude_delta < self.magnitude_tolerance
            && violation.energy_delta < self.energy_tolerance
            && violation.information_delta < self.information_tolerance
    }
}

/// Track conservation over time.
#[derive(Debug, Clone)]
pub struct ConservationTracker {
    history: Vec<ConservationMetrics>,
    config: ConservationConfig,
}

impl ConservationTracker {
    /// Create a new tracker.
    pub fn new(config: ConservationConfig) -> Self {
        Self {
            history: Vec::new(),
            config,
        }
    }

    /// Record a conservation snapshot.
    pub fn record(&mut self, metrics: ConservationMetrics) {
        self.history.push(metrics);
    }

    /// Get the most recent metrics.
    pub fn current(&self) -> Option<&ConservationMetrics> {
        self.history.last()
    }

    /// Get the initial metrics.
    pub fn initial(&self) -> Option<&ConservationMetrics> {
        self.history.first()
    }

    /// Check if conservation is maintained from initial state.
    pub fn is_conserved_from_initial(&self) -> Option<bool> {
        let initial = self.initial()?;
        let current = self.current()?;
        Some(self.config.is_acceptable(&initial.violation(current)))
    }

    /// Get the total drift from initial state.
    pub fn total_drift(&self) -> Option<ConservationViolation> {
        let initial = self.initial()?;
        let current = self.current()?;
        Some(initial.violation(current))
    }

    /// Get all recorded metrics.
    pub fn history(&self) -> &[ConservationMetrics] {
        &self.history
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Compute the attention-weighted centroid of embeddings.
///
/// Useful for checking if the "center of mass" is preserved.
pub fn weighted_centroid(embeddings: &[&[f32]], attention: &[f32]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let mut centroid = vec![0.0_f32; dim];

    for (e, &a) in embeddings.iter().zip(attention.iter()) {
        for (c, &v) in centroid.iter_mut().zip(e.iter()) {
            *c += a * v;
        }
    }

    centroid
}

/// Compute the attention-weighted covariance matrix (flattened).
///
/// Returns the upper triangle of the covariance matrix as a flat vector.
pub fn weighted_covariance(embeddings: &[&[f32]], attention: &[f32]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let centroid = weighted_centroid(embeddings, attention);

    // Compute upper triangle of covariance
    let n_cov = (dim * (dim + 1)) / 2;
    let mut cov = vec![0.0_f32; n_cov];

    for (e, &a) in embeddings.iter().zip(attention.iter()) {
        let mut idx = 0;
        for i in 0..dim {
            for j in i..dim {
                let diff_i = e[i] - centroid[i];
                let diff_j = e[j] - centroid[j];
                cov[idx] += a * diff_i * diff_j;
                idx += 1;
            }
        }
    }

    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn test_compute_metrics() {
        let embeddings = make_embeddings();
        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let attention = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let metrics = ConservationMetrics::compute(&refs, &attention);

        // Magnitude: (1/3) * 1.0 * 3 = 1.0
        assert!((metrics.magnitude - 1.0).abs() < 1e-5);

        // Information: -3 * (1/3) * ln(1/3) = ln(3)
        let expected_info = 3.0_f32.ln();
        assert!((metrics.information - expected_info).abs() < 1e-5);
    }

    #[test]
    fn test_is_conserved() {
        let embeddings = make_embeddings();
        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
        let attention = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let m1 = ConservationMetrics::compute(&refs, &attention);
        let m2 = ConservationMetrics::compute(&refs, &attention);

        assert!(m1.is_conserved(&m2, 0.01));
    }

    #[test]
    fn test_conservation_violation() {
        let m1 = ConservationMetrics {
            magnitude: 1.0,
            energy: 0.5,
            information: 1.0,
        };

        let m2 = ConservationMetrics {
            magnitude: 1.1,
            energy: 0.6,
            information: 0.9,
        };

        let violation = m1.violation(&m2);
        assert!((violation.magnitude_delta - 0.1).abs() < 1e-5);
        assert!((violation.energy_delta - 0.1).abs() < 1e-5);
        assert!((violation.information_delta - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_max_entropy() {
        // ln(1) = 0
        assert!((ConservationMetrics::max_entropy(1) - 0.0).abs() < 1e-5);

        // ln(2)
        assert!((ConservationMetrics::max_entropy(2) - 2.0_f32.ln()).abs() < 1e-5);

        // ln(10)
        assert!((ConservationMetrics::max_entropy(10) - 10.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_normalized_entropy() {
        let embeddings = make_embeddings();
        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();

        // Uniform distribution should have normalized entropy = 1
        let uniform_attention = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let uniform_metrics = ConservationMetrics::compute(&refs, &uniform_attention);
        assert!((uniform_metrics.normalized_entropy(3) - 1.0).abs() < 1e-5);

        // Concentrated distribution should have lower normalized entropy
        let concentrated = vec![0.9, 0.05, 0.05];
        let concentrated_metrics = ConservationMetrics::compute(&refs, &concentrated);
        assert!(concentrated_metrics.normalized_entropy(3) < 0.5);
    }

    #[test]
    fn test_tracker() {
        let config = ConservationConfig::default();
        let mut tracker = ConservationTracker::new(config);

        let m1 = ConservationMetrics {
            magnitude: 1.0,
            energy: 0.5,
            information: 1.0,
        };

        let m2 = ConservationMetrics {
            magnitude: 1.001,
            energy: 0.501,
            information: 1.01,
        };

        tracker.record(m1);
        tracker.record(m2);

        assert!(tracker.is_conserved_from_initial().unwrap());
        assert_eq!(tracker.history().len(), 2);
    }

    #[test]
    fn test_weighted_centroid() {
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();

        // Equal weights
        let attention = vec![0.5, 0.5];
        let centroid = weighted_centroid(&refs, &attention);
        assert!((centroid[0] - 0.5).abs() < 1e-5);
        assert!((centroid[1] - 0.5).abs() < 1e-5);

        // Unequal weights
        let attention2 = vec![0.8, 0.2];
        let centroid2 = weighted_centroid(&refs, &attention2);
        assert!((centroid2[0] - 0.8).abs() < 1e-5);
        assert!((centroid2[1] - 0.2).abs() < 1e-5);
    }
}
