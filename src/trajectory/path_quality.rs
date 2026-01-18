//! Path Quality Computation
//!
//! Implements TPO (Topological Preference Optimization) path quality scoring
//! for enhanced salience computation. Quality measures how "good" a conversation
//! path is, which helps weight turns within that path.
//!
//! # Path Quality Formula
//!
//! ```text
//! Q(P) = α·L(P) + β·T(P) + γ·S(P) + δ·C(P)
//! ```
//!
//! | Factor | Formula | Meaning | Default Weight |
//! |--------|---------|---------|----------------|
//! | L(P) | 1 - var(depth_changes) | Linearity - smooth depth progression | α = 0.25 |
//! | T(P) | terminal_node_score | Terminal quality - how path ends | β = 0.30 |
//! | S(P) | mean(homogeneity) | Semantic coherence along path | γ = 0.25 |
//! | C(P) | path_length / max_length | Completion - path reaches conclusion | δ = 0.20 |
//!
//! # Usage
//!
//! ```
//! use rag_plusplus_core::trajectory::{PathQuality, PathQualityWeights, PathQualityFactors};
//!
//! // Compute factors for a path
//! let factors = PathQualityFactors::from_path_data(
//!     &[1, 2, 3, 4],           // depths along path
//!     &[0.9, 0.85, 0.8, 0.75], // homogeneity values
//!     5,                        // max possible depth
//!     true,                     // is terminal
//!     0.8,                      // terminal score if terminal
//! );
//!
//! // Compute quality score with default weights
//! let quality = factors.compute_quality(&PathQualityWeights::default());
//!
//! // Use for salience enhancement
//! let base_salience = 0.5;
//! let enhanced_salience = PathQuality::enhance_salience(base_salience, quality, 0.3);
//! ```

/// Weights for path quality components.
///
/// Default weights from TPO empirical analysis:
/// - α (linearity): 0.25 - Smooth depth progression
/// - β (terminal): 0.30 - How path ends (conclusions matter more)
/// - γ (coherence): 0.25 - Semantic consistency
/// - δ (completion): 0.20 - Path completeness
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathQualityWeights {
    /// Weight for linearity factor (α)
    pub alpha: f32,
    /// Weight for terminal quality factor (β)
    pub beta: f32,
    /// Weight for coherence factor (γ)
    pub gamma: f32,
    /// Weight for completion factor (δ)
    pub delta: f32,
}

impl Default for PathQualityWeights {
    fn default() -> Self {
        Self {
            alpha: 0.25,
            beta: 0.30,
            gamma: 0.25,
            delta: 0.20,
        }
    }
}

impl PathQualityWeights {
    /// Create custom weights.
    ///
    /// Weights should sum to ~1.0 for normalized output.
    pub fn new(alpha: f32, beta: f32, gamma: f32, delta: f32) -> Self {
        Self { alpha, beta, gamma, delta }
    }

    /// Weights emphasizing terminal quality (for synthesis/conclusion paths).
    pub fn terminal_focused() -> Self {
        Self {
            alpha: 0.15,
            beta: 0.50,
            gamma: 0.20,
            delta: 0.15,
        }
    }

    /// Weights emphasizing coherence (for focused, consistent paths).
    pub fn coherence_focused() -> Self {
        Self {
            alpha: 0.20,
            beta: 0.20,
            gamma: 0.45,
            delta: 0.15,
        }
    }

    /// Weights emphasizing completion (for thorough, complete paths).
    pub fn completion_focused() -> Self {
        Self {
            alpha: 0.20,
            beta: 0.25,
            gamma: 0.15,
            delta: 0.40,
        }
    }
}

/// Computed path quality factors.
///
/// Each factor is in [0, 1] where 1 is best.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathQualityFactors {
    /// L(P): Linearity - smooth depth progression
    /// 1.0 = perfectly linear (depth increases by 1 each step)
    /// 0.0 = highly erratic depth changes
    pub linearity: f32,

    /// T(P): Terminal quality - how well the path ends
    /// 1.0 = terminates at a high-quality conclusion
    /// 0.5 = non-terminal or neutral ending
    /// 0.0 = poor ending (abandoned, error)
    pub terminal_score: f32,

    /// S(P): Semantic coherence - average homogeneity along path
    /// 1.0 = all turns highly similar to their parents
    /// 0.0 = no semantic continuity
    pub coherence: f32,

    /// C(P): Completion - how complete the path is
    /// 1.0 = reached maximum expected depth
    /// 0.0 = very short/incomplete path
    pub completion: f32,
}

impl PathQualityFactors {
    /// Create factors directly.
    pub fn new(linearity: f32, terminal_score: f32, coherence: f32, completion: f32) -> Self {
        Self {
            linearity: linearity.clamp(0.0, 1.0),
            terminal_score: terminal_score.clamp(0.0, 1.0),
            coherence: coherence.clamp(0.0, 1.0),
            completion: completion.clamp(0.0, 1.0),
        }
    }

    /// Compute factors from path data.
    ///
    /// # Arguments
    ///
    /// * `depths` - Depth values along the path
    /// * `homogeneities` - Homogeneity values along the path
    /// * `max_depth` - Maximum possible depth in the trajectory
    /// * `is_terminal` - Whether this path ends at a terminal node
    /// * `terminal_quality` - Quality score of the terminal node [0, 1]
    pub fn from_path_data(
        depths: &[u32],
        homogeneities: &[f32],
        max_depth: u32,
        is_terminal: bool,
        terminal_quality: f32,
    ) -> Self {
        let linearity = Self::compute_linearity(depths);
        let terminal_score = if is_terminal { terminal_quality } else { 0.5 };
        let coherence = Self::compute_coherence(homogeneities);
        let completion = Self::compute_completion(depths.len(), max_depth as usize);

        Self::new(linearity, terminal_score, coherence, completion)
    }

    /// Compute linearity from depth sequence.
    ///
    /// Uses 1 - normalized_variance of depth changes.
    fn compute_linearity(depths: &[u32]) -> f32 {
        if depths.len() < 2 {
            return 1.0; // Single node is perfectly "linear"
        }

        // Compute depth changes
        let changes: Vec<i32> = depths.windows(2)
            .map(|w| w[1] as i32 - w[0] as i32)
            .collect();

        // Compute variance of changes
        let n = changes.len() as f32;
        let mean: f32 = changes.iter().map(|&c| c as f32).sum::<f32>() / n;
        let variance: f32 = changes.iter()
            .map(|&c| (c as f32 - mean).powi(2))
            .sum::<f32>() / n;

        // Ideal change is 1 (going one level deeper each time)
        // Variance of 0 means perfectly linear
        // Normalize: variance of 1 is "bad", variance of 0 is "good"
        // Max expected variance for erratic paths is ~4-5
        (1.0 - variance / 5.0).clamp(0.0, 1.0)
    }

    /// Compute coherence from homogeneity sequence.
    fn compute_coherence(homogeneities: &[f32]) -> f32 {
        if homogeneities.is_empty() {
            return 0.5; // Neutral for empty
        }

        let sum: f32 = homogeneities.iter().sum();
        sum / homogeneities.len() as f32
    }

    /// Compute completion from path length vs max depth.
    fn compute_completion(path_length: usize, max_depth: usize) -> f32 {
        if max_depth == 0 {
            return 1.0; // Single-node trajectory is complete
        }

        // A path of max_depth + 1 nodes is fully complete
        let expected_length = max_depth + 1;
        (path_length as f32 / expected_length as f32).min(1.0)
    }

    /// Compute weighted quality score.
    ///
    /// Returns value in [0, 1] (approximately, depending on weights).
    #[inline]
    pub fn compute_quality(&self, weights: &PathQualityWeights) -> f32 {
        weights.alpha * self.linearity
            + weights.beta * self.terminal_score
            + weights.gamma * self.coherence
            + weights.delta * self.completion
    }

    /// Compute quality with default weights.
    #[inline]
    pub fn quality(&self) -> f32 {
        self.compute_quality(&PathQualityWeights::default())
    }

    /// Check if this is a high-quality path (quality > 0.7).
    #[inline]
    pub fn is_high_quality(&self) -> bool {
        self.quality() > 0.7
    }

    /// Check if this is a low-quality path (quality < 0.4).
    #[inline]
    pub fn is_low_quality(&self) -> bool {
        self.quality() < 0.4
    }
}

impl Default for PathQualityFactors {
    /// Default neutral quality factors.
    fn default() -> Self {
        Self {
            linearity: 0.5,
            terminal_score: 0.5,
            coherence: 0.5,
            completion: 0.5,
        }
    }
}

/// Path quality utilities.
pub struct PathQuality;

impl PathQuality {
    /// Enhance a base salience score using path quality.
    ///
    /// # Arguments
    ///
    /// * `base_salience` - Original salience score [0, 1]
    /// * `quality` - Path quality score [0, 1]
    /// * `blend` - How much quality affects salience [0, 1]
    ///   - 0.0 = quality has no effect
    ///   - 1.0 = quality fully replaces base salience
    ///
    /// # Returns
    ///
    /// Enhanced salience in [0, 1].
    #[inline]
    pub fn enhance_salience(base_salience: f32, quality: f32, blend: f32) -> f32 {
        let blend = blend.clamp(0.0, 1.0);
        (1.0 - blend) * base_salience + blend * quality
    }

    /// Boost terminal nodes based on path quality.
    ///
    /// Terminal nodes in high-quality paths get boosted more.
    #[inline]
    pub fn terminal_boost(base_salience: f32, quality: f32, is_terminal: bool) -> f32 {
        if is_terminal {
            // Terminal nodes get up to 50% boost based on quality
            base_salience + 0.5 * quality * (1.0 - base_salience)
        } else {
            base_salience
        }
    }

    /// Compute path quality for a sequence of episodes.
    ///
    /// # Arguments
    ///
    /// * `depths` - Depth of each episode
    /// * `homogeneities` - Homogeneity (semantic similarity to parent) of each episode
    /// * `max_depth` - Maximum depth in the full trajectory
    /// * `is_terminal` - Whether the last episode is a terminal node
    /// * `terminal_quality` - Quality score for terminal (use feedback if available)
    pub fn compute(
        depths: &[u32],
        homogeneities: &[f32],
        max_depth: u32,
        is_terminal: bool,
        terminal_quality: f32,
    ) -> PathQualityFactors {
        PathQualityFactors::from_path_data(
            depths,
            homogeneities,
            max_depth,
            is_terminal,
            terminal_quality,
        )
    }

    /// Estimate terminal quality from phase.
    ///
    /// Synthesis and Planning phases typically end better than Debugging.
    pub fn terminal_quality_from_phase(phase: Option<&str>) -> f32 {
        match phase {
            Some("synthesis") => 0.9,    // Conclusions are high quality
            Some("planning") => 0.85,    // Plans are good endings
            Some("consolidation") => 0.7, // Building understanding
            Some("exploration") => 0.5,  // Neutral - still exploring
            Some("debugging") => 0.4,    // Often indicates problems
            None => 0.5,                 // Unknown phase
            _ => 0.5,
        }
    }

    /// Estimate terminal quality from feedback.
    pub fn terminal_quality_from_feedback(has_thumbs_up: bool, has_thumbs_down: bool) -> f32 {
        if has_thumbs_up {
            0.95
        } else if has_thumbs_down {
            0.1
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let w = PathQualityWeights::default();
        let sum = w.alpha + w.beta + w.gamma + w.delta;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_terminal_focused_weights() {
        let w = PathQualityWeights::terminal_focused();
        assert!(w.beta > w.alpha);
        assert!(w.beta > w.gamma);
        assert!(w.beta > w.delta);
    }

    #[test]
    fn test_factors_new() {
        let f = PathQualityFactors::new(0.8, 0.7, 0.6, 0.5);
        assert!((f.linearity - 0.8).abs() < 1e-6);
        assert!((f.terminal_score - 0.7).abs() < 1e-6);
        assert!((f.coherence - 0.6).abs() < 1e-6);
        assert!((f.completion - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_factors_clamped() {
        let f = PathQualityFactors::new(1.5, -0.5, 0.5, 0.5);
        assert!((f.linearity - 1.0).abs() < 1e-6);
        assert!((f.terminal_score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_linearity_perfect() {
        // Perfect linear progression: 0, 1, 2, 3, 4
        let depths = vec![0, 1, 2, 3, 4];
        let linearity = PathQualityFactors::compute_linearity(&depths);
        // All changes are +1, variance = 0, so linearity = 1.0
        assert!((linearity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_linearity_erratic() {
        // Erratic: 0, 5, 1, 4, 2
        let depths = vec![0, 5, 1, 4, 2];
        let linearity = PathQualityFactors::compute_linearity(&depths);
        // High variance, so linearity should be low
        assert!(linearity < 0.5);
    }

    #[test]
    fn test_compute_coherence() {
        let homogeneities = vec![0.9, 0.8, 0.7, 0.6];
        let coherence = PathQualityFactors::compute_coherence(&homogeneities);
        // Mean = (0.9 + 0.8 + 0.7 + 0.6) / 4 = 0.75
        assert!((coherence - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_compute_completion() {
        // Path length 5, max depth 4 (so full path would be 5 nodes)
        let completion = PathQualityFactors::compute_completion(5, 4);
        assert!((completion - 1.0).abs() < 1e-6);

        // Path length 3, max depth 4 (3/5 = 0.6)
        let completion = PathQualityFactors::compute_completion(3, 4);
        assert!((completion - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_from_path_data() {
        let depths = vec![0, 1, 2, 3];
        let homogeneities = vec![1.0, 0.9, 0.8, 0.7];
        let max_depth = 5;

        let factors = PathQualityFactors::from_path_data(
            &depths,
            &homogeneities,
            max_depth,
            true,
            0.9,
        );

        // Linearity should be high (linear progression)
        assert!(factors.linearity > 0.9);

        // Terminal score is 0.9 (as given)
        assert!((factors.terminal_score - 0.9).abs() < 1e-6);

        // Coherence = mean of homogeneities = 0.85
        assert!((factors.coherence - 0.85).abs() < 1e-6);

        // Completion = 4 / 6 ≈ 0.67
        assert!(factors.completion > 0.6);
        assert!(factors.completion < 0.7);
    }

    #[test]
    fn test_compute_quality() {
        let factors = PathQualityFactors::new(0.8, 0.9, 0.7, 0.6);
        let weights = PathQualityWeights::default();

        // Q = 0.25*0.8 + 0.30*0.9 + 0.25*0.7 + 0.20*0.6
        //   = 0.2 + 0.27 + 0.175 + 0.12 = 0.765
        let quality = factors.compute_quality(&weights);
        assert!((quality - 0.765).abs() < 1e-5);
    }

    #[test]
    fn test_is_high_quality() {
        let high = PathQualityFactors::new(0.9, 0.9, 0.9, 0.9);
        let low = PathQualityFactors::new(0.2, 0.2, 0.2, 0.2);

        assert!(high.is_high_quality());
        assert!(!low.is_high_quality());
        assert!(low.is_low_quality());
        assert!(!high.is_low_quality());
    }

    #[test]
    fn test_enhance_salience() {
        let base = 0.5;
        let quality = 0.8;

        // No blend - original salience
        assert!((PathQuality::enhance_salience(base, quality, 0.0) - 0.5).abs() < 1e-6);

        // Full blend - quality replaces salience
        assert!((PathQuality::enhance_salience(base, quality, 1.0) - 0.8).abs() < 1e-6);

        // 50% blend
        assert!((PathQuality::enhance_salience(base, quality, 0.5) - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_terminal_boost() {
        let base = 0.6;
        let quality = 0.8;

        // Non-terminal - no boost
        let non_terminal = PathQuality::terminal_boost(base, quality, false);
        assert!((non_terminal - 0.6).abs() < 1e-6);

        // Terminal - gets boosted
        let terminal = PathQuality::terminal_boost(base, quality, true);
        // boost = 0.5 * 0.8 * 0.4 = 0.16
        // result = 0.6 + 0.16 = 0.76
        assert!((terminal - 0.76).abs() < 1e-6);
    }

    #[test]
    fn test_terminal_quality_from_phase() {
        assert!(PathQuality::terminal_quality_from_phase(Some("synthesis")) > 0.8);
        assert!(PathQuality::terminal_quality_from_phase(Some("debugging")) < 0.5);
        assert!((PathQuality::terminal_quality_from_phase(None) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_terminal_quality_from_feedback() {
        assert!(PathQuality::terminal_quality_from_feedback(true, false) > 0.9);
        assert!(PathQuality::terminal_quality_from_feedback(false, true) < 0.2);
        assert!((PathQuality::terminal_quality_from_feedback(false, false) - 0.5).abs() < 1e-6);
    }
}
