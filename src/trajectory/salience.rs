//! Salience Scoring System
//!
//! Computes importance weights for episodes, enabling bounded forgetting
//! and prioritized retrieval.
//!
//! Per SPECIFICATION_HARD.md Section 9:
//!
//! ```text
//! salience_score = weighted_sum([
//!     user_feedback,           // thumbs_up = +0.35, thumbs_down = -0.35
//!     phase_transition,        // +0.10 at transitions
//!     downstream_references,   // +0.05 per ref, capped at +0.15
//!     novelty,                 // +0.10 max based on embedding distance
//! ])
//! ```
//!
//! Scores are bounded to [0, 1].
//!
//! # Debug Logging
//!
//! Enable the `salience-debug` feature to log salience distribution statistics
//! for tuning and analysis.

use crate::distance::cosine_similarity;

/// User feedback type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feedback {
    ThumbsUp,
    ThumbsDown,
    None,
}

/// Factors contributing to salience score.
#[derive(Debug, Clone, Default)]
pub struct SalienceFactors {
    /// Episode identifier
    pub turn_id: u64,
    /// User feedback on this episode
    pub feedback: Option<Feedback>,
    /// Whether this is a phase transition point
    pub is_phase_transition: bool,
    /// Number of times referenced by later episodes
    pub reference_count: usize,
    /// Embedding of this episode (for novelty)
    pub embedding: Option<Vec<f32>>,
    /// Episode role
    pub role: String,
    /// Content length
    pub content_length: usize,
}

/// Result of salience computation for an episode.
#[derive(Debug, Clone)]
pub struct TurnSalience {
    pub turn_id: u64,
    pub score: f32,
    pub feedback_contribution: f32,
    pub phase_contribution: f32,
    pub reference_contribution: f32,
    pub novelty_contribution: f32,
}

/// Corpus-level salience statistics.
#[derive(Debug, Clone, Default)]
pub struct CorpusSalienceStats {
    pub total_turns: usize,
    pub mean_salience: f32,
    pub std_salience: f32,
    pub min_salience: f32,
    pub max_salience: f32,
    pub high_salience_count: usize,  // > 0.7
    pub low_salience_count: usize,   // < 0.3
}

/// Configuration for salience scoring.
#[derive(Debug, Clone)]
pub struct SalienceConfig {
    /// Base salience score (neutral)
    pub base_score: f32,
    /// Boost for thumbs_up
    pub feedback_up_boost: f32,
    /// Penalty for thumbs_down
    pub feedback_down_penalty: f32,
    /// Boost for phase transitions
    pub phase_transition_boost: f32,
    /// Boost per reference (capped)
    pub reference_boost_per: f32,
    /// Maximum reference contribution
    pub reference_boost_cap: f32,
    /// Maximum novelty contribution
    pub novelty_boost_max: f32,
    /// Size of recent window for novelty computation
    pub novelty_window_size: usize,
    /// Version string
    pub version: String,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        // Per SPECIFICATION_HARD.md Section 9
        Self {
            base_score: 0.5,
            feedback_up_boost: 0.35,
            feedback_down_penalty: 0.35,
            phase_transition_boost: 0.10,
            reference_boost_per: 0.05,
            reference_boost_cap: 0.15,
            novelty_boost_max: 0.10,
            novelty_window_size: 10,
            version: "v1.0".to_string(),
        }
    }
}

/// Salience scoring engine.
#[derive(Debug, Clone)]
pub struct SalienceScorer {
    config: SalienceConfig,
}

impl SalienceScorer {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: SalienceConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SalienceConfig) -> Self {
        Self { config }
    }

    /// Compute salience for a single episode.
    ///
    /// # Arguments
    ///
    /// * `factors` - Features of the episode
    /// * `recent_embeddings` - Embeddings of recent episodes (for novelty)
    ///
    /// # Returns
    ///
    /// TurnSalience with score and contribution breakdown.
    pub fn score_single(
        &self,
        factors: &SalienceFactors,
        recent_embeddings: Option<&[Vec<f32>]>,
    ) -> TurnSalience {
        let mut score = self.config.base_score;

        // Feedback contribution
        let feedback_contribution = match factors.feedback {
            Some(Feedback::ThumbsUp) => self.config.feedback_up_boost,
            Some(Feedback::ThumbsDown) => -self.config.feedback_down_penalty,
            Some(Feedback::None) | None => 0.0,
        };
        score += feedback_contribution;

        // Phase transition contribution
        let phase_contribution = if factors.is_phase_transition {
            self.config.phase_transition_boost
        } else {
            0.0
        };
        score += phase_contribution;

        // Reference contribution (capped)
        let reference_contribution = (factors.reference_count as f32 * self.config.reference_boost_per)
            .min(self.config.reference_boost_cap);
        score += reference_contribution;

        // Novelty contribution
        let novelty_contribution = self.compute_novelty(factors, recent_embeddings);
        score += novelty_contribution;

        // Clamp to [0, 1]
        score = score.clamp(0.0, 1.0);

        TurnSalience {
            turn_id: factors.turn_id,
            score,
            feedback_contribution,
            phase_contribution,
            reference_contribution,
            novelty_contribution,
        }
    }

    /// Compute novelty based on embedding distance from recent window.
    fn compute_novelty(
        &self,
        factors: &SalienceFactors,
        recent_embeddings: Option<&[Vec<f32>]>,
    ) -> f32 {
        let embedding = match &factors.embedding {
            Some(e) => e,
            None => return 0.0,
        };

        let recent = match recent_embeddings {
            Some(r) if !r.is_empty() => r,
            _ => return self.config.novelty_boost_max, // First episode is maximally novel
        };

        // Compute average similarity to recent episodes
        let similarities: Vec<f32> = recent
            .iter()
            .take(self.config.novelty_window_size)
            .map(|other| cosine_similarity(embedding, other))
            .collect();

        if similarities.is_empty() {
            return self.config.novelty_boost_max;
        }

        let avg_similarity = similarities.iter().sum::<f32>() / similarities.len() as f32;

        // Novelty = 1 - similarity, scaled to max boost
        let novelty = 1.0 - avg_similarity;
        (novelty * self.config.novelty_boost_max).clamp(0.0, self.config.novelty_boost_max)
    }

    /// Compute salience for all episodes in a corpus.
    ///
    /// This is the high-performance batch version that processes
    /// the entire corpus efficiently.
    ///
    /// # Arguments
    ///
    /// * `turns` - All episodes with their factors
    ///
    /// # Returns
    ///
    /// Vector of salience scores for each episode.
    pub fn score_corpus(&self, turns: &[SalienceFactors]) -> Vec<TurnSalience> {
        let mut results = Vec::with_capacity(turns.len());
        let mut recent_embeddings: Vec<&Vec<f32>> = Vec::with_capacity(self.config.novelty_window_size);

        for factors in turns {
            // Build recent embeddings window
            let recent: Vec<Vec<f32>> = recent_embeddings
                .iter()
                .map(|e| (*e).clone())
                .collect();

            let salience = self.score_single(
                factors,
                if recent.is_empty() { None } else { Some(&recent) },
            );

            results.push(salience);

            // Update recent window
            if let Some(ref emb) = factors.embedding {
                recent_embeddings.push(emb);
                if recent_embeddings.len() > self.config.novelty_window_size {
                    recent_embeddings.remove(0);
                }
            }
        }

        // Debug logging for salience distribution tuning
        #[cfg(feature = "salience-debug")]
        self.log_salience_distribution(&results);

        results
    }

    /// Log salience distribution statistics for tuning.
    ///
    /// Only available with `salience-debug` feature enabled.
    #[cfg(feature = "salience-debug")]
    fn log_salience_distribution(&self, saliences: &[TurnSalience]) {
        if saliences.is_empty() {
            return;
        }

        let scores: Vec<f32> = saliences.iter().map(|s| s.score).collect();
        let n = scores.len() as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute percentiles
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p25 = sorted.get((n * 0.25) as usize).copied().unwrap_or(0.0);
        let p50 = sorted.get((n * 0.50) as usize).copied().unwrap_or(0.0);
        let p75 = sorted.get((n * 0.75) as usize).copied().unwrap_or(0.0);

        // Count by range
        let low = scores.iter().filter(|&&s| s < 0.3).count();
        let mid = scores.iter().filter(|&&s| s >= 0.3 && s <= 0.7).count();
        let high = scores.iter().filter(|&&s| s > 0.7).count();

        // Log with tracing
        tracing::debug!(
            n = scores.len(),
            mean = %format!("{:.3}", mean),
            std = %format!("{:.3}", std),
            min = %format!("{:.3}", min),
            max = %format!("{:.3}", max),
            p25 = %format!("{:.3}", p25),
            p50 = %format!("{:.3}", p50),
            p75 = %format!("{:.3}", p75),
            low_count = low,
            mid_count = mid,
            high_count = high,
            "Salience distribution"
        );
    }

    /// Compute corpus-level statistics.
    pub fn compute_stats(&self, saliences: &[TurnSalience]) -> CorpusSalienceStats {
        if saliences.is_empty() {
            return CorpusSalienceStats::default();
        }

        let scores: Vec<f32> = saliences.iter().map(|s| s.score).collect();
        let n = scores.len();

        let mean = scores.iter().sum::<f32>() / n as f32;
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / n as f32;
        let std = variance.sqrt();

        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let high_count = scores.iter().filter(|&&s| s > 0.7).count();
        let low_count = scores.iter().filter(|&&s| s < 0.3).count();

        CorpusSalienceStats {
            total_turns: n,
            mean_salience: mean,
            std_salience: std,
            min_salience: min,
            max_salience: max,
            high_salience_count: high_count,
            low_salience_count: low_count,
        }
    }

    /// Normalize scores to have mean 0.5 and bounded to [0, 1].
    ///
    /// Useful when raw scores cluster too tightly.
    pub fn normalize_scores(&self, saliences: &mut [TurnSalience]) {
        if saliences.is_empty() {
            return;
        }

        let scores: Vec<f32> = saliences.iter().map(|s| s.score).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let std = {
            let variance = scores.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f32>() / scores.len() as f32;
            variance.sqrt().max(0.01) // Avoid division by zero
        };

        // Z-score normalization centered at 0.5
        for s in saliences.iter_mut() {
            let z = (s.score - mean) / std;
            s.score = (0.5 + z * 0.2).clamp(0.0, 1.0);
        }
    }

    /// Get version string.
    #[inline]
    pub fn version(&self) -> &str {
        &self.config.version
    }
}

impl Default for SalienceScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_factors(turn_id: u64, feedback: Option<Feedback>) -> SalienceFactors {
        SalienceFactors {
            turn_id,
            feedback,
            is_phase_transition: false,
            reference_count: 0,
            embedding: None,
            role: "assistant".to_string(),
            content_length: 100,
        }
    }

    #[test]
    fn test_base_score() {
        let scorer = SalienceScorer::new();
        let factors = make_factors(1, None);
        let result = scorer.score_single(&factors, None);

        assert!((result.score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_thumbs_up_boost() {
        let scorer = SalienceScorer::new();
        let factors = make_factors(1, Some(Feedback::ThumbsUp));
        let result = scorer.score_single(&factors, None);

        assert!((result.score - 0.85).abs() < 0.01); // 0.5 + 0.35
        assert!((result.feedback_contribution - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_thumbs_down_penalty() {
        let scorer = SalienceScorer::new();
        let factors = make_factors(1, Some(Feedback::ThumbsDown));
        let result = scorer.score_single(&factors, None);

        assert!((result.score - 0.15).abs() < 0.01); // 0.5 - 0.35
    }

    #[test]
    fn test_phase_transition_boost() {
        let scorer = SalienceScorer::new();
        let mut factors = make_factors(1, None);
        factors.is_phase_transition = true;
        let result = scorer.score_single(&factors, None);

        assert!((result.score - 0.6).abs() < 0.01); // 0.5 + 0.10
        assert!((result.phase_contribution - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_reference_boost_capped() {
        let scorer = SalienceScorer::new();
        let mut factors = make_factors(1, None);
        factors.reference_count = 10; // Would be 0.5 without cap
        let result = scorer.score_single(&factors, None);

        assert!((result.reference_contribution - 0.15).abs() < 0.01); // Capped
        assert!((result.score - 0.65).abs() < 0.01); // 0.5 + 0.15
    }

    #[test]
    fn test_score_clamping() {
        let scorer = SalienceScorer::new();
        let mut factors = make_factors(1, Some(Feedback::ThumbsUp));
        factors.is_phase_transition = true;
        factors.reference_count = 5;
        // Would be 0.5 + 0.35 + 0.10 + 0.15 + novelty = > 1.0
        let result = scorer.score_single(&factors, None);

        assert!(result.score <= 1.0);
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_corpus_scoring() {
        let scorer = SalienceScorer::new();
        let turns = vec![
            make_factors(1, None),
            make_factors(2, Some(Feedback::ThumbsUp)),
            make_factors(3, Some(Feedback::ThumbsDown)),
        ];

        let results = scorer.score_corpus(&turns);
        assert_eq!(results.len(), 3);

        // Verify ordering
        assert!(results[1].score > results[0].score); // ThumbsUp > None
        assert!(results[0].score > results[2].score); // None > ThumbsDown
    }

    #[test]
    fn test_corpus_stats() {
        let scorer = SalienceScorer::new();
        let saliences = vec![
            TurnSalience { turn_id: 1, score: 0.2, feedback_contribution: 0.0, phase_contribution: 0.0, reference_contribution: 0.0, novelty_contribution: 0.0 },
            TurnSalience { turn_id: 2, score: 0.5, feedback_contribution: 0.0, phase_contribution: 0.0, reference_contribution: 0.0, novelty_contribution: 0.0 },
            TurnSalience { turn_id: 3, score: 0.8, feedback_contribution: 0.0, phase_contribution: 0.0, reference_contribution: 0.0, novelty_contribution: 0.0 },
        ];

        let stats = scorer.compute_stats(&saliences);
        assert_eq!(stats.total_turns, 3);
        assert!((stats.mean_salience - 0.5).abs() < 0.01);
        assert!(stats.min_salience < 0.3);
        assert!(stats.max_salience > 0.7);
    }
}
