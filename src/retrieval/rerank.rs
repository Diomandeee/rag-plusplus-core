//! Reranking Module
//!
//! Provides reranking algorithms for retrieved results.
//!
//! # Reranking Strategies
//!
//! - **Outcome-weighted**: Rerank based on historical outcome statistics
//! - **Recency**: Boost more recent records
//! - **MMR (Maximal Marginal Relevance)**: Diversify results
//! - **Composite**: Combine multiple strategies

use crate::retrieval::engine::RetrievedRecord;
use ordered_float::OrderedFloat;

/// Type of reranking algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankerType {
    /// No reranking (keep original order)
    #[default]
    None,
    /// Rerank by outcome statistics (mean + confidence)
    OutcomeWeighted,
    /// Boost recent records
    Recency,
    /// Maximal Marginal Relevance for diversity
    MMR,
    /// Combine outcome and recency
    Composite,
}

/// Reranker configuration.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Reranking strategy
    pub strategy: RerankerType,
    /// Weight for original score (0-1)
    pub original_weight: f32,
    /// Weight for outcome score (0-1)
    pub outcome_weight: f32,
    /// Weight for recency (0-1)
    pub recency_weight: f32,
    /// Recency decay half-life in seconds
    pub recency_half_life: f64,
    /// MMR lambda (0 = pure diversity, 1 = pure relevance)
    pub mmr_lambda: f32,
    /// Minimum sample count for outcome weighting
    pub min_samples: u64,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            strategy: RerankerType::OutcomeWeighted,
            original_weight: 0.5,
            outcome_weight: 0.3,
            recency_weight: 0.2,
            recency_half_life: 86400.0 * 7.0, // 7 days
            mmr_lambda: 0.7,
            min_samples: 3,
        }
    }
}

impl RerankerConfig {
    /// Create new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: RerankerType) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set outcome weight.
    #[must_use]
    pub const fn with_outcome_weight(mut self, weight: f32) -> Self {
        self.outcome_weight = weight;
        self
    }

    /// Set MMR lambda.
    #[must_use]
    pub const fn with_mmr_lambda(mut self, lambda: f32) -> Self {
        self.mmr_lambda = lambda;
        self
    }
}

/// Reranker for improving result ordering.
pub struct Reranker {
    config: RerankerConfig,
}

impl Reranker {
    /// Create a new reranker.
    #[must_use]
    pub fn new(config: RerankerConfig) -> Self {
        Self { config }
    }

    /// Rerank results according to configured strategy.
    #[must_use]
    pub fn rerank(&self, results: Vec<RetrievedRecord>) -> Vec<RetrievedRecord> {
        match self.config.strategy {
            RerankerType::None => results,
            RerankerType::OutcomeWeighted => self.rerank_by_outcome(results),
            RerankerType::Recency => self.rerank_by_recency(results),
            RerankerType::MMR => self.rerank_mmr(results),
            RerankerType::Composite => self.rerank_composite(results),
        }
    }

    /// Rerank by outcome statistics.
    fn rerank_by_outcome(&self, mut results: Vec<RetrievedRecord>) -> Vec<RetrievedRecord> {
        for result in &mut results {
            let outcome_score = self.compute_outcome_score(&result.record);
            result.score = self.config.original_weight * result.score
                + self.config.outcome_weight * outcome_score;
        }

        results.sort_by(|a, b| {
            OrderedFloat(b.score).cmp(&OrderedFloat(a.score))
        });

        results
    }

    /// Compute outcome score for a record.
    fn compute_outcome_score(&self, record: &crate::types::MemoryRecord) -> f32 {
        let stats = &record.stats;

        if stats.count() < self.config.min_samples {
            // Not enough data, use initial outcome
            return record.outcome as f32;
        }

        // Use lower bound of confidence interval for conservative estimate
        // This implements "optimistic pessimism" - we're optimistic about
        // exploring but pessimistic in our estimates
        if let Some((lower, _upper)) = stats.confidence_interval(0.90) {
            // Return first dimension's lower bound
            lower.first().copied().unwrap_or(record.outcome as f32)
        } else {
            record.outcome as f32
        }
    }

    /// Rerank by recency.
    fn rerank_by_recency(&self, mut results: Vec<RetrievedRecord>) -> Vec<RetrievedRecord> {
        let now = current_time_secs();

        for result in &mut results {
            let age_secs = (now - result.record.created_at) as f64;
            let recency_score = self.compute_recency_score(age_secs);

            result.score = self.config.original_weight * result.score
                + self.config.recency_weight * recency_score;
        }

        results.sort_by(|a, b| {
            OrderedFloat(b.score).cmp(&OrderedFloat(a.score))
        });

        results
    }

    /// Compute recency score with exponential decay.
    fn compute_recency_score(&self, age_secs: f64) -> f32 {
        // Exponential decay: score = exp(-age / half_life * ln(2))
        let decay = (-age_secs / self.config.recency_half_life * std::f64::consts::LN_2).exp();
        decay as f32
    }

    /// Rerank using MMR for diversity.
    fn rerank_mmr(&self, results: Vec<RetrievedRecord>) -> Vec<RetrievedRecord> {
        if results.len() <= 1 {
            return results;
        }

        let lambda = self.config.mmr_lambda;
        let mut reranked = Vec::with_capacity(results.len());
        let mut remaining: Vec<_> = results.into_iter().collect();

        // Select first by pure relevance
        remaining.sort_by(|a, b| OrderedFloat(b.score).cmp(&OrderedFloat(a.score)));
        reranked.push(remaining.remove(0));

        // Select remaining by MMR
        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, candidate) in remaining.iter().enumerate() {
                // Relevance term
                let relevance = candidate.score;

                // Diversity term (max similarity to already selected)
                let max_sim = reranked
                    .iter()
                    .map(|r| self.embedding_similarity(&candidate.record.embedding, &r.record.embedding))
                    .fold(0.0f32, f32::max);

                // MMR score
                let mmr = lambda * relevance - (1.0 - lambda) * max_sim;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            reranked.push(remaining.remove(best_idx));
        }

        reranked
    }

    /// Compute cosine similarity between embeddings.
    fn embedding_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Composite reranking (outcome + recency).
    fn rerank_composite(&self, mut results: Vec<RetrievedRecord>) -> Vec<RetrievedRecord> {
        let now = current_time_secs();

        for result in &mut results {
            let outcome_score = self.compute_outcome_score(&result.record);
            let age_secs = (now - result.record.created_at) as f64;
            let recency_score = self.compute_recency_score(age_secs);

            result.score = self.config.original_weight * result.score
                + self.config.outcome_weight * outcome_score
                + self.config.recency_weight * recency_score;
        }

        results.sort_by(|a, b| {
            OrderedFloat(b.score).cmp(&OrderedFloat(a.score))
        });

        results
    }
}

/// Get current time in seconds (Unix epoch).
fn current_time_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::OutcomeStats;
    use crate::types::{MemoryRecord, RecordStatus};

    fn create_test_result(id: &str, score: f32, outcome: f64, age_secs: u64) -> RetrievedRecord {
        let now = current_time_secs();
        let created_at = now.saturating_sub(age_secs);

        RetrievedRecord {
            record: MemoryRecord {
                id: id.into(),
                embedding: vec![1.0, 0.0, 0.0],
                context: format!("Context {id}"),
                outcome,
                metadata: Default::default(),
                created_at,
                status: RecordStatus::Active,
                stats: OutcomeStats::new(1),
            },
            score,
            rank: 0,
            source_index: "test".into(),
        }
    }

    fn create_result_with_stats(id: &str, score: f32, outcomes: &[f64]) -> RetrievedRecord {
        let mut stats = OutcomeStats::new(1);
        for &o in outcomes {
            stats.update_scalar(o);
        }

        RetrievedRecord {
            record: MemoryRecord {
                id: id.into(),
                embedding: vec![1.0, 0.0, 0.0],
                context: format!("Context {id}"),
                outcome: outcomes.first().copied().unwrap_or(0.5),
                metadata: Default::default(),
                created_at: current_time_secs(),
                status: RecordStatus::Active,
                stats,
            },
            score,
            rank: 0,
            source_index: "test".into(),
        }
    }

    #[test]
    fn test_no_reranking() {
        let reranker = Reranker::new(RerankerConfig::new().with_strategy(RerankerType::None));

        let results = vec![
            create_test_result("a", 0.9, 0.5, 0),
            create_test_result("b", 0.8, 0.9, 0),
        ];

        let reranked = reranker.rerank(results);

        assert_eq!(reranked[0].record.id.as_str(), "a");
        assert_eq!(reranked[1].record.id.as_str(), "b");
    }

    #[test]
    fn test_outcome_reranking() {
        let reranker = Reranker::new(
            RerankerConfig::new()
                .with_strategy(RerankerType::OutcomeWeighted)
                .with_outcome_weight(0.8),
        );

        // b has better outcome stats
        let results = vec![
            create_result_with_stats("a", 0.9, &[0.3, 0.4, 0.3, 0.4]),
            create_result_with_stats("b", 0.8, &[0.9, 0.8, 0.9, 0.85]),
        ];

        let reranked = reranker.rerank(results);

        // b should be ranked higher due to better outcomes
        assert_eq!(reranked[0].record.id.as_str(), "b");
    }

    #[test]
    fn test_recency_reranking() {
        let reranker = Reranker::new(
            RerankerConfig::new()
                .with_strategy(RerankerType::Recency),
        );

        let results = vec![
            create_test_result("old", 0.9, 0.5, 86400 * 30), // 30 days old
            create_test_result("new", 0.8, 0.5, 3600),        // 1 hour old
        ];

        let reranked = reranker.rerank(results);

        // new should be ranked higher due to recency
        assert_eq!(reranked[0].record.id.as_str(), "new");
    }

    #[test]
    fn test_mmr_diversity() {
        let reranker = Reranker::new(
            RerankerConfig::new()
                .with_strategy(RerankerType::MMR)
                .with_mmr_lambda(0.5),
        );

        // Create results with similar embeddings
        let mut results = vec![
            RetrievedRecord {
                record: MemoryRecord {
                    id: "a".into(),
                    embedding: vec![1.0, 0.0, 0.0],
                    context: "a".into(),
                    outcome: 0.5,
                    metadata: Default::default(),
                    created_at: 0,
                    status: RecordStatus::Active,
                    stats: OutcomeStats::new(1),
                },
                score: 0.95,
                rank: 0,
                source_index: "test".into(),
            },
            RetrievedRecord {
                record: MemoryRecord {
                    id: "b".into(),
                    embedding: vec![0.99, 0.01, 0.0], // Very similar to a
                    context: "b".into(),
                    outcome: 0.5,
                    metadata: Default::default(),
                    created_at: 0,
                    status: RecordStatus::Active,
                    stats: OutcomeStats::new(1),
                },
                score: 0.9,
                rank: 0,
                source_index: "test".into(),
            },
            RetrievedRecord {
                record: MemoryRecord {
                    id: "c".into(),
                    embedding: vec![0.0, 1.0, 0.0], // Different from a
                    context: "c".into(),
                    outcome: 0.5,
                    metadata: Default::default(),
                    created_at: 0,
                    status: RecordStatus::Active,
                    stats: OutcomeStats::new(1),
                },
                score: 0.85,
                rank: 0,
                source_index: "test".into(),
            },
        ];

        let reranked = reranker.rerank(results);

        // First should still be "a" (highest score)
        assert_eq!(reranked[0].record.id.as_str(), "a");

        // Second should be "c" (diverse) despite lower score
        // because "b" is too similar to "a"
        assert_eq!(reranked[1].record.id.as_str(), "c");
    }

    #[test]
    fn test_composite_reranking() {
        let reranker = Reranker::new(
            RerankerConfig::new().with_strategy(RerankerType::Composite),
        );

        let results = vec![
            create_test_result("a", 0.9, 0.5, 86400 * 30),
            create_test_result("b", 0.7, 0.9, 3600),
        ];

        let reranked = reranker.rerank(results);

        // Results should be reranked based on combined factors
        assert_eq!(reranked.len(), 2);
    }

    #[test]
    fn test_empty_results() {
        let reranker = Reranker::new(RerankerConfig::new());
        let results = Vec::new();
        let reranked = reranker.rerank(results);
        assert!(reranked.is_empty());
    }

    #[test]
    fn test_single_result() {
        let reranker = Reranker::new(RerankerConfig::new());
        let results = vec![create_test_result("a", 0.9, 0.5, 0)];
        let reranked = reranker.rerank(results);
        assert_eq!(reranked.len(), 1);
    }
}
