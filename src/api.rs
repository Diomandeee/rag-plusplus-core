//! RAG++ Public API Contracts
//!
//! This module defines the formal input/output contracts for RAG++.
//! These types form the stable interface for external consumers.

use crate::error::Result;
use crate::stats::OutcomeStats;
use crate::types::{MemoryRecord, RecordId};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// INPUT CONTRACTS
// ============================================================================

/// Query request to the RAG++ retrieval engine.
///
/// This is the primary input contract for retrieval operations.
#[derive(Debug, Clone)]
pub struct RetrievalRequest {
    /// Query embedding vector (must match index dimension)
    pub embedding: Vec<f32>,

    /// Number of candidates to retrieve
    pub k: usize,

    /// Optional metadata filter expression
    pub filter: Option<FilterExpression>,

    /// Optional: specific indexes to search (None = search all)
    pub index_names: Option<Vec<String>>,

    /// Whether to compute priors from retrieved records
    pub compute_priors: bool,

    /// Optional timeout for the query
    pub timeout: Option<Duration>,
}

impl RetrievalRequest {
    /// Create a simple retrieval request.
    #[must_use]
    pub fn new(embedding: Vec<f32>, k: usize) -> Self {
        Self {
            embedding,
            k,
            filter: None,
            index_names: None,
            compute_priors: true,
            timeout: None,
        }
    }

    /// Add a metadata filter.
    #[must_use]
    pub fn with_filter(mut self, filter: FilterExpression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Specify which indexes to search.
    #[must_use]
    pub fn with_indexes(mut self, names: Vec<String>) -> Self {
        self.index_names = Some(names);
        self
    }

    /// Set query timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Validate the request.
    pub fn validate(&self, expected_dim: usize) -> Result<()> {
        if self.embedding.len() != expected_dim {
            return Err(crate::error::Error::DimensionMismatch {
                expected: expected_dim,
                got: self.embedding.len(),
            });
        }
        if self.k == 0 {
            return Err(crate::error::Error::InvalidQuery {
                reason: "k must be greater than 0".into(),
            });
        }
        if self.embedding.iter().any(|x| !x.is_finite()) {
            return Err(crate::error::Error::InvalidQuery {
                reason: "embedding contains NaN or Inf".into(),
            });
        }
        Ok(())
    }
}

/// Simplified filter expression for the public API.
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// Field equals value
    Eq(String, FilterValue),
    /// Field not equals value
    Ne(String, FilterValue),
    /// Field greater than value
    Gt(String, FilterValue),
    /// Field greater than or equal
    Gte(String, FilterValue),
    /// Field less than value
    Lt(String, FilterValue),
    /// Field less than or equal
    Lte(String, FilterValue),
    /// Field in set of values
    In(String, Vec<FilterValue>),
    /// Logical AND of expressions
    And(Vec<FilterExpression>),
    /// Logical OR of expressions
    Or(Vec<FilterExpression>),
    /// Logical NOT
    Not(Box<FilterExpression>),
}

/// Filter value types.
#[derive(Debug, Clone)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// Record to be ingested into the RAG++ corpus.
#[derive(Debug, Clone)]
pub struct IngestRecord {
    /// Unique identifier (must be unique within corpus)
    pub id: String,

    /// Embedding vector
    pub embedding: Vec<f32>,

    /// Human-readable context description
    pub context: String,

    /// Primary outcome metric
    pub outcome: f64,

    /// Arbitrary metadata
    pub metadata: HashMap<String, MetadataValue>,
}

/// Metadata value for ingestion.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StringList(Vec<String>),
}

// ============================================================================
// OUTPUT CONTRACTS
// ============================================================================

/// Response from a RAG++ retrieval query.
///
/// This is the primary output contract for retrieval operations.
#[derive(Debug, Clone)]
pub struct RetrievalResponse {
    /// Statistical priors computed from retrieved records
    pub prior: PriorBundle,

    /// Ranked candidates with scores
    pub candidates: Vec<RankedCandidate>,

    /// Query execution latency
    pub latency: Duration,

    /// Which indexes were searched
    pub indexes_searched: Vec<String>,

    /// Total number of records considered
    pub records_scanned: usize,

    /// Whether the query hit the cache
    pub cache_hit: bool,
}

/// Statistical priors from retrieved trajectories.
///
/// This is the core value proposition of RAG++ - surfacing implicit
/// knowledge from past execution outcomes as queryable statistics.
#[derive(Debug, Clone, Default)]
pub struct PriorBundle {
    /// Mean outcome of retrieved trajectories
    pub mean: Option<f64>,

    /// Variance of outcomes
    pub variance: Option<f64>,

    /// Standard deviation
    pub std_dev: Option<f64>,

    /// Confidence in the estimate (0-1, based on sample count)
    pub confidence: f64,

    /// Number of samples contributing to statistics
    pub count: u64,

    /// Minimum observed outcome
    pub min: Option<f64>,

    /// Maximum observed outcome
    pub max: Option<f64>,

    /// Weighted mean (by retrieval score)
    pub weighted_mean: Option<f64>,
}

impl PriorBundle {
    /// Create from outcome statistics.
    #[must_use]
    pub fn from_stats(stats: &OutcomeStats) -> Self {
        let count = stats.count();
        let confidence = Self::compute_confidence(count);

        Self {
            mean: stats.mean_scalar(),
            variance: stats.variance_scalar(),
            std_dev: stats.std_scalar(),
            confidence,
            count,
            min: stats.min().and_then(|m| m.first().copied().map(f64::from)),
            max: stats.max().and_then(|m| m.first().copied().map(f64::from)),
            weighted_mean: None,
        }
    }

    /// Create from a set of outcomes with optional weights.
    #[must_use]
    pub fn from_outcomes(outcomes: &[f64], weights: Option<&[f64]>) -> Self {
        if outcomes.is_empty() {
            return Self::default();
        }

        let count = outcomes.len() as u64;
        let confidence = Self::compute_confidence(count);

        // Simple statistics
        let mean = outcomes.iter().sum::<f64>() / outcomes.len() as f64;
        let variance = if outcomes.len() > 1 {
            let sum_sq: f64 = outcomes.iter().map(|x| (x - mean).powi(2)).sum();
            Some(sum_sq / (outcomes.len() - 1) as f64)
        } else {
            None
        };
        let std_dev = variance.map(|v| v.sqrt());
        let min = outcomes.iter().copied().fold(f64::INFINITY, f64::min);
        let max = outcomes.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Weighted mean
        let weighted_mean = weights.map(|w| {
            let total_weight: f64 = w.iter().sum();
            if total_weight > 0.0 {
                outcomes
                    .iter()
                    .zip(w.iter())
                    .map(|(o, w)| o * w)
                    .sum::<f64>()
                    / total_weight
            } else {
                mean
            }
        });

        Self {
            mean: Some(mean),
            variance,
            std_dev,
            confidence,
            count,
            min: Some(min),
            max: Some(max),
            weighted_mean,
        }
    }

    /// Compute confidence based on sample count.
    ///
    /// Uses a logistic function that approaches 1.0 as count increases.
    fn compute_confidence(count: u64) -> f64 {
        if count == 0 {
            return 0.0;
        }
        // Logistic: 1 / (1 + e^(-k(x-x0)))
        // Tuned so: count=5 -> ~0.5, count=20 -> ~0.9, count=50 -> ~0.99
        let k = 0.15;
        let x0 = 10.0;
        1.0 / (1.0 + (-(k * (count as f64 - x0))).exp())
    }

    /// Whether the prior has enough samples to be reliable.
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Whether any statistics are available.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// A ranked candidate from retrieval.
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    /// Record identifier
    pub record_id: String,

    /// Retrieval score (higher = more relevant)
    pub score: f64,

    /// Raw distance from query
    pub distance: f64,

    /// Rank position (1-indexed)
    pub rank: u32,

    /// Outcome value from the record
    pub outcome: f64,

    /// Record context string
    pub context: String,
}

// ============================================================================
// TRAIT DEFINITIONS
// ============================================================================

/// Core trait for RAG++ retrieval engines.
///
/// Implementations must provide thread-safe retrieval operations.
pub trait RetrievalEngine: Send + Sync {
    /// Execute a retrieval query.
    fn query(&self, request: &RetrievalRequest) -> Result<RetrievalResponse>;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;

    /// Get the number of records in the corpus.
    fn corpus_size(&self) -> usize;

    /// Get available index names.
    fn index_names(&self) -> Vec<String>;
}

/// Trait for record storage.
pub trait Corpus: Send + Sync {
    /// Ingest a record into the corpus.
    fn ingest(&mut self, record: IngestRecord) -> Result<RecordId>;

    /// Ingest multiple records.
    fn ingest_batch(&mut self, records: Vec<IngestRecord>) -> Result<Vec<RecordId>>;

    /// Update outcome statistics for a record.
    fn update_outcome(&mut self, id: &RecordId, outcome: f64) -> Result<()>;

    /// Remove a record from the corpus.
    fn remove(&mut self, id: &RecordId) -> Result<bool>;

    /// Get a record by ID.
    fn get(&self, id: &RecordId) -> Option<MemoryRecord>;

    /// Get corpus size.
    fn size(&self) -> usize;
}

/// Trait for vector indexes.
pub trait VectorSearcher: Send + Sync {
    /// Add a vector to the index.
    fn add(&mut self, id: &str, vector: &[f32]) -> Result<()>;

    /// Search for nearest neighbors.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>>;

    /// Remove a vector from the index.
    fn remove(&mut self, id: &str) -> Result<bool>;

    /// Get the dimension.
    fn dimension(&self) -> usize;

    /// Get the number of vectors.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A search hit from vector search.
#[derive(Debug, Clone)]
pub struct SearchHit {
    /// Record ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
    /// Score (typically 1 / (1 + distance) or similar)
    pub score: f32,
}

// ============================================================================
// BUILDER PATTERN
// ============================================================================

/// Builder for constructing RAG++ instances.
#[derive(Debug, Clone)]
pub struct RAGBuilder {
    dimension: usize,
    index_type: IndexType,
    cache_enabled: bool,
    cache_size: usize,
    default_k: usize,
}

/// Index type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// Exact search (brute force)
    Flat,
    /// Approximate search (HNSW)
    Hnsw,
}

impl Default for RAGBuilder {
    fn default() -> Self {
        Self {
            dimension: 512,
            index_type: IndexType::Flat,
            cache_enabled: true,
            cache_size: 10000,
            default_k: 10,
        }
    }
}

impl RAGBuilder {
    /// Create a new builder with the given embedding dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the index type.
    #[must_use]
    pub fn index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Enable or disable caching.
    #[must_use]
    pub fn cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Set cache size.
    #[must_use]
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Set default k for queries.
    #[must_use]
    pub fn default_k(mut self, k: usize) -> Self {
        self.default_k = k;
        self
    }

    /// Get the configured dimension.
    #[must_use]
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configured index type.
    #[must_use]
    pub fn get_index_type(&self) -> IndexType {
        self.index_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_request_validation() {
        let valid = RetrievalRequest::new(vec![1.0, 2.0, 3.0], 10);
        assert!(valid.validate(3).is_ok());

        let wrong_dim = RetrievalRequest::new(vec![1.0, 2.0], 10);
        assert!(wrong_dim.validate(3).is_err());

        let zero_k = RetrievalRequest::new(vec![1.0, 2.0, 3.0], 0);
        assert!(zero_k.validate(3).is_err());

        let nan = RetrievalRequest::new(vec![1.0, f32::NAN, 3.0], 10);
        assert!(nan.validate(3).is_err());
    }

    #[test]
    fn test_prior_bundle_from_outcomes() {
        let outcomes = vec![0.8, 0.9, 0.7, 0.85];
        let prior = PriorBundle::from_outcomes(&outcomes, None);

        assert!(prior.mean.is_some());
        assert!((prior.mean.unwrap() - 0.8125).abs() < 1e-6);
        assert_eq!(prior.count, 4);
        assert!(prior.confidence > 0.0);
    }

    #[test]
    fn test_prior_bundle_empty() {
        let prior = PriorBundle::from_outcomes(&[], None);
        assert!(prior.is_empty());
        assert!(!prior.is_reliable());
    }

    #[test]
    fn test_prior_bundle_weighted() {
        let outcomes = vec![1.0, 0.0];
        let weights = vec![0.8, 0.2];
        let prior = PriorBundle::from_outcomes(&outcomes, Some(&weights));

        // Weighted mean: (1.0 * 0.8 + 0.0 * 0.2) / 1.0 = 0.8
        assert!(prior.weighted_mean.is_some());
        assert!((prior.weighted_mean.unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_scaling() {
        assert!(PriorBundle::compute_confidence(0) == 0.0);
        assert!(PriorBundle::compute_confidence(5) > 0.3);
        assert!(PriorBundle::compute_confidence(20) > 0.8);
        assert!(PriorBundle::compute_confidence(100) > 0.99);
    }

    #[test]
    fn test_builder() {
        let builder = RAGBuilder::new(768)
            .index_type(IndexType::Hnsw)
            .cache(true)
            .cache_size(5000)
            .default_k(20);

        assert_eq!(builder.get_dimension(), 768);
        assert_eq!(builder.get_index_type(), IndexType::Hnsw);
    }
}
