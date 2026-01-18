//! Core Data Types
//!
//! Production-grade data structures for memory-conditioned retrieval.
//!
//! # Invariants
//!
//! - INV-001: MemoryRecord is immutable after creation (only stats can be updated)
//! - All types are Send + Sync

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::stats::OutcomeStats;

/// Unique record identifier (typically UUID or content hash).
pub type RecordId = String;

/// Reference to a vector (borrowed slice).
pub type VectorRef<'a> = &'a [f32];

/// Owned vector type.
pub type Vector = Vec<f32>;

/// Metadata key-value store.
pub type Metadata = HashMap<String, MetadataValue>;

/// Metadata value variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StringList(Vec<String>),
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

/// Record status in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordStatus {
    Active,
    Archived,
    Deleted,
}

impl Default for RecordStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// Memory record with outcome annotations.
///
/// This is the fundamental unit of storage in RAG++.
///
/// # Invariants
///
/// - INV-001: Once created, embedding and ID never change
/// - Only `stats` can be mutated after insertion
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    /// Unique record identifier
    pub id: RecordId,
    /// Primary embedding vector
    pub embedding: Vector,
    /// Context/description text
    pub context: String,
    /// Primary outcome value
    pub outcome: f64,
    /// Arbitrary metadata for filtering
    pub metadata: Metadata,
    /// Creation timestamp (Unix seconds)
    pub created_at: u64,
    /// Record status
    pub status: RecordStatus,
    /// Running outcome statistics
    pub stats: OutcomeStats,
}

impl MemoryRecord {
    /// Create a new memory record.
    #[must_use]
    pub fn new(id: impl Into<RecordId>, embedding: Vector, context: impl Into<String>, outcome: f64) -> Self {
        Self {
            id: id.into(),
            embedding,
            context: context.into(),
            outcome,
            metadata: Metadata::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1), // Default to 1D stats
        }
    }

    /// Embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.embedding.len()
    }

    /// Check if record is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status == RecordStatus::Active
    }
}

/// Query specification for retrieval.
#[derive(Debug, Clone)]
pub struct QueryBundle {
    /// Primary query embedding
    pub query_embedding: Vector,
    /// Number of results to retrieve
    pub k: usize,
    /// Metadata filters
    pub filters: Metadata,
    /// HNSW ef_search parameter
    pub ef_search: usize,
    /// Query timeout in milliseconds
    pub timeout_ms: u64,
}

impl QueryBundle {
    /// Create a new query bundle.
    #[must_use]
    pub fn new(query_embedding: Vector, k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        assert!(!query_embedding.is_empty(), "Query embedding cannot be empty");

        Self {
            query_embedding,
            k,
            filters: Metadata::new(),
            ef_search: 128,
            timeout_ms: 10_000,
        }
    }

    /// Query embedding dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.query_embedding.len()
    }

    /// Whether query has metadata filters.
    #[must_use]
    pub fn has_filters(&self) -> bool {
        !self.filters.is_empty()
    }

    /// Whether query uses multiple indexes.
    #[must_use]
    pub fn is_multi_index(&self) -> bool {
        false // Simplified - not using secondary embeddings in this version
    }

    /// Add a filter.
    #[must_use]
    pub fn with_filter(mut self, key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        self.filters.insert(key.into(), value.into());
        self
    }

    /// Set ef_search.
    #[must_use]
    pub const fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set timeout.
    #[must_use]
    pub const fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

/// Aggregated priors from retrieval for downstream conditioning.
#[derive(Debug, Clone)]
pub struct PriorBundle {
    /// Weighted mean of outcomes
    pub mean_outcome: f64,
    /// Weighted standard deviation
    pub std_outcome: f64,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (f64, f64),
    /// Sample count
    pub sample_count: u64,
    /// Prototype record IDs from top-k
    pub prototype_ids: Vec<RecordId>,
}

impl PriorBundle {
    /// Number of prototypes.
    #[must_use]
    pub fn num_prototypes(&self) -> usize {
        self.prototype_ids.len()
    }
}

/// Single retrieval result.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Matched record
    pub record: MemoryRecord,
    /// Similarity score (higher = more similar)
    pub score: f32,
    /// Rank in result set (1-indexed)
    pub rank: usize,
    /// Raw distance from query
    pub distance: f32,
}

/// Collection of retrieval results.
#[derive(Debug, Clone)]
pub struct RetrievalResults {
    /// Results ordered by score
    pub results: Vec<RetrievalResult>,
    /// Original query
    pub query: QueryBundle,
    /// Total candidates before filtering
    pub total_candidates: usize,
    /// End-to-end latency in milliseconds
    pub latency_ms: f64,
}

impl RetrievalResults {
    /// Number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Get record IDs.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.results.iter().map(|r| r.record.id.as_str()).collect()
    }

    /// Get scores.
    #[must_use]
    pub fn scores(&self) -> Vec<f32> {
        self.results.iter().map(|r| r.score).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_record_creation() {
        let embedding = vec![0.1, 0.2, 0.3];
        let record = MemoryRecord::new("test_001", embedding.clone(), "Test context", 0.8);

        assert_eq!(record.id, "test_001");
        assert_eq!(record.dim(), 3);
        assert_eq!(record.context, "Test context");
        assert!((record.outcome - 0.8).abs() < 0.001);
        assert!(record.is_active());
    }

    #[test]
    fn test_query_bundle() {
        let query = QueryBundle::new(vec![1.0, 2.0, 3.0], 10)
            .with_filter("genre", "electronic")
            .with_ef_search(256);

        assert_eq!(query.k, 10);
        assert_eq!(query.dim(), 3);
        assert!(query.has_filters());
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn test_zero_k_panics() {
        let _ = QueryBundle::new(vec![1.0], 0);
    }
}
