//! Hybrid Retrieval Engine
//!
//! Combines dense vector search with sparse BM25 retrieval for improved recall.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    HybridQueryEngine                              │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  Query: (embedding, text_query)                                   │
//! │         ↓                                                         │
//! │  ┌─────────────┐        ┌─────────────┐                          │
//! │  │ Dense Index │        │ BM25 Index  │                          │
//! │  │  (HNSW/Flat)│        │  (Sparse)   │                          │
//! │  └──────┬──────┘        └──────┬──────┘                          │
//! │         │                      │                                  │
//! │         └──────────┬───────────┘                                  │
//! │                    ↓                                              │
//! │            Score Fusion (RRF/Linear)                              │
//! │                    ↓                                              │
//! │             Rerank + Filter                                       │
//! │                    ↓                                              │
//! │              Build Priors                                         │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rag_plusplus_core::retrieval::hybrid::{HybridQueryEngine, HybridQueryRequest};
//!
//! let engine = HybridQueryEngine::new(config, &dense_index, &sparse_index, &store);
//!
//! // Hybrid query with both embedding and text
//! let request = HybridQueryRequest::new(embedding)
//!     .with_text("neural network architecture")
//!     .with_k(10);
//!
//! let response = engine.query(request)?;
//! ```

use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::filter::{CompiledFilter, FilterExpr};
use crate::index::{SearchResult, VectorIndex};
use crate::retrieval::rerank::{Reranker, RerankerConfig};
use crate::sparse::{BM25Index, HybridFusionStrategy, HybridResult, HybridSearchConfig, HybridSearcher};
use crate::stats::OutcomeStats;
use crate::store::RecordStore;
use crate::types::{MemoryRecord, PriorBundle, RecordId};

/// Configuration for hybrid query engine.
#[derive(Debug, Clone)]
pub struct HybridQueryEngineConfig {
    /// Default number of results to return
    pub default_k: usize,
    /// Maximum allowed k
    pub max_k: usize,
    /// Query timeout in milliseconds
    pub timeout_ms: u64,
    /// Hybrid search configuration
    pub hybrid_config: HybridSearchConfig,
    /// Reranker configuration
    pub reranker: Option<RerankerConfig>,
    /// Whether to build priors from results
    pub build_priors: bool,
    /// Default fusion strategy when no text query provided (dense-only)
    pub fallback_to_dense: bool,
}

impl Default for HybridQueryEngineConfig {
    fn default() -> Self {
        Self {
            default_k: 10,
            max_k: 1000,
            timeout_ms: 5000,
            hybrid_config: HybridSearchConfig::default(),
            reranker: None,
            build_priors: true,
            fallback_to_dense: true,
        }
    }
}

impl HybridQueryEngineConfig {
    /// Create new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default k.
    #[must_use]
    pub const fn with_default_k(mut self, k: usize) -> Self {
        self.default_k = k;
        self
    }

    /// Set timeout.
    #[must_use]
    pub const fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set fusion strategy.
    #[must_use]
    pub fn with_fusion_strategy(mut self, strategy: HybridFusionStrategy) -> Self {
        self.hybrid_config.strategy = strategy;
        self
    }

    /// Set candidates per index.
    #[must_use]
    pub fn with_candidates_per_index(mut self, n: usize) -> Self {
        self.hybrid_config.candidates_per_index = n;
        self
    }

    /// Set reranker.
    #[must_use]
    pub fn with_reranker(mut self, config: RerankerConfig) -> Self {
        self.reranker = Some(config);
        self
    }
}

/// Hybrid query request.
#[derive(Debug, Clone)]
pub struct HybridQueryRequest {
    /// Dense embedding for vector search
    pub embedding: Vec<f32>,
    /// Text query for BM25 sparse search (optional)
    pub text_query: Option<String>,
    /// Number of results (uses default if None)
    pub k: Option<usize>,
    /// Metadata filter (optional)
    pub filter: Option<FilterExpr>,
    /// Timeout override (milliseconds)
    pub timeout_ms: Option<u64>,
    /// Override fusion strategy for this query
    pub fusion_strategy: Option<HybridFusionStrategy>,
}

impl HybridQueryRequest {
    /// Create a new hybrid query request with embedding only.
    #[must_use]
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            text_query: None,
            k: None,
            filter: None,
            timeout_ms: None,
            fusion_strategy: None,
        }
    }

    /// Set text query for sparse (BM25) search.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text_query = Some(text.into());
        self
    }

    /// Set k.
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Set filter.
    #[must_use]
    pub fn with_filter(mut self, filter: FilterExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set timeout override.
    #[must_use]
    pub const fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Set fusion strategy override.
    #[must_use]
    pub fn with_fusion_strategy(mut self, strategy: HybridFusionStrategy) -> Self {
        self.fusion_strategy = Some(strategy);
        self
    }

    /// Check if this is a hybrid query (has both embedding and text).
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        self.text_query.is_some()
    }
}

/// Single result in hybrid query response.
#[derive(Debug, Clone)]
pub struct HybridRetrievedRecord {
    /// The full record
    pub record: MemoryRecord,
    /// Fused score (combined dense + sparse)
    pub score: f32,
    /// Dense (vector) score component (if available)
    pub dense_score: Option<f32>,
    /// Sparse (BM25) score component (if available)
    pub sparse_score: Option<f32>,
    /// Rank in results (1-indexed)
    pub rank: usize,
}

/// Hybrid query response.
#[derive(Debug, Clone)]
pub struct HybridQueryResponse {
    /// Retrieved records
    pub results: Vec<HybridRetrievedRecord>,
    /// Prior bundle built from results
    pub priors: Option<PriorBundle>,
    /// Query execution time
    pub latency: Duration,
    /// Whether hybrid search was used (vs dense-only fallback)
    pub used_hybrid: bool,
    /// Total candidates considered
    pub candidates_considered: usize,
}

impl HybridQueryResponse {
    /// Get top result (if any).
    #[must_use]
    pub fn top(&self) -> Option<&HybridRetrievedRecord> {
        self.results.first()
    }

    /// Check if any results were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }
}

/// Hybrid query engine combining dense and sparse retrieval.
pub struct HybridQueryEngine<'a, I: VectorIndex, S: RecordStore> {
    /// Configuration
    config: HybridQueryEngineConfig,
    /// Dense (vector) index
    dense_index: &'a I,
    /// Sparse (BM25) index
    sparse_index: &'a BM25Index,
    /// Record store
    store: &'a S,
    /// Reranker (if configured)
    reranker: Option<Reranker>,
}

impl<'a, I: VectorIndex, S: RecordStore> HybridQueryEngine<'a, I, S> {
    /// Create a new hybrid query engine.
    #[must_use]
    pub fn new(
        config: HybridQueryEngineConfig,
        dense_index: &'a I,
        sparse_index: &'a BM25Index,
        store: &'a S,
    ) -> Self {
        let reranker = config.reranker.clone().map(Reranker::new);
        Self {
            config,
            dense_index,
            sparse_index,
            store,
            reranker,
        }
    }

    /// Execute a hybrid query.
    ///
    /// If `text_query` is provided, performs hybrid search (dense + sparse).
    /// Otherwise, falls back to dense-only search.
    ///
    /// # Errors
    ///
    /// Returns error if query is invalid, timeout occurs, or search fails.
    pub fn query(&self, request: HybridQueryRequest) -> Result<HybridQueryResponse> {
        let start = Instant::now();
        let timeout = Duration::from_millis(
            request.timeout_ms.unwrap_or(self.config.timeout_ms),
        );

        // Validate query
        self.validate_query(&request)?;

        // Determine k
        let k = request.k.unwrap_or(self.config.default_k).min(self.config.max_k);

        // Execute search
        let (hybrid_results, used_hybrid) = self.execute_search(&request, k)?;

        // Check timeout
        if start.elapsed() > timeout {
            return Err(Error::QueryTimeout {
                elapsed_ms: start.elapsed().as_millis() as u64,
                budget_ms: timeout.as_millis() as u64,
            });
        }

        // Fetch records and build results
        let mut results = self.build_results(hybrid_results)?;
        let candidates_considered = results.len();

        // Apply filter if specified
        if let Some(ref filter_expr) = request.filter {
            let filter = CompiledFilter::compile(filter_expr.clone());
            results.retain(|r| filter.evaluate(&r.record.metadata));
        }

        // Rerank if configured
        if let Some(ref reranker) = self.reranker {
            results = self.rerank_results(reranker, results);
        }

        // Truncate to k
        results.truncate(k);

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // Build priors
        let priors = if self.config.build_priors && !results.is_empty() {
            Some(self.build_priors(&results))
        } else {
            None
        };

        Ok(HybridQueryResponse {
            results,
            priors,
            latency: start.elapsed(),
            used_hybrid,
            candidates_considered,
        })
    }

    /// Validate query request.
    fn validate_query(&self, request: &HybridQueryRequest) -> Result<()> {
        if request.embedding.is_empty() {
            return Err(Error::InvalidQuery {
                reason: "Empty embedding".into(),
            });
        }

        // Check dimension matches
        let dim = self.dense_index.dimension();
        if dim > 0 && request.embedding.len() != dim {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "Dimension mismatch: query has {}, index expects {}",
                    request.embedding.len(),
                    dim
                ),
            });
        }

        if let Some(k) = request.k {
            if k == 0 {
                return Err(Error::InvalidQuery {
                    reason: "k must be > 0".into(),
                });
            }
            if k > self.config.max_k {
                return Err(Error::InvalidQuery {
                    reason: format!("k exceeds maximum ({})", self.config.max_k),
                });
            }
        }

        Ok(())
    }

    /// Execute search (hybrid or dense-only).
    fn execute_search(
        &self,
        request: &HybridQueryRequest,
        k: usize,
    ) -> Result<(Vec<HybridResult>, bool)> {
        // Build hybrid config with optional strategy override
        let mut hybrid_config = self.config.hybrid_config.clone();
        if let Some(strategy) = request.fusion_strategy {
            hybrid_config.strategy = strategy;
        }

        let searcher = HybridSearcher::new(self.dense_index, self.sparse_index)
            .with_config(hybrid_config);

        if let Some(ref text_query) = request.text_query {
            // Hybrid search
            let results = searcher.search(&request.embedding, text_query, k)?;
            Ok((results, true))
        } else if self.config.fallback_to_dense {
            // Dense-only fallback
            let results = searcher.search_dense_only(&request.embedding, k)?;
            Ok((results, false))
        } else {
            Err(Error::InvalidQuery {
                reason: "Text query required for hybrid search".into(),
            })
        }
    }

    /// Build result records from hybrid results.
    fn build_results(
        &self,
        hybrid_results: Vec<HybridResult>,
    ) -> Result<Vec<HybridRetrievedRecord>> {
        let mut results = Vec::with_capacity(hybrid_results.len());

        for hr in hybrid_results {
            let id: RecordId = hr.id.into();

            if let Some(record) = self.store.get(&id) {
                results.push(HybridRetrievedRecord {
                    record,
                    score: hr.score,
                    dense_score: hr.dense_score,
                    sparse_score: hr.sparse_score,
                    rank: 0, // Set later
                });
            }
        }

        Ok(results)
    }

    /// Rerank results.
    fn rerank_results(
        &self,
        reranker: &Reranker,
        results: Vec<HybridRetrievedRecord>,
    ) -> Vec<HybridRetrievedRecord> {
        // Convert to RetrievedRecord for reranking, then back
        use crate::retrieval::engine::RetrievedRecord;

        let converted: Vec<RetrievedRecord> = results
            .iter()
            .map(|r| RetrievedRecord {
                record: r.record.clone(),
                score: r.score,
                rank: r.rank,
                source_index: "hybrid".to_string(),
            })
            .collect();

        let reranked = reranker.rerank(converted);

        // Map back preserving component scores
        reranked
            .into_iter()
            .map(|rr| {
                // Find original to get component scores
                let original = results.iter().find(|r| r.record.id == rr.record.id);
                HybridRetrievedRecord {
                    record: rr.record,
                    score: rr.score,
                    dense_score: original.and_then(|o| o.dense_score),
                    sparse_score: original.and_then(|o| o.sparse_score),
                    rank: rr.rank,
                }
            })
            .collect()
    }

    /// Build priors from results.
    fn build_priors(&self, results: &[HybridRetrievedRecord]) -> PriorBundle {
        let mut stats = OutcomeStats::new(1);

        for result in results {
            stats.update_scalar(result.record.outcome);
            if result.record.stats.dim() == 1 {
                stats = stats.merge(&result.record.stats);
            }
        }

        let mean = stats.mean_scalar().unwrap_or(0.0);
        let std_dev = stats.std_scalar().unwrap_or(0.0);
        let ci = stats.confidence_interval(0.95)
            .map(|(l, u)| (l[0] as f64, u[0] as f64))
            .unwrap_or((mean, mean));

        PriorBundle {
            mean_outcome: mean,
            std_outcome: std_dev,
            confidence_interval: ci,
            sample_count: stats.count(),
            prototype_ids: results.iter().take(3).map(|r| r.record.id.clone()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{FlatIndex, IndexConfig};
    use crate::store::InMemoryStore;
    use crate::types::RecordStatus;

    fn create_test_record(id: &str, embedding: Vec<f32>, text: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.into(),
            embedding,
            context: text.to_string(),
            outcome: 0.8,
            metadata: Default::default(),
            created_at: 1234567890,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        }
    }

    fn setup_test_env() -> (FlatIndex, BM25Index, InMemoryStore) {
        let mut dense_index = FlatIndex::new(IndexConfig::new(4));
        let mut sparse_index = BM25Index::new();
        let mut store = InMemoryStore::new();

        // Add records
        let records = vec![
            ("rec-1", vec![1.0, 0.0, 0.0, 0.0], "machine learning algorithms"),
            ("rec-2", vec![0.0, 1.0, 0.0, 0.0], "deep neural networks"),
            ("rec-3", vec![0.0, 0.0, 1.0, 0.0], "natural language processing"),
            ("rec-4", vec![1.0, 1.0, 0.0, 0.0], "reinforcement learning agents"),
            ("rec-5", vec![0.0, 1.0, 1.0, 0.0], "computer vision models"),
        ];

        for (id, embedding, text) in records {
            let record = create_test_record(id, embedding.clone(), text);
            dense_index.add(record.id.to_string(), &embedding).unwrap();
            sparse_index.add(record.id.to_string(), text);
            store.insert(record).unwrap();
        }

        (dense_index, sparse_index, store)
    }

    #[test]
    fn test_hybrid_query_basic() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_text("machine learning")
            .with_k(3);

        let response = engine.query(request).unwrap();

        assert!(response.used_hybrid);
        assert!(!response.is_empty());
        assert!(response.len() <= 3);

        // First result should have both scores
        let top = response.top().unwrap();
        assert!(top.dense_score.is_some() || top.sparse_score.is_some());
    }

    #[test]
    fn test_dense_only_fallback() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        // No text query - should fallback to dense-only
        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_k(3);

        let response = engine.query(request).unwrap();

        assert!(!response.used_hybrid);
        assert!(!response.is_empty());

        // Should only have dense scores
        for result in &response.results {
            assert!(result.dense_score.is_some());
            assert!(result.sparse_score.is_none());
        }
    }

    #[test]
    fn test_hybrid_with_rrf() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new()
            .with_fusion_strategy(HybridFusionStrategy::RRF { k: 60.0 });
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_text("learning")
            .with_k(5);

        let response = engine.query(request).unwrap();

        assert!(response.used_hybrid);
        assert!(!response.is_empty());
    }

    #[test]
    fn test_hybrid_with_linear_fusion() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new()
            .with_fusion_strategy(HybridFusionStrategy::Linear { alpha: 0.7 });
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![0.0, 1.0, 0.0, 0.0])
            .with_text("neural networks")
            .with_k(3);

        let response = engine.query(request).unwrap();

        assert!(response.used_hybrid);
        // rec-2 should rank highly (matches both dense and sparse)
        let ids: Vec<_> = response.results.iter().map(|r| r.record.id.to_string()).collect();
        assert!(ids.contains(&"rec-2".to_string()));
    }

    #[test]
    fn test_query_strategy_override() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new()
            .with_fusion_strategy(HybridFusionStrategy::RRF { k: 60.0 });
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        // Override with linear fusion for this query
        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_text("machine learning")
            .with_k(3)
            .with_fusion_strategy(HybridFusionStrategy::Linear { alpha: 0.5 });

        let response = engine.query(request).unwrap();

        assert!(response.used_hybrid);
        assert!(!response.is_empty());
    }

    #[test]
    fn test_hybrid_with_priors() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_text("learning")
            .with_k(5);

        let response = engine.query(request).unwrap();

        assert!(response.priors.is_some());
        let priors = response.priors.unwrap();
        assert!(priors.sample_count > 0);
    }

    #[test]
    fn test_empty_query_validation() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![]);
        let result = engine.query(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_k_zero_validation() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_k(0);
        let result = engine.query(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_response_latency() {
        let (dense, sparse, store) = setup_test_env();
        let config = HybridQueryEngineConfig::new();
        let engine = HybridQueryEngine::new(config, &dense, &sparse, &store);

        let request = HybridQueryRequest::new(vec![1.0, 0.0, 0.0, 0.0])
            .with_text("machine")
            .with_k(3);

        let response = engine.query(request).unwrap();

        assert!(response.latency.as_micros() > 0);
    }
}
