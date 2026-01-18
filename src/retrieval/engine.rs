//! Query Engine
//!
//! End-to-end query execution with validation, search, and result building.

use crate::error::{Error, Result};
use crate::filter::{CompiledFilter, FilterExpr};
use crate::index::{
    IndexRegistry, MultiIndexResults, ParallelSearcher, SearchResult,
    rrf_fuse,
};
use crate::retrieval::rerank::{Reranker, RerankerConfig};
use crate::stats::OutcomeStats;
use crate::store::RecordStore;
use crate::types::{MemoryRecord, PriorBundle, RecordId};
use std::time::{Duration, Instant};

/// Query engine configuration.
#[derive(Debug, Clone)]
pub struct QueryEngineConfig {
    /// Default number of results to return
    pub default_k: usize,
    /// Maximum allowed k
    pub max_k: usize,
    /// Query timeout in milliseconds
    pub timeout_ms: u64,
    /// Whether to use parallel search for multi-index queries
    pub parallel_search: bool,
    /// Reranker configuration
    pub reranker: Option<RerankerConfig>,
    /// Whether to build priors from results
    pub build_priors: bool,
}

impl Default for QueryEngineConfig {
    fn default() -> Self {
        Self {
            default_k: 10,
            max_k: 1000,
            timeout_ms: 5000,
            parallel_search: true,
            reranker: None,
            build_priors: true,
        }
    }
}

impl QueryEngineConfig {
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

    /// Set reranker.
    #[must_use]
    pub fn with_reranker(mut self, config: RerankerConfig) -> Self {
        self.reranker = Some(config);
        self
    }
}

/// Query request.
#[derive(Debug, Clone)]
pub struct QueryRequest {
    /// Query embedding
    pub embedding: Vec<f32>,
    /// Number of results (uses default if None)
    pub k: Option<usize>,
    /// Metadata filter (optional)
    pub filter: Option<FilterExpr>,
    /// Specific index names to search (None = all)
    pub indexes: Option<Vec<String>>,
    /// Timeout override (milliseconds)
    pub timeout_ms: Option<u64>,
}

impl QueryRequest {
    /// Create a new query request.
    #[must_use]
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            k: None,
            filter: None,
            indexes: None,
            timeout_ms: None,
        }
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

    /// Set specific indexes to search.
    #[must_use]
    pub fn with_indexes(mut self, indexes: Vec<String>) -> Self {
        self.indexes = Some(indexes);
        self
    }
}

/// Single result in query response.
#[derive(Debug, Clone)]
pub struct RetrievedRecord {
    /// The full record
    pub record: MemoryRecord,
    /// Similarity score (0-1, higher is better)
    pub score: f32,
    /// Rank in results (1-indexed)
    pub rank: usize,
    /// Source index name
    pub source_index: String,
}

/// Query response.
#[derive(Debug, Clone)]
pub struct QueryResponse {
    /// Retrieved records
    pub results: Vec<RetrievedRecord>,
    /// Prior bundle built from results
    pub priors: Option<PriorBundle>,
    /// Query execution time
    pub latency: Duration,
    /// Number of indexes searched
    pub indexes_searched: usize,
    /// Total candidates considered
    pub candidates_considered: usize,
}

impl QueryResponse {
    /// Get top result (if any).
    #[must_use]
    pub fn top(&self) -> Option<&RetrievedRecord> {
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

/// Query engine for executing retrieval queries.
///
/// Provides end-to-end query execution including:
/// - Query validation
/// - Vector search (single or multi-index)
/// - Metadata filtering
/// - Result reranking
/// - Prior building
pub struct QueryEngine<'a, S: RecordStore> {
    /// Configuration
    config: QueryEngineConfig,
    /// Index registry
    registry: &'a IndexRegistry,
    /// Record store
    store: &'a S,
    /// Reranker (if configured)
    reranker: Option<Reranker>,
}

impl<'a, S: RecordStore> QueryEngine<'a, S> {
    /// Create a new query engine.
    #[must_use]
    pub fn new(
        config: QueryEngineConfig,
        registry: &'a IndexRegistry,
        store: &'a S,
    ) -> Self {
        let reranker = config.reranker.clone().map(Reranker::new);
        Self {
            config,
            registry,
            store,
            reranker,
        }
    }

    /// Execute a query.
    ///
    /// # Errors
    ///
    /// Returns error if query is invalid, timeout occurs, or search fails.
    pub fn query(&self, request: QueryRequest) -> Result<QueryResponse> {
        let start = Instant::now();
        let timeout = Duration::from_millis(
            request.timeout_ms.unwrap_or(self.config.timeout_ms),
        );

        // Validate query
        self.validate_query(&request)?;

        // Determine k
        let k = request.k.unwrap_or(self.config.default_k).min(self.config.max_k);

        // Execute search
        let (search_results, indexes_searched) = self.execute_search(&request, k)?;

        // Check timeout
        if start.elapsed() > timeout {
            return Err(Error::QueryTimeout {
                elapsed_ms: start.elapsed().as_millis() as u64,
                budget_ms: timeout.as_millis() as u64,
            });
        }

        // Fetch records and build results
        let mut results = self.build_results(search_results, &request)?;
        let candidates_considered = results.len();

        // Apply filter if specified
        if let Some(ref filter_expr) = request.filter {
            let filter = CompiledFilter::compile(filter_expr.clone());
            results.retain(|r| filter.evaluate(&r.record.metadata));
        }

        // Rerank if configured
        if let Some(ref reranker) = self.reranker {
            results = reranker.rerank(results);
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

        Ok(QueryResponse {
            results,
            priors,
            latency: start.elapsed(),
            indexes_searched,
            candidates_considered,
        })
    }

    /// Validate query request.
    fn validate_query(&self, request: &QueryRequest) -> Result<()> {
        if request.embedding.is_empty() {
            return Err(Error::InvalidQuery {
                reason: "Empty embedding".into(),
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

        // Check that at least one index has matching dimension
        let dim = request.embedding.len();
        let has_compatible = self.registry.info().iter().any(|i| i.dimension == dim);

        if !has_compatible {
            return Err(Error::InvalidQuery {
                reason: format!("No index with dimension {dim}"),
            });
        }

        Ok(())
    }

    /// Execute vector search.
    fn execute_search(
        &self,
        request: &QueryRequest,
        k: usize,
    ) -> Result<(Vec<(String, SearchResult)>, usize)> {
        let query = &request.embedding;

        // Multi-index or specific indexes?
        let multi_results: MultiIndexResults = if let Some(ref index_names) = request.indexes {
            // Search specific indexes
            let names: Vec<&str> = index_names.iter().map(String::as_str).collect();
            if self.config.parallel_search && names.len() > 1 {
                let searcher = ParallelSearcher::new(self.registry);
                searcher.search_indexes_parallel(&names, query, k)?
            } else {
                self.registry.search_indexes(&names, query, k)?
            }
        } else {
            // Search all compatible indexes
            if self.config.parallel_search {
                let searcher = ParallelSearcher::new(self.registry);
                searcher.search_parallel(query, k)?
            } else {
                self.registry.search_all(query, k)?
            }
        };

        let indexes_searched = multi_results.by_index.len();

        // Fuse results if multiple indexes
        let results: Vec<(String, SearchResult)> = if indexes_searched > 1 {
            let fused = rrf_fuse(&multi_results);
            fused
                .into_iter()
                .map(|f| {
                    let source = f.sources.first().cloned().unwrap_or_default();
                    (
                        source,
                        SearchResult {
                            id: f.id,
                            distance: 0.0, // Not meaningful after fusion
                            score: f.fused_score,
                        },
                    )
                })
                .collect()
        } else {
            multi_results.flatten()
        };

        Ok((results, indexes_searched))
    }

    /// Build result records from search results.
    fn build_results(
        &self,
        search_results: Vec<(String, SearchResult)>,
        _request: &QueryRequest,
    ) -> Result<Vec<RetrievedRecord>> {
        let mut results = Vec::with_capacity(search_results.len());

        for (index_name, sr) in search_results {
            let id: RecordId = sr.id.into();

            if let Some(record) = self.store.get(&id) {
                results.push(RetrievedRecord {
                    record,
                    score: sr.score,
                    rank: 0, // Set later
                    source_index: index_name,
                });
            }
        }

        Ok(results)
    }

    /// Build priors from results.
    fn build_priors(&self, results: &[RetrievedRecord]) -> PriorBundle {
        let mut stats = OutcomeStats::new(1);

        for result in results {
            stats.update_scalar(result.record.outcome);
            // Merge record's stats if compatible (same dimension)
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
    use crate::index::{FlatIndex, IndexConfig, VectorIndex};
    use crate::store::InMemoryStore;
    use crate::types::RecordStatus;
    use crate::OutcomeStats;

    fn create_test_record(id: &str, embedding: Vec<f32>) -> MemoryRecord {
        MemoryRecord {
            id: id.into(),
            embedding,
            context: format!("Context for {id}"),
            outcome: 0.8,
            metadata: Default::default(),
            created_at: 1234567890,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        }
    }

    fn setup_test_env() -> (IndexRegistry, InMemoryStore) {
        let mut registry = IndexRegistry::new();
        let mut store = InMemoryStore::new();

        // Create index
        let mut index = FlatIndex::new(IndexConfig::new(4));

        // Add records
        for i in 0..10 {
            let embedding = vec![i as f32, 0.0, 0.0, 0.0];
            let record = create_test_record(&format!("rec-{i}"), embedding.clone());

            index.add(record.id.to_string(), &embedding).unwrap();
            store.insert(record).unwrap();
        }

        registry.register("test", index).unwrap();
        (registry, store)
    }

    #[test]
    fn test_basic_query() {
        let (registry, store) = setup_test_env();
        let engine = QueryEngine::new(
            QueryEngineConfig::new(),
            &registry,
            &store,
        );

        let request = QueryRequest::new(vec![5.0, 0.0, 0.0, 0.0]).with_k(3);
        let response = engine.query(request).unwrap();

        assert_eq!(response.len(), 3);
        assert!(!response.is_empty());
        assert!(response.priors.is_some());
    }

    #[test]
    fn test_query_validation_empty_embedding() {
        let (registry, store) = setup_test_env();
        let engine = QueryEngine::new(
            QueryEngineConfig::new(),
            &registry,
            &store,
        );

        let request = QueryRequest::new(vec![]);
        let result = engine.query(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_query_validation_k_zero() {
        let (registry, store) = setup_test_env();
        let engine = QueryEngine::new(
            QueryEngineConfig::new(),
            &registry,
            &store,
        );

        let request = QueryRequest::new(vec![1.0, 0.0, 0.0, 0.0]).with_k(0);
        let result = engine.query(request);

        assert!(result.is_err());
    }

    #[test]
    fn test_query_with_priors() {
        let (registry, store) = setup_test_env();
        let config = QueryEngineConfig::new();
        let engine = QueryEngine::new(config, &registry, &store);

        let request = QueryRequest::new(vec![5.0, 0.0, 0.0, 0.0]).with_k(5);
        let response = engine.query(request).unwrap();

        let priors = response.priors.unwrap();
        assert!(priors.sample_count > 0);
        assert!(!priors.prototype_ids.is_empty());
    }

    #[test]
    fn test_multi_index_query() {
        let mut registry = IndexRegistry::new();
        let mut store = InMemoryStore::new();

        // Create two indexes
        let mut index1 = FlatIndex::new(IndexConfig::new(4));
        let mut index2 = FlatIndex::new(IndexConfig::new(4));

        // Add to first index
        let rec1 = create_test_record("rec-a", vec![1.0, 0.0, 0.0, 0.0]);
        index1.add(rec1.id.to_string(), &rec1.embedding).unwrap();
        store.insert(rec1).unwrap();

        // Add to second index
        let rec2 = create_test_record("rec-b", vec![0.0, 1.0, 0.0, 0.0]);
        index2.add(rec2.id.to_string(), &rec2.embedding).unwrap();
        store.insert(rec2).unwrap();

        registry.register("idx1", index1).unwrap();
        registry.register("idx2", index2).unwrap();

        let engine = QueryEngine::new(
            QueryEngineConfig::new(),
            &registry,
            &store,
        );

        let request = QueryRequest::new(vec![0.5, 0.5, 0.0, 0.0]).with_k(5);
        let response = engine.query(request).unwrap();

        assert_eq!(response.indexes_searched, 2);
        assert_eq!(response.len(), 2);
    }

    #[test]
    fn test_response_latency() {
        let (registry, store) = setup_test_env();
        let engine = QueryEngine::new(
            QueryEngineConfig::new(),
            &registry,
            &store,
        );

        let request = QueryRequest::new(vec![5.0, 0.0, 0.0, 0.0]).with_k(3);
        let response = engine.query(request).unwrap();

        assert!(response.latency.as_micros() > 0);
    }
}
