//! Parallel Search
//!
//! High-performance parallel search across multiple indexes using rayon.
//!
//! # Overview
//!
//! This module provides parallel search capabilities for multi-index scenarios,
//! distributing queries across CPU cores for maximum throughput.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    ParallelSearcher                         │
//! ├────────────────────────────────────────────────────────────┤
//! │  registry: &IndexRegistry                                   │
//! │  thread_pool: Option<rayon::ThreadPool>                     │
//! ├────────────────────────────────────────────────────────────┤
//! │  + search_parallel(query, k) -> MultiIndexResults           │
//! │  + search_batch(queries, k) -> Vec<MultiIndexResults>       │
//! │  + search_indexes_parallel(names, query, k)                 │
//! └────────────────────────────────────────────────────────────┘
//!                            │
//!              ┌─────────────┴─────────────┐
//!              │       rayon parallel       │
//!              │         iteration          │
//!              └────────────┬──────────────┘
//!                           │
//!       ┌───────────┬───────┴───────┬───────────┐
//!       ▼           ▼               ▼           ▼
//!   Index 0     Index 1         Index 2     Index N
//! ```

use crate::error::Result;
use crate::index::registry::{IndexRegistry, MultiIndexResult, MultiIndexResults};
use crate::index::traits::SearchResult;
use rayon::prelude::*;

/// Configuration for parallel search.
#[derive(Debug, Clone)]
pub struct ParallelSearchConfig {
    /// Number of threads to use (0 = auto-detect)
    pub num_threads: usize,
    /// Minimum indexes per thread (avoid over-parallelization)
    pub min_indexes_per_thread: usize,
    /// Enable batch query parallelization
    pub batch_parallel: bool,
}

impl Default for ParallelSearchConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // auto-detect
            min_indexes_per_thread: 1,
            batch_parallel: true,
        }
    }
}

impl ParallelSearchConfig {
    /// Create new config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of threads.
    #[must_use]
    pub const fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set minimum indexes per thread.
    #[must_use]
    pub const fn with_min_indexes_per_thread(mut self, min: usize) -> Self {
        self.min_indexes_per_thread = min;
        self
    }
}

/// Parallel searcher for multi-index queries.
///
/// Wraps an `IndexRegistry` and provides parallel search operations.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::index::{IndexRegistry, ParallelSearcher, FlatIndex, IndexConfig};
///
/// let mut registry = IndexRegistry::new();
/// registry.register("text", FlatIndex::new(IndexConfig::new(768)))?;
/// registry.register("code", FlatIndex::new(IndexConfig::new(512)))?;
///
/// let searcher = ParallelSearcher::new(&registry);
/// let results = searcher.search_parallel(&query, 10)?;
/// ```
pub struct ParallelSearcher<'a> {
    /// Reference to the registry
    registry: &'a IndexRegistry,
    /// Configuration
    config: ParallelSearchConfig,
}

impl<'a> ParallelSearcher<'a> {
    /// Create a new parallel searcher with default config.
    #[must_use]
    pub fn new(registry: &'a IndexRegistry) -> Self {
        Self {
            registry,
            config: ParallelSearchConfig::default(),
        }
    }

    /// Create a new parallel searcher with custom config.
    #[must_use]
    pub fn with_config(registry: &'a IndexRegistry, config: ParallelSearchConfig) -> Self {
        Self { registry, config }
    }

    /// Search all compatible indexes in parallel.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of results per index
    ///
    /// # Returns
    ///
    /// Results from all indexes with matching dimension.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Result<MultiIndexResults> {
        let query_dim = query.len();

        // Collect compatible indexes
        let indexes: Vec<_> = self
            .registry
            .info()
            .into_iter()
            .filter(|info| info.dimension == query_dim)
            .map(|info| info.name)
            .collect();

        if indexes.is_empty() {
            return Ok(MultiIndexResults::new());
        }

        // Decide on parallelization strategy
        let use_parallel = indexes.len() >= self.config.min_indexes_per_thread * 2;

        let results: Vec<MultiIndexResult> = if use_parallel {
            // Parallel execution
            indexes
                .par_iter()
                .filter_map(|name| {
                    self.registry
                        .search(name, query, k)
                        .ok()
                        .map(|results| MultiIndexResult {
                            index_name: name.clone(),
                            results,
                        })
                })
                .collect()
        } else {
            // Sequential execution for small number of indexes
            indexes
                .iter()
                .filter_map(|name| {
                    self.registry
                        .search(name, query, k)
                        .ok()
                        .map(|results| MultiIndexResult {
                            index_name: name.clone(),
                            results,
                        })
                })
                .collect()
        };

        let total_count = results.iter().map(|r| r.results.len()).sum();

        Ok(MultiIndexResults {
            by_index: results,
            total_count,
        })
    }

    /// Search specific indexes in parallel.
    ///
    /// # Arguments
    ///
    /// * `names` - Index names to search
    /// * `query` - Query vector
    /// * `k` - Number of results per index
    pub fn search_indexes_parallel(
        &self,
        names: &[&str],
        query: &[f32],
        k: usize,
    ) -> Result<MultiIndexResults> {
        let use_parallel = names.len() >= self.config.min_indexes_per_thread * 2;

        let results: Vec<MultiIndexResult> = if use_parallel {
            names
                .par_iter()
                .filter_map(|name| {
                    self.registry
                        .search(name, query, k)
                        .ok()
                        .map(|results| MultiIndexResult {
                            index_name: (*name).to_string(),
                            results,
                        })
                })
                .collect()
        } else {
            names
                .iter()
                .filter_map(|name| {
                    self.registry
                        .search(name, query, k)
                        .ok()
                        .map(|results| MultiIndexResult {
                            index_name: (*name).to_string(),
                            results,
                        })
                })
                .collect()
        };

        let total_count = results.iter().map(|r| r.results.len()).sum();

        Ok(MultiIndexResults {
            by_index: results,
            total_count,
        })
    }

    /// Batch search: run multiple queries in parallel.
    ///
    /// # Arguments
    ///
    /// * `queries` - Multiple query vectors
    /// * `k` - Number of results per query per index
    ///
    /// # Returns
    ///
    /// Results for each query.
    pub fn search_batch(&self, queries: &[Vec<f32>], k: usize) -> Vec<Result<MultiIndexResults>> {
        if self.config.batch_parallel && queries.len() > 1 {
            queries
                .par_iter()
                .map(|query| self.search_parallel(query, k))
                .collect()
        } else {
            queries
                .iter()
                .map(|query| self.search_parallel(query, k))
                .collect()
        }
    }

    /// Batch search specific indexes.
    pub fn search_indexes_batch(
        &self,
        names: &[&str],
        queries: &[Vec<f32>],
        k: usize,
    ) -> Vec<Result<MultiIndexResults>> {
        if self.config.batch_parallel && queries.len() > 1 {
            queries
                .par_iter()
                .map(|query| self.search_indexes_parallel(names, query, k))
                .collect()
        } else {
            queries
                .iter()
                .map(|query| self.search_indexes_parallel(names, query, k))
                .collect()
        }
    }
}

/// Parallel add operation for batch indexing.
///
/// Adds vectors to multiple indexes in parallel.
pub fn parallel_add_batch(
    registry: &mut IndexRegistry,
    index_name: &str,
    ids: Vec<String>,
    vectors: &[Vec<f32>],
) -> Result<()> {
    // Validate inputs
    if ids.len() != vectors.len() {
        return Err(crate::error::Error::InvalidQuery {
            reason: format!(
                "IDs count ({}) doesn't match vectors count ({})",
                ids.len(),
                vectors.len()
            ),
        });
    }

    // For now, we add sequentially but could use interior mutability
    // patterns for true parallel writes in the future
    for (id, vector) in ids.into_iter().zip(vectors.iter()) {
        registry.add(index_name, id, vector)?;
    }

    Ok(())
}

/// Results aggregator for parallel searches.
#[derive(Debug, Default)]
pub struct ResultsAggregator {
    /// Results by query index
    results: Vec<MultiIndexResults>,
}

impl ResultsAggregator {
    /// Create new aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add results from a query.
    pub fn add(&mut self, results: MultiIndexResults) {
        self.results.push(results);
    }

    /// Get all results.
    #[must_use]
    pub fn results(&self) -> &[MultiIndexResults] {
        &self.results
    }

    /// Total number of results across all queries.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.results.iter().map(|r| r.total_count).sum()
    }

    /// Flatten all results with query index.
    #[must_use]
    pub fn flatten_with_query(&self) -> Vec<(usize, String, SearchResult)> {
        self.results
            .iter()
            .enumerate()
            .flat_map(|(qi, mir)| {
                mir.by_index.iter().flat_map(move |idx_result| {
                    idx_result
                        .results
                        .iter()
                        .cloned()
                        .map(move |r| (qi, idx_result.index_name.clone(), r))
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{FlatIndex, IndexConfig, VectorIndex};

    fn setup_test_registry() -> IndexRegistry {
        let mut registry = IndexRegistry::new();

        // Create multiple indexes
        let mut idx1 = FlatIndex::new(IndexConfig::new(4));
        idx1.add("a1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx1.add("a2".to_string(), &[0.9, 0.1, 0.0, 0.0]).unwrap();

        let mut idx2 = FlatIndex::new(IndexConfig::new(4));
        idx2.add("b1".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx2.add("b2".to_string(), &[0.1, 0.9, 0.0, 0.0]).unwrap();

        let mut idx3 = FlatIndex::new(IndexConfig::new(4));
        idx3.add("c1".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();
        idx3.add("c2".to_string(), &[0.0, 0.1, 0.9, 0.0]).unwrap();

        registry.register("idx1", idx1).unwrap();
        registry.register("idx2", idx2).unwrap();
        registry.register("idx3", idx3).unwrap();

        registry
    }

    #[test]
    fn test_parallel_search() {
        let registry = setup_test_registry();
        let searcher = ParallelSearcher::new(&registry);

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = searcher.search_parallel(&query, 10).unwrap();

        // Should search all 3 indexes
        assert_eq!(results.by_index.len(), 3);
        assert_eq!(results.total_count, 6); // 2 per index
    }

    #[test]
    fn test_search_indexes_parallel() {
        let registry = setup_test_registry();
        let searcher = ParallelSearcher::new(&registry);

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = searcher
            .search_indexes_parallel(&["idx1", "idx2"], &query, 10)
            .unwrap();

        // Should only search specified indexes
        assert_eq!(results.by_index.len(), 2);
        assert_eq!(results.total_count, 4);
    }

    #[test]
    fn test_search_batch() {
        let registry = setup_test_registry();
        let searcher = ParallelSearcher::new(&registry);

        let queries = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let results = searcher.search_batch(&queries, 10);

        assert_eq!(results.len(), 3);
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelSearchConfig::new()
            .with_threads(4)
            .with_min_indexes_per_thread(2);

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.min_indexes_per_thread, 2);
    }

    #[test]
    fn test_results_aggregator() {
        let mut aggregator = ResultsAggregator::new();

        let mut results1 = MultiIndexResults::new();
        results1.add(
            "idx1".to_string(),
            vec![SearchResult::new(
                "a".to_string(),
                0.5,
                crate::index::DistanceType::L2,
            )],
        );

        let mut results2 = MultiIndexResults::new();
        results2.add(
            "idx2".to_string(),
            vec![SearchResult::new(
                "b".to_string(),
                0.3,
                crate::index::DistanceType::L2,
            )],
        );

        aggregator.add(results1);
        aggregator.add(results2);

        assert_eq!(aggregator.results().len(), 2);
        assert_eq!(aggregator.total_count(), 2);

        let flat = aggregator.flatten_with_query();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].0, 0); // Query index 0
        assert_eq!(flat[1].0, 1); // Query index 1
    }

    #[test]
    fn test_incompatible_dimension_skipped() {
        let mut registry = IndexRegistry::new();

        // Different dimensions
        let mut idx1 = FlatIndex::new(IndexConfig::new(4));
        idx1.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut idx2 = FlatIndex::new(IndexConfig::new(8));
        idx2.add("b".to_string(), &[1.0; 8]).unwrap();

        registry.register("idx1", idx1).unwrap();
        registry.register("idx2", idx2).unwrap();

        let searcher = ParallelSearcher::new(&registry);

        // Query with dim 4 should only search idx1
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = searcher.search_parallel(&query, 10).unwrap();

        assert_eq!(results.by_index.len(), 1);
    }
}
