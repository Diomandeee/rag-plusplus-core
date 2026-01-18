//! Hybrid Search: Dense + Sparse Fusion
//!
//! Combines dense (vector) and sparse (BM25) retrieval for improved recall
//! and precision. Uses configurable score fusion strategies.
//!
//! # Algorithm
//!
//! 1. Query dense index for top-k candidates by vector similarity
//! 2. Query sparse index for top-k candidates by BM25 score
//! 3. Fuse results using one of:
//!    - RRF (Reciprocal Rank Fusion): position-based, no normalization needed
//!    - Linear combination: weighted sum of normalized scores
//!    - Convex combination: α * dense + (1-α) * sparse
//!
//! # Performance Notes
//!
//! - Hybrid search typically improves recall by 10-20% over dense-only
//! - Particularly effective for queries with specific terminology
//! - RRF is robust to score scale differences

use ahash::AHashMap;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::bm25::{BM25Index, SparseResult};
use crate::index::{SearchResult, VectorIndex};

/// Fusion strategy for combining dense and sparse results.
#[derive(Debug, Clone, Copy)]
pub enum HybridFusionStrategy {
    /// Reciprocal Rank Fusion: score = Σ 1/(k + rank)
    RRF { k: f32 },
    /// Linear combination: α * dense_score + (1-α) * sparse_score
    Linear { alpha: f32 },
    /// Weighted RRF: weight_dense * rrf_dense + weight_sparse * rrf_sparse
    WeightedRRF { dense_weight: f32, sparse_weight: f32, k: f32 },
}

impl Default for HybridFusionStrategy {
    fn default() -> Self {
        Self::RRF { k: 60.0 }
    }
}

/// Configuration for hybrid search.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Fusion strategy
    pub strategy: HybridFusionStrategy,
    /// Number of candidates to retrieve from each index
    pub candidates_per_index: usize,
    /// Minimum sparse score to include (filters low-relevance BM25 matches)
    pub min_sparse_score: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            strategy: HybridFusionStrategy::default(),
            candidates_per_index: 100,
            min_sparse_score: 0.0,
        }
    }
}

/// Result from hybrid search.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Document ID
    pub id: String,
    /// Combined score
    pub score: f32,
    /// Dense (vector) score component (if found in dense results)
    pub dense_score: Option<f32>,
    /// Sparse (BM25) score component (if found in sparse results)
    pub sparse_score: Option<f32>,
}

/// Hybrid searcher combining dense and sparse retrieval.
pub struct HybridSearcher<'a, I: VectorIndex> {
    /// Dense (vector) index
    dense_index: &'a I,
    /// Sparse (BM25) index
    sparse_index: &'a BM25Index,
    /// Configuration
    config: HybridSearchConfig,
}

impl<'a, I: VectorIndex> HybridSearcher<'a, I> {
    /// Create a new hybrid searcher.
    pub fn new(dense_index: &'a I, sparse_index: &'a BM25Index) -> Self {
        Self {
            dense_index,
            sparse_index,
            config: HybridSearchConfig::default(),
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: HybridSearchConfig) -> Self {
        self.config = config;
        self
    }

    /// Search with both dense vector and sparse text query.
    ///
    /// # Arguments
    /// * `vector_query` - Dense embedding vector
    /// * `text_query` - Sparse text query for BM25
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Top-k hybrid results sorted by fused score
    pub fn search(
        &self,
        vector_query: &[f32],
        text_query: &str,
        k: usize,
    ) -> crate::error::Result<Vec<HybridResult>> {
        // Get dense results
        let dense_results = self.dense_index.search(
            vector_query,
            self.config.candidates_per_index,
        )?;

        // Get sparse results
        let sparse_results = self.sparse_index.search(
            text_query,
            self.config.candidates_per_index,
        );

        // Fuse results
        let fused = self.fuse_results(&dense_results, &sparse_results);

        // Return top-k
        Ok(fused.into_iter().take(k).collect())
    }

    /// Search with only dense query (sparse disabled).
    pub fn search_dense_only(
        &self,
        vector_query: &[f32],
        k: usize,
    ) -> crate::error::Result<Vec<HybridResult>> {
        let dense_results = self.dense_index.search(vector_query, k)?;

        Ok(dense_results
            .into_iter()
            .map(|r| HybridResult {
                id: r.id,
                score: r.score,
                dense_score: Some(r.score),
                sparse_score: None,
            })
            .collect())
    }

    /// Search with only sparse query (dense disabled).
    pub fn search_sparse_only(
        &self,
        text_query: &str,
        k: usize,
    ) -> Vec<HybridResult> {
        let sparse_results = self.sparse_index.search(text_query, k);

        sparse_results
            .into_iter()
            .map(|r| HybridResult {
                id: r.id,
                score: r.score,
                dense_score: None,
                sparse_score: Some(r.score),
            })
            .collect()
    }

    /// Fuse dense and sparse results.
    fn fuse_results(
        &self,
        dense_results: &[SearchResult],
        sparse_results: &[SparseResult],
    ) -> Vec<HybridResult> {
        match self.config.strategy {
            HybridFusionStrategy::RRF { k } => {
                self.rrf_fusion(dense_results, sparse_results, k, 1.0, 1.0)
            }
            HybridFusionStrategy::Linear { alpha } => {
                self.linear_fusion(dense_results, sparse_results, alpha)
            }
            HybridFusionStrategy::WeightedRRF { dense_weight, sparse_weight, k } => {
                self.rrf_fusion(dense_results, sparse_results, k, dense_weight, sparse_weight)
            }
        }
    }

    /// RRF (Reciprocal Rank Fusion).
    fn rrf_fusion(
        &self,
        dense_results: &[SearchResult],
        sparse_results: &[SparseResult],
        k: f32,
        dense_weight: f32,
        sparse_weight: f32,
    ) -> Vec<HybridResult> {
        let mut scores: AHashMap<String, HybridResult> = AHashMap::new();

        // Add dense results with RRF score
        for (rank, result) in dense_results.iter().enumerate() {
            let rrf_score = dense_weight / (k + (rank + 1) as f32);
            let entry = scores.entry(result.id.clone()).or_insert(HybridResult {
                id: result.id.clone(),
                score: 0.0,
                dense_score: None,
                sparse_score: None,
            });
            entry.score += rrf_score;
            entry.dense_score = Some(result.score);
        }

        // Add sparse results with RRF score
        for (rank, result) in sparse_results.iter().enumerate() {
            if result.score < self.config.min_sparse_score {
                continue;
            }
            let rrf_score = sparse_weight / (k + (rank + 1) as f32);
            let entry = scores.entry(result.id.clone()).or_insert(HybridResult {
                id: result.id.clone(),
                score: 0.0,
                dense_score: None,
                sparse_score: None,
            });
            entry.score += rrf_score;
            entry.sparse_score = Some(result.score);
        }

        // Sort by fused score
        let mut results: Vec<_> = scores.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Linear score fusion.
    fn linear_fusion(
        &self,
        dense_results: &[SearchResult],
        sparse_results: &[SparseResult],
        alpha: f32,
    ) -> Vec<HybridResult> {
        // Normalize scores to [0, 1] for each result set
        let (dense_min, dense_max) = min_max_scores_dense(dense_results);
        let (sparse_min, sparse_max) = min_max_scores_sparse(sparse_results);

        let mut scores: AHashMap<String, HybridResult> = AHashMap::new();

        // Add normalized dense scores
        for result in dense_results {
            let norm_score = normalize_score(result.score, dense_min, dense_max);
            let entry = scores.entry(result.id.clone()).or_insert(HybridResult {
                id: result.id.clone(),
                score: 0.0,
                dense_score: None,
                sparse_score: None,
            });
            entry.score += alpha * norm_score;
            entry.dense_score = Some(result.score);
        }

        // Add normalized sparse scores
        for result in sparse_results {
            if result.score < self.config.min_sparse_score {
                continue;
            }
            let norm_score = normalize_score(result.score, sparse_min, sparse_max);
            let entry = scores.entry(result.id.clone()).or_insert(HybridResult {
                id: result.id.clone(),
                score: 0.0,
                dense_score: None,
                sparse_score: None,
            });
            entry.score += (1.0 - alpha) * norm_score;
            entry.sparse_score = Some(result.score);
        }

        // Sort by fused score
        let mut results: Vec<_> = scores.into_values().collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }
}

/// Get min/max scores from dense results.
fn min_max_scores_dense(results: &[SearchResult]) -> (f32, f32) {
    if results.is_empty() {
        return (0.0, 1.0);
    }
    let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
    let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

/// Get min/max scores from sparse results.
fn min_max_scores_sparse(results: &[SparseResult]) -> (f32, f32) {
    if results.is_empty() {
        return (0.0, 1.0);
    }
    let min = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
    let max = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
    (min, max)
}

/// Normalize score to [0, 1].
fn normalize_score(score: f32, min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-10 {
        0.5
    } else {
        (score - min) / (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{FlatIndex, IndexConfig};

    fn create_test_indexes() -> (FlatIndex, BM25Index) {
        let mut dense = FlatIndex::new(IndexConfig::new(4));
        let mut sparse = BM25Index::new();

        // Add some test documents
        dense.add("doc1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        dense.add("doc2".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        dense.add("doc3".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();

        sparse.add("doc1".to_string(), "machine learning algorithms");
        sparse.add("doc2".to_string(), "deep learning neural networks");
        sparse.add("doc3".to_string(), "natural language processing");

        (dense, sparse)
    }

    #[test]
    fn test_hybrid_search_basic() {
        let (dense, sparse) = create_test_indexes();
        let searcher = HybridSearcher::new(&dense, &sparse);

        let results = searcher
            .search(&[1.0, 0.0, 0.0, 0.0], "machine learning", 3)
            .unwrap();

        assert!(!results.is_empty());
        // doc1 should rank high (matches both dense and sparse)
        assert!(results.iter().any(|r| r.id == "doc1"));
    }

    #[test]
    fn test_dense_only() {
        let (dense, sparse) = create_test_indexes();
        let searcher = HybridSearcher::new(&dense, &sparse);

        let results = searcher.search_dense_only(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].dense_score.is_some());
        assert!(results[0].sparse_score.is_none());
    }

    #[test]
    fn test_sparse_only() {
        let (dense, sparse) = create_test_indexes();
        let searcher = HybridSearcher::new(&dense, &sparse);

        let results = searcher.search_sparse_only("neural networks", 3);

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc2");
        assert!(results[0].dense_score.is_none());
        assert!(results[0].sparse_score.is_some());
    }

    #[test]
    fn test_rrf_fusion() {
        let (dense, sparse) = create_test_indexes();
        let config = HybridSearchConfig {
            strategy: HybridFusionStrategy::RRF { k: 60.0 },
            candidates_per_index: 10,
            min_sparse_score: 0.0,
        };
        let searcher = HybridSearcher::new(&dense, &sparse).with_config(config);

        let results = searcher
            .search(&[1.0, 0.0, 0.0, 0.0], "machine learning", 3)
            .unwrap();

        // Check that results have fused scores
        assert!(results.iter().all(|r| r.score > 0.0));
    }

    #[test]
    fn test_linear_fusion() {
        let (dense, sparse) = create_test_indexes();
        let config = HybridSearchConfig {
            strategy: HybridFusionStrategy::Linear { alpha: 0.7 },
            candidates_per_index: 10,
            min_sparse_score: 0.0,
        };
        let searcher = HybridSearcher::new(&dense, &sparse).with_config(config);

        let results = searcher
            .search(&[1.0, 0.0, 0.0, 0.0], "machine", 3)
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_weighted_rrf() {
        let (dense, sparse) = create_test_indexes();
        let config = HybridSearchConfig {
            strategy: HybridFusionStrategy::WeightedRRF {
                dense_weight: 0.8,
                sparse_weight: 0.2,
                k: 60.0,
            },
            candidates_per_index: 10,
            min_sparse_score: 0.0,
        };
        let searcher = HybridSearcher::new(&dense, &sparse).with_config(config);

        let results = searcher
            .search(&[1.0, 0.0, 0.0, 0.0], "machine learning", 3)
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_score_components() {
        let (dense, sparse) = create_test_indexes();
        let searcher = HybridSearcher::new(&dense, &sparse);

        let results = searcher
            .search(&[1.0, 0.0, 0.0, 0.0], "machine learning", 3)
            .unwrap();

        // doc1 should have both scores
        let doc1 = results.iter().find(|r| r.id == "doc1");
        assert!(doc1.is_some());
        let doc1 = doc1.unwrap();
        assert!(doc1.dense_score.is_some());
        assert!(doc1.sparse_score.is_some());
    }
}
