//! Score Fusion
//!
//! Algorithms for merging results from multiple indexes or retrieval methods.
//!
//! # Overview
//!
//! When searching multiple indexes with potentially different distance metrics
//! or scoring scales, raw scores cannot be directly compared. Score fusion
//! provides principled methods to combine rankings.
//!
//! # Algorithms
//!
//! - **RRF (Reciprocal Rank Fusion)**: Rank-based fusion that ignores score magnitudes
//! - **CombSUM**: Sum of normalized scores across sources
//! - **CombMNZ**: CombSUM weighted by number of sources containing the result
//! - **Weighted**: User-specified weights per index
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                    ScoreFusion                              │
//! ├────────────────────────────────────────────────────────────┤
//! │  strategy: FusionStrategy                                   │
//! │  k: usize (RRF constant, default 60)                        │
//! │  weights: Option<HashMap<String, f32>>                      │
//! ├────────────────────────────────────────────────────────────┤
//! │  + fuse(MultiIndexResults) -> Vec<FusedResult>              │
//! │  + fuse_top_k(MultiIndexResults, k) -> Vec<FusedResult>     │
//! └────────────────────────────────────────────────────────────┘
//! ```

use crate::index::registry::MultiIndexResults;
use ahash::AHashMap;
use ordered_float::OrderedFloat;

/// Fusion strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion - rank-based, ignores score magnitudes
    #[default]
    RRF,
    /// Sum of normalized scores
    CombSUM,
    /// CombSUM weighted by occurrence count
    CombMNZ,
    /// Maximum score across sources
    CombMAX,
    /// Minimum score across sources
    CombMIN,
}

/// Result after score fusion.
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// Record ID
    pub id: String,
    /// Fused score (higher = more relevant)
    pub fused_score: f32,
    /// Source indexes that contributed to this result
    pub sources: Vec<String>,
    /// Original scores from each source (index_name -> score)
    pub source_scores: AHashMap<String, f32>,
}

impl FusedResult {
    /// Create a new fused result.
    #[must_use]
    pub fn new(id: String, fused_score: f32) -> Self {
        Self {
            id,
            fused_score,
            sources: Vec::new(),
            source_scores: AHashMap::new(),
        }
    }

    /// Add a source contribution.
    pub fn add_source(&mut self, index_name: String, score: f32) {
        self.sources.push(index_name.clone());
        self.source_scores.insert(index_name, score);
    }

    /// Number of sources contributing to this result.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

/// Configuration for score fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Fusion strategy
    pub strategy: FusionStrategy,
    /// RRF constant k (default 60, higher = more weight to lower ranks)
    pub rrf_k: usize,
    /// Per-index weights for weighted fusion (None = equal weights)
    pub weights: Option<AHashMap<String, f32>>,
    /// Whether to normalize scores before fusion (for Comb methods)
    pub normalize_scores: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::RRF,
            rrf_k: 60,
            weights: None,
            normalize_scores: true,
        }
    }
}

impl FusionConfig {
    /// Create new config with default RRF settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Use RRF fusion strategy.
    #[must_use]
    pub const fn with_rrf(mut self, k: usize) -> Self {
        self.strategy = FusionStrategy::RRF;
        self.rrf_k = k;
        self
    }

    /// Use CombSUM fusion strategy.
    #[must_use]
    pub const fn with_comb_sum(mut self) -> Self {
        self.strategy = FusionStrategy::CombSUM;
        self
    }

    /// Use CombMNZ fusion strategy.
    #[must_use]
    pub const fn with_comb_mnz(mut self) -> Self {
        self.strategy = FusionStrategy::CombMNZ;
        self
    }

    /// Set per-index weights.
    #[must_use]
    pub fn with_weights(mut self, weights: AHashMap<String, f32>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set whether to normalize scores.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize_scores = normalize;
        self
    }
}

/// Score fusion engine.
pub struct ScoreFusion {
    config: FusionConfig,
}

impl ScoreFusion {
    /// Create a new score fusion engine with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: FusionConfig::default(),
        }
    }

    /// Create with custom configuration.
    #[must_use]
    pub fn with_config(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Create RRF fusion engine with default k=60.
    #[must_use]
    pub fn rrf() -> Self {
        Self::new()
    }

    /// Create RRF fusion engine with custom k.
    #[must_use]
    pub fn rrf_with_k(k: usize) -> Self {
        Self::with_config(FusionConfig::new().with_rrf(k))
    }

    /// Fuse results from multiple indexes.
    #[must_use]
    pub fn fuse(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        match self.config.strategy {
            FusionStrategy::RRF => self.fuse_rrf(results),
            FusionStrategy::CombSUM => self.fuse_comb_sum(results),
            FusionStrategy::CombMNZ => self.fuse_comb_mnz(results),
            FusionStrategy::CombMAX => self.fuse_comb_max(results),
            FusionStrategy::CombMIN => self.fuse_comb_min(results),
        }
    }

    /// Fuse and return only top-k results.
    #[must_use]
    pub fn fuse_top_k(&self, results: &MultiIndexResults, k: usize) -> Vec<FusedResult> {
        let mut fused = self.fuse(results);
        fused.truncate(k);
        fused
    }

    /// Reciprocal Rank Fusion.
    ///
    /// Score = sum over sources of 1 / (k + rank)
    /// where rank is 1-indexed position in that source.
    fn fuse_rrf(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        let k = self.config.rrf_k as f32;
        let mut scores: AHashMap<String, FusedResult> = AHashMap::new();

        for idx_result in &results.by_index {
            let index_name = &idx_result.index_name;
            let weight = self.get_weight(index_name);

            for (rank, result) in idx_result.results.iter().enumerate() {
                let rrf_score = weight / (k + (rank + 1) as f32);

                let fused = scores.entry(result.id.clone()).or_insert_with(|| {
                    FusedResult::new(result.id.clone(), 0.0)
                });

                fused.fused_score += rrf_score;
                fused.add_source(index_name.clone(), result.score);
            }
        }

        self.sort_results(scores)
    }

    /// CombSUM: Sum of (normalized) scores.
    fn fuse_comb_sum(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        let normalized = if self.config.normalize_scores {
            self.normalize_per_index(results)
        } else {
            self.collect_scores(results)
        };

        let mut scores: AHashMap<String, FusedResult> = AHashMap::new();

        for (id, index_scores) in normalized {
            let mut fused = FusedResult::new(id.clone(), 0.0);

            for (index_name, score) in index_scores {
                let weight = self.get_weight(&index_name);
                fused.fused_score += weight * score;
                fused.add_source(index_name, score);
            }

            scores.insert(id, fused);
        }

        self.sort_results(scores)
    }

    /// CombMNZ: CombSUM weighted by number of sources.
    fn fuse_comb_mnz(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        let normalized = if self.config.normalize_scores {
            self.normalize_per_index(results)
        } else {
            self.collect_scores(results)
        };

        let mut scores: AHashMap<String, FusedResult> = AHashMap::new();

        for (id, index_scores) in normalized {
            let mut fused = FusedResult::new(id.clone(), 0.0);
            let mut sum = 0.0;

            for (index_name, score) in index_scores {
                let weight = self.get_weight(&index_name);
                sum += weight * score;
                fused.add_source(index_name, score);
            }

            // Multiply by number of sources (MNZ = "multiply by non-zero")
            fused.fused_score = sum * fused.source_count() as f32;
            scores.insert(id, fused);
        }

        self.sort_results(scores)
    }

    /// CombMAX: Maximum score across sources.
    fn fuse_comb_max(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        let normalized = if self.config.normalize_scores {
            self.normalize_per_index(results)
        } else {
            self.collect_scores(results)
        };

        let mut scores: AHashMap<String, FusedResult> = AHashMap::new();

        for (id, index_scores) in normalized {
            let mut fused = FusedResult::new(id.clone(), 0.0);
            let mut max_score: f32 = 0.0;

            for (index_name, score) in index_scores {
                let weight = self.get_weight(&index_name);
                let weighted = weight * score;
                max_score = max_score.max(weighted);
                fused.add_source(index_name, score);
            }

            fused.fused_score = max_score;
            scores.insert(id, fused);
        }

        self.sort_results(scores)
    }

    /// CombMIN: Minimum score across sources.
    fn fuse_comb_min(&self, results: &MultiIndexResults) -> Vec<FusedResult> {
        let normalized = if self.config.normalize_scores {
            self.normalize_per_index(results)
        } else {
            self.collect_scores(results)
        };

        let mut scores: AHashMap<String, FusedResult> = AHashMap::new();

        for (id, index_scores) in normalized {
            let mut fused = FusedResult::new(id.clone(), 0.0);
            let mut min_score: f32 = f32::MAX;

            for (index_name, score) in index_scores {
                let weight = self.get_weight(&index_name);
                let weighted = weight * score;
                min_score = min_score.min(weighted);
                fused.add_source(index_name, score);
            }

            fused.fused_score = if min_score == f32::MAX { 0.0 } else { min_score };
            scores.insert(id, fused);
        }

        self.sort_results(scores)
    }

    /// Get weight for an index (default 1.0).
    fn get_weight(&self, index_name: &str) -> f32 {
        self.config
            .weights
            .as_ref()
            .and_then(|w| w.get(index_name))
            .copied()
            .unwrap_or(1.0)
    }

    /// Collect scores without normalization.
    fn collect_scores(&self, results: &MultiIndexResults) -> AHashMap<String, Vec<(String, f32)>> {
        let mut collected: AHashMap<String, Vec<(String, f32)>> = AHashMap::new();

        for idx_result in &results.by_index {
            for result in &idx_result.results {
                collected
                    .entry(result.id.clone())
                    .or_default()
                    .push((idx_result.index_name.clone(), result.score));
            }
        }

        collected
    }

    /// Normalize scores per index to [0, 1] range.
    fn normalize_per_index(
        &self,
        results: &MultiIndexResults,
    ) -> AHashMap<String, Vec<(String, f32)>> {
        let mut collected: AHashMap<String, Vec<(String, f32)>> = AHashMap::new();

        for idx_result in &results.by_index {
            // Find min/max for this index
            let scores: Vec<f32> = idx_result.results.iter().map(|r| r.score).collect();
            let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_score - min_score;

            for result in &idx_result.results {
                let normalized = if range > f32::EPSILON {
                    (result.score - min_score) / range
                } else {
                    1.0 // All scores equal
                };

                collected
                    .entry(result.id.clone())
                    .or_default()
                    .push((idx_result.index_name.clone(), normalized));
            }
        }

        collected
    }

    /// Sort results by fused score (descending).
    fn sort_results(&self, scores: AHashMap<String, FusedResult>) -> Vec<FusedResult> {
        let mut sorted: Vec<FusedResult> = scores.into_values().collect();
        sorted.sort_by(|a, b| {
            OrderedFloat(b.fused_score).cmp(&OrderedFloat(a.fused_score))
        });
        sorted
    }
}

impl Default for ScoreFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for RRF fusion.
#[must_use]
pub fn rrf_fuse(results: &MultiIndexResults) -> Vec<FusedResult> {
    ScoreFusion::rrf().fuse(results)
}

/// Convenience function for RRF fusion with top-k.
#[must_use]
pub fn rrf_fuse_top_k(results: &MultiIndexResults, k: usize) -> Vec<FusedResult> {
    ScoreFusion::rrf().fuse_top_k(results, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::registry::MultiIndexResult;
    use crate::SearchResult;

    fn make_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            distance: 1.0 - score, // Fake distance
            score,
        }
    }

    fn make_multi_results() -> MultiIndexResults {
        MultiIndexResults {
            by_index: vec![
                MultiIndexResult {
                    index_name: "idx1".to_string(),
                    results: vec![
                        make_result("a", 0.9),
                        make_result("b", 0.8),
                        make_result("c", 0.7),
                    ],
                },
                MultiIndexResult {
                    index_name: "idx2".to_string(),
                    results: vec![
                        make_result("b", 0.95), // b is top in idx2
                        make_result("a", 0.85),
                        make_result("d", 0.75),
                    ],
                },
            ],
            total_count: 6,
        }
    }

    #[test]
    fn test_rrf_fusion() {
        let results = make_multi_results();
        let fused = ScoreFusion::rrf().fuse(&results);

        // Should have 4 unique IDs: a, b, c, d
        assert_eq!(fused.len(), 4);

        // a and b appear in both indexes with symmetric ranks, so they have equal RRF scores
        // Top two should be a and b (either order is valid for tied scores)
        assert!(fused[0].id == "a" || fused[0].id == "b");
        assert_eq!(fused[0].source_count(), 2);

        assert!(fused[1].id == "a" || fused[1].id == "b");
        assert_eq!(fused[1].source_count(), 2);
        assert_ne!(fused[0].id, fused[1].id); // Both should be present

        // c and d only appear in one index each
        assert!(fused[2].id == "c" || fused[2].id == "d");
        assert!(fused[3].id == "c" || fused[3].id == "d");
    }

    #[test]
    fn test_rrf_scores() {
        let results = make_multi_results();
        let fusion = ScoreFusion::rrf_with_k(60);
        let fused = fusion.fuse(&results);

        // For item b:
        // idx1: rank 2 -> 1/(60+2) = 1/62
        // idx2: rank 1 -> 1/(60+1) = 1/61
        // Total: 1/62 + 1/61 ≈ 0.01613 + 0.01639 ≈ 0.03252
        let b = fused.iter().find(|r| r.id == "b").unwrap();
        let expected = 1.0 / 62.0 + 1.0 / 61.0;
        assert!((b.fused_score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_comb_sum() {
        let results = make_multi_results();
        let fusion = ScoreFusion::with_config(FusionConfig::new().with_comb_sum());
        let fused = fusion.fuse(&results);

        // a and b appear in both indexes with equal combined scores (0.9+0.85 = 0.8+0.95 = 1.75)
        // Either order is valid for tied scores
        assert!(fused[0].id == "a" || fused[0].id == "b");
        assert!(fused[1].id == "a" || fused[1].id == "b");
        assert_ne!(fused[0].id, fused[1].id);
    }

    #[test]
    fn test_comb_mnz() {
        let results = make_multi_results();
        let fusion = ScoreFusion::with_config(FusionConfig::new().with_comb_mnz());
        let fused = fusion.fuse(&results);

        // Items appearing in both indexes should be boosted
        let b = fused.iter().find(|r| r.id == "b").unwrap();
        let c = fused.iter().find(|r| r.id == "c").unwrap();

        // b appears in 2 sources, c in 1
        assert_eq!(b.source_count(), 2);
        assert_eq!(c.source_count(), 1);
    }

    #[test]
    fn test_weighted_fusion() {
        let results = make_multi_results();

        let mut weights = AHashMap::new();
        weights.insert("idx1".to_string(), 2.0);
        weights.insert("idx2".to_string(), 1.0);

        let fusion = ScoreFusion::with_config(FusionConfig::new().with_weights(weights));
        let fused = fusion.fuse(&results);

        // 'a' is ranked #1 in idx1 (weight 2.0) vs 'b' ranked #2
        // With double weight on idx1, 'a' should score higher
        assert_eq!(fused[0].id, "a");
    }

    #[test]
    fn test_top_k() {
        let results = make_multi_results();
        let fused = ScoreFusion::rrf().fuse_top_k(&results, 2);

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_convenience_functions() {
        let results = make_multi_results();

        let fused1 = rrf_fuse(&results);
        let fused2 = rrf_fuse_top_k(&results, 2);

        assert_eq!(fused1.len(), 4);
        assert_eq!(fused2.len(), 2);
    }

    #[test]
    fn test_empty_results() {
        let results = MultiIndexResults::default();
        let fused = ScoreFusion::rrf().fuse(&results);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_single_index() {
        let results = MultiIndexResults {
            by_index: vec![MultiIndexResult {
                index_name: "only".to_string(),
                results: vec![make_result("a", 0.9), make_result("b", 0.8)],
            }],
            total_count: 2,
        };

        let fused = ScoreFusion::rrf().fuse(&results);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[1].id, "b");
    }

    #[test]
    fn test_fused_result_sources() {
        let results = make_multi_results();
        let fused = ScoreFusion::rrf().fuse(&results);

        let b = fused.iter().find(|r| r.id == "b").unwrap();
        assert!(b.sources.contains(&"idx1".to_string()));
        assert!(b.sources.contains(&"idx2".to_string()));
        assert!(b.source_scores.contains_key("idx1"));
        assert!(b.source_scores.contains_key("idx2"));
    }

    #[test]
    fn test_comb_max() {
        let results = MultiIndexResults {
            by_index: vec![
                MultiIndexResult {
                    index_name: "idx1".to_string(),
                    results: vec![make_result("a", 0.5), make_result("b", 0.9)],
                },
                MultiIndexResult {
                    index_name: "idx2".to_string(),
                    results: vec![make_result("a", 0.8), make_result("b", 0.3)],
                },
            ],
            total_count: 4,
        };

        let fusion = ScoreFusion::with_config(FusionConfig {
            strategy: FusionStrategy::CombMAX,
            normalize_scores: false,
            ..Default::default()
        });
        let fused = fusion.fuse(&results);

        // a: max(0.5, 0.8) = 0.8
        // b: max(0.9, 0.3) = 0.9
        assert_eq!(fused[0].id, "b");
        assert!((fused[0].fused_score - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_comb_min() {
        let results = MultiIndexResults {
            by_index: vec![
                MultiIndexResult {
                    index_name: "idx1".to_string(),
                    results: vec![make_result("a", 0.5), make_result("b", 0.9)],
                },
                MultiIndexResult {
                    index_name: "idx2".to_string(),
                    results: vec![make_result("a", 0.8), make_result("b", 0.3)],
                },
            ],
            total_count: 4,
        };

        let fusion = ScoreFusion::with_config(FusionConfig {
            strategy: FusionStrategy::CombMIN,
            normalize_scores: false,
            ..Default::default()
        });
        let fused = fusion.fuse(&results);

        // a: min(0.5, 0.8) = 0.5
        // b: min(0.9, 0.3) = 0.3
        // a should be ranked higher with min strategy
        assert_eq!(fused[0].id, "a");
        assert!((fused[0].fused_score - 0.5).abs() < 0.001);
    }
}
