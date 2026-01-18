//! Index Traits and Common Types
//!
//! Defines the core abstraction for vector indexes.

use crate::error::Result;
use std::fmt::Debug;

/// Distance metric for similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceType {
    /// Euclidean (L2) distance - smaller is more similar
    #[default]
    L2,
    /// Inner product - larger is more similar (use with normalized vectors)
    InnerProduct,
    /// Cosine similarity - larger is more similar
    Cosine,
}

/// Index configuration.
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub distance_type: DistanceType,
    /// Whether to normalize vectors before indexing
    pub normalize: bool,
}

impl IndexConfig {
    /// Create new index configuration.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            distance_type: DistanceType::L2,
            normalize: false,
        }
    }

    /// Set distance type.
    #[must_use]
    pub const fn with_distance(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self
    }

    /// Enable vector normalization.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Result from a nearest neighbor search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Record ID
    pub id: String,
    /// Distance from query (interpretation depends on metric)
    pub distance: f32,
    /// Similarity score (higher = more similar, normalized to 0-1)
    pub score: f32,
}

impl SearchResult {
    /// Create a new search result.
    #[must_use]
    pub fn new(id: String, distance: f32, distance_type: DistanceType) -> Self {
        let score = Self::distance_to_score(distance, distance_type);
        Self { id, distance, score }
    }

    /// Convert distance to similarity score (0-1, higher is better).
    fn distance_to_score(distance: f32, distance_type: DistanceType) -> f32 {
        match distance_type {
            DistanceType::L2 => {
                // Convert L2 distance to similarity: 1 / (1 + distance)
                1.0 / (1.0 + distance)
            }
            DistanceType::InnerProduct | DistanceType::Cosine => {
                // Inner product / cosine: already a similarity (may need clamping)
                distance.clamp(0.0, 1.0)
            }
        }
    }
}

/// Core trait for vector indexes.
///
/// All index implementations must implement this trait.
pub trait VectorIndex: Send + Sync + Debug {
    /// Add a vector to the index.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector to index
    ///
    /// # Errors
    ///
    /// Returns error if dimension mismatch or capacity exceeded.
    fn add(&mut self, id: String, vector: &[f32]) -> Result<()>;

    /// Add multiple vectors in batch.
    ///
    /// Default implementation calls `add` repeatedly.
    fn add_batch(&mut self, ids: Vec<String>, vectors: &[Vec<f32>]) -> Result<()> {
        for (id, vector) in ids.into_iter().zip(vectors.iter()) {
            self.add(id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    ///
    /// Vector of search results, sorted by distance (ascending).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;

    /// Search with pre-filter (IDs to consider).
    ///
    /// Default implementation searches all, then filters.
    fn search_with_ids(&self, query: &[f32], k: usize, ids: &[String]) -> Result<Vec<SearchResult>> {
        let results = self.search(query, self.len().min(k * 10))?;
        let id_set: std::collections::HashSet<_> = ids.iter().collect();
        Ok(results
            .into_iter()
            .filter(|r| id_set.contains(&r.id))
            .take(k)
            .collect())
    }

    /// Remove a vector from the index.
    ///
    /// # Returns
    ///
    /// `true` if vector was found and removed, `false` otherwise.
    fn remove(&mut self, id: &str) -> Result<bool>;

    /// Check if ID exists in index.
    fn contains(&self, id: &str) -> bool;

    /// Number of vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension.
    fn dimension(&self) -> usize;

    /// Distance type used by this index.
    fn distance_type(&self) -> DistanceType;

    /// Clear all vectors from the index.
    fn clear(&mut self);

    /// Get memory usage estimate in bytes.
    fn memory_usage(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_score() {
        // Distance 0 -> score 1.0
        let r = SearchResult::new("a".to_string(), 0.0, DistanceType::L2);
        assert!((r.score - 1.0).abs() < 1e-6);

        // Distance 1 -> score 0.5
        let r = SearchResult::new("b".to_string(), 1.0, DistanceType::L2);
        assert!((r.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config() {
        let config = IndexConfig::new(256)
            .with_distance(DistanceType::Cosine)
            .with_normalize(true);

        assert_eq!(config.dimension, 256);
        assert_eq!(config.distance_type, DistanceType::Cosine);
        assert!(config.normalize);
    }
}
