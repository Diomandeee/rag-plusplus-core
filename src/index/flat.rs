//! Flat Index (Exact Search)
//!
//! Brute-force nearest neighbor search. O(n) per query.
//! Best for: small datasets (<10k vectors), testing, ground truth.

use ahash::AHashMap;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use super::traits::{DistanceType, IndexConfig, SearchResult, VectorIndex};

/// Flat index using brute-force search.
///
/// Provides exact nearest neighbor results but O(n) per query.
#[derive(Debug)]
pub struct FlatIndex {
    /// Configuration
    config: IndexConfig,
    /// ID -> vector mapping
    vectors: AHashMap<String, Vec<f32>>,
    /// Whether to use parallel search
    parallel: bool,
    /// Parallel threshold (use parallel if len > this)
    parallel_threshold: usize,
}

impl FlatIndex {
    /// Create a new flat index.
    #[must_use]
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            vectors: AHashMap::new(),
            parallel: true,
            parallel_threshold: 1000,
        }
    }

    /// Create with specified capacity.
    #[must_use]
    pub fn with_capacity(config: IndexConfig, capacity: usize) -> Self {
        Self {
            config,
            vectors: AHashMap::with_capacity(capacity),
            parallel: true,
            parallel_threshold: 1000,
        }
    }

    /// Enable/disable parallel search.
    #[must_use]
    pub const fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Set parallel threshold.
    #[must_use]
    pub const fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Compute distance between two vectors.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_type {
            DistanceType::L2 => Self::l2_distance(a, b),
            DistanceType::InnerProduct => Self::inner_product(a, b),
            DistanceType::Cosine => Self::cosine_similarity(a, b),
        }
    }

    /// L2 (Euclidean) squared distance.
    #[inline]
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Inner product (dot product).
    #[inline]
    fn inner_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Cosine similarity.
    #[inline]
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Normalize a vector in-place.
    fn normalize(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Sequential search implementation.
    fn search_sequential(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Use a min-heap to track top-k (by distance for L2, by -score for IP/cosine)
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, String)> = BinaryHeap::new();
        
        let is_similarity = matches!(
            self.config.distance_type,
            DistanceType::InnerProduct | DistanceType::Cosine
        );

        for (id, vector) in &self.vectors {
            let dist = self.compute_distance(query, vector);
            let key = if is_similarity {
                // For similarity metrics, we want largest values
                Reverse(OrderedFloat(-dist))
            } else {
                // For distance metrics, we want smallest values
                Reverse(OrderedFloat(dist))
            };

            if heap.len() < k {
                heap.push((key, id.clone()));
            } else if let Some((top_key, _)) = heap.peek() {
                if key > *top_key {
                    heap.pop();
                    heap.push((key, id.clone()));
                }
            }
        }

        // Convert heap to sorted results
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|(Reverse(OrderedFloat(dist)), id)| {
                let actual_dist = if is_similarity { -dist } else { dist };
                SearchResult::new(id, actual_dist, self.config.distance_type)
            })
            .collect();

        // Sort by distance (ascending for L2, descending for similarity)
        if is_similarity {
            results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
        } else {
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        }

        results
    }

    /// Parallel search implementation.
    fn search_parallel(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let is_similarity = matches!(
            self.config.distance_type,
            DistanceType::InnerProduct | DistanceType::Cosine
        );

        // Compute distances in parallel
        let mut distances: Vec<_> = self.vectors
            .par_iter()
            .map(|(id, vector)| {
                let dist = self.compute_distance(query, vector);
                (id.clone(), dist)
            })
            .collect();

        // Sort by distance
        if is_similarity {
            distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else {
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        // Take top-k
        distances
            .into_iter()
            .take(k)
            .map(|(id, dist)| SearchResult::new(id, dist, self.config.distance_type))
            .collect()
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, id: String, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    vector.len()
                ),
            });
        }

        let mut vec = vector.to_vec();
        if self.config.normalize {
            Self::normalize(&mut vec);
        }

        self.vectors.insert(id, vec);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.config.dimension,
                    query.len()
                ),
            });
        }

        if self.vectors.is_empty() {
            return Ok(vec![]);
        }

        let k = k.min(self.vectors.len());

        // Normalize query if needed
        let query = if self.config.normalize {
            let mut q = query.to_vec();
            Self::normalize(&mut q);
            q
        } else {
            query.to_vec()
        };

        // Choose sequential or parallel based on index size
        let results = if self.parallel && self.vectors.len() > self.parallel_threshold {
            self.search_parallel(&query, k)
        } else {
            self.search_sequential(&query, k)
        };

        Ok(results)
    }

    fn remove(&mut self, id: &str) -> Result<bool> {
        Ok(self.vectors.remove(id).is_some())
    }

    fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn distance_type(&self) -> DistanceType {
        self.config.distance_type
    }

    fn clear(&mut self) {
        self.vectors.clear();
    }

    fn memory_usage(&self) -> usize {
        // Rough estimate: 
        // - Each vector: dim * 4 bytes
        // - Each ID: ~32 bytes average
        // - HashMap overhead: ~48 bytes per entry
        let per_entry = self.config.dimension * 4 + 32 + 48;
        self.vectors.len() * per_entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> FlatIndex {
        let config = IndexConfig::new(4);
        let mut index = FlatIndex::new(config).with_parallel(false);
        
        // Add test vectors
        index.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add("b".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.add("c".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();
        index.add("d".to_string(), &[0.5, 0.5, 0.0, 0.0]).unwrap();
        
        index
    }

    #[test]
    fn test_add_and_len() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);
        assert!(index.contains("a"));
        assert!(!index.contains("z"));
    }

    #[test]
    fn test_search_l2() {
        let index = create_test_index();
        
        // Query close to "a"
        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // Closest
    }

    #[test]
    fn test_search_cosine() {
        let config = IndexConfig::new(4).with_distance(DistanceType::Cosine);
        let mut index = FlatIndex::new(config).with_parallel(false);
        
        index.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add("b".to_string(), &[1.0, 1.0, 0.0, 0.0]).unwrap();
        index.add("c".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        
        assert_eq!(results[0].id, "a"); // Exact match
        assert!((results[0].distance - 1.0).abs() < 1e-6); // Cosine = 1.0
    }

    #[test]
    fn test_remove() {
        let mut index = create_test_index();
        
        assert!(index.remove("a").unwrap());
        assert!(!index.contains("a"));
        assert_eq!(index.len(), 3);
        
        assert!(!index.remove("z").unwrap()); // Not found
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = create_test_index();
        
        let result = index.add("e".to_string(), &[1.0, 2.0]); // Wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_search() {
        let config = IndexConfig::new(4);
        let index = FlatIndex::new(config);
        
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut index = create_test_index();
        assert_eq!(index.len(), 4);
        
        index.clear();
        assert!(index.is_empty());
    }
}
