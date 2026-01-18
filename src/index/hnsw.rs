//! HNSW Index (Hierarchical Navigable Small World)
//!
//! Approximate nearest neighbor search with logarithmic query time.
//! Production default for RAG++.
//!
//! # Performance Characteristics
//!
//! - Build: O(n log n)
//! - Query: O(log n) with ef_search
//! - Memory: O(n * M) where M is connectivity
//!
//! # Configuration
//!
//! - `m`: Connections per layer (higher = better recall, more memory)
//! - `ef_construction`: Build-time search depth (higher = better quality)
//! - `ef_search`: Query-time search depth (higher = better recall, slower)

use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use super::traits::{DistanceType, IndexConfig, SearchResult, VectorIndex};

/// HNSW configuration parameters.
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Base index configuration
    pub base: IndexConfig,
    /// Number of connections per layer (default: 16)
    pub m: usize,
    /// Maximum connections for layer 0 (default: 2 * m)
    pub m_max0: usize,
    /// Construction-time search depth (default: 200)
    pub ef_construction: usize,
    /// Default query-time search depth (default: 128)
    pub ef_search: usize,
    /// Level multiplier (default: 1 / ln(m))
    pub ml: f64,
}

impl HNSWConfig {
    /// Create default HNSW config for given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        let m = 16;
        Self {
            base: IndexConfig::new(dimension),
            m,
            m_max0: 2 * m,
            ef_construction: 200,
            ef_search: 128,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Set M parameter.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max0 = 2 * m;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set ef_construction.
    #[must_use]
    pub const fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set distance type.
    #[must_use]
    pub fn with_distance(mut self, distance_type: DistanceType) -> Self {
        self.base.distance_type = distance_type;
        self
    }
}

/// Node in the HNSW graph.
#[derive(Debug, Clone)]
struct HNSWNode {
    /// Node ID
    id: String,
    /// Vector data
    vector: Vec<f32>,
    /// Max level this node appears in (used for graph traversal)
    #[allow(dead_code)]
    level: usize,
    /// Neighbors at each level (level -> set of neighbor indices)
    neighbors: Vec<AHashSet<usize>>,
}

/// HNSW Index implementation.
///
/// Thread-safe approximate nearest neighbor index.
#[derive(Debug)]
pub struct HNSWIndex {
    /// Configuration
    config: HNSWConfig,
    /// All nodes
    nodes: Vec<HNSWNode>,
    /// ID to index mapping
    id_to_idx: AHashMap<String, usize>,
    /// Entry point (highest level node index)
    entry_point: Option<usize>,
    /// Maximum level in the graph
    max_level: usize,
    /// RNG for level generation
    rng: parking_lot::Mutex<rand::rngs::SmallRng>,
}

impl HNSWIndex {
    /// Create a new HNSW index.
    #[must_use]
    pub fn new(config: HNSWConfig) -> Self {
        use rand::SeedableRng;
        Self {
            config,
            nodes: Vec::new(),
            id_to_idx: AHashMap::new(),
            entry_point: None,
            max_level: 0,
            rng: parking_lot::Mutex::new(rand::rngs::SmallRng::from_entropy()),
        }
    }

    /// Generate random level for new node.
    fn random_level(&self) -> usize {
        let mut rng = self.rng.lock();
        let mut level = 0;
        while rng.gen::<f64>() < self.config.ml && level < 16 {
            level += 1;
        }
        level
    }

    /// Compute distance between two vectors.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.base.distance_type {
            DistanceType::L2 => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceType::InnerProduct => {
                // Negative for min-heap compatibility
                -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
            }
            DistanceType::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (norm_a * norm_b))
                }
            }
        }
    }

    /// Search layer for nearest neighbors.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: Vec<usize>,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited: AHashSet<usize> = entry_points.iter().copied().collect();
        
        // Min-heap for candidates (distance, idx)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
        
        // Max-heap for results (distance, idx)  
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        // Initialize with entry points
        for &ep in &entry_points {
            let dist = self.distance(query, &self.nodes[ep].vector);
            candidates.push(Reverse((OrderedFloat(dist), ep)));
            results.push((OrderedFloat(dist), ep));
        }

        while let Some(Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            // Get furthest in results
            let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
            
            if c_dist > f_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            if level < self.nodes[c_idx].neighbors.len() {
                for &neighbor_idx in &self.nodes[c_idx].neighbors[level] {
                    if visited.insert(neighbor_idx) {
                        let dist = self.distance(query, &self.nodes[neighbor_idx].vector);
                        let f_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
                        
                        if dist < f_dist || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), neighbor_idx)));
                            results.push((OrderedFloat(dist), neighbor_idx));
                            
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Return sorted results
        let mut result_vec: Vec<_> = results.into_iter().map(|(d, idx)| (d.0, idx)).collect();
        result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result_vec
    }

    /// Select neighbors using simple heuristic.
    fn select_neighbors(&self, candidates: &[(f32, usize)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(_, idx)| *idx).collect()
    }

    /// Get max connections for a level.
    fn get_max_connections(&self, level: usize) -> usize {
        if level == 0 {
            self.config.m_max0
        } else {
            self.config.m
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn add(&mut self, id: String, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.base.dimension {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.config.base.dimension,
                    vector.len()
                ),
            });
        }

        if self.id_to_idx.contains_key(&id) {
            return Err(Error::DuplicateRecord { record_id: id });
        }

        let level = self.random_level();
        let new_idx = self.nodes.len();

        // Create new node
        let mut node = HNSWNode {
            id: id.clone(),
            vector: vector.to_vec(),
            level,
            neighbors: vec![AHashSet::new(); level + 1],
        };

        // First node
        if self.entry_point.is_none() {
            self.nodes.push(node);
            self.id_to_idx.insert(id, new_idx);
            self.entry_point = Some(new_idx);
            self.max_level = level;
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();
        let mut curr_ep = vec![entry_point];

        // Traverse from top to insertion level + 1
        for lc in (level + 1..=self.max_level).rev() {
            let nearest = self.search_layer(vector, curr_ep.clone(), 1, lc);
            if !nearest.is_empty() {
                curr_ep = vec![nearest[0].1];
            }
        }

        // Insert at each level from level down to 0
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(
                vector,
                curr_ep.clone(),
                self.config.ef_construction,
                lc,
            );

            let m = self.get_max_connections(lc);
            let neighbors = self.select_neighbors(&candidates, m);

            // Add bidirectional connections
            node.neighbors[lc] = neighbors.iter().copied().collect();

            for &neighbor_idx in &neighbors {
                if lc < self.nodes[neighbor_idx].neighbors.len() {
                    self.nodes[neighbor_idx].neighbors[lc].insert(new_idx);

                    // Prune if too many connections
                    if self.nodes[neighbor_idx].neighbors[lc].len() > m {
                        let neighbor_vec = &self.nodes[neighbor_idx].vector;
                        // Exclude new_idx from distance calc since node not pushed yet
                        let new_node_vec = vector;
                        let mut scored: Vec<_> = self.nodes[neighbor_idx].neighbors[lc]
                            .iter()
                            .map(|&idx| {
                                let dist = if idx == new_idx {
                                    self.distance(neighbor_vec, new_node_vec)
                                } else {
                                    self.distance(neighbor_vec, &self.nodes[idx].vector)
                                };
                                (dist, idx)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        self.nodes[neighbor_idx].neighbors[lc] =
                            scored.into_iter().take(m).map(|(_, idx)| idx).collect();
                    }
                }
            }

            if !candidates.is_empty() {
                curr_ep = vec![candidates[0].1];
            }
        }

        self.nodes.push(node);
        self.id_to_idx.insert(id, new_idx);

        // Update entry point if new node has higher level
        if level > self.max_level {
            self.entry_point = Some(new_idx);
            self.max_level = level;
        }

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.base.dimension {
            return Err(Error::InvalidQuery {
                reason: format!(
                    "Query dimension mismatch: expected {}, got {}",
                    self.config.base.dimension,
                    query.len()
                ),
            });
        }

        if self.nodes.is_empty() {
            return Ok(vec![]);
        }

        let entry_point = self.entry_point.unwrap();
        let mut curr_ep = vec![entry_point];

        // Traverse from top to level 1
        for lc in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, curr_ep.clone(), 1, lc);
            if !nearest.is_empty() {
                curr_ep = vec![nearest[0].1];
            }
        }

        // Search at level 0 with ef_search
        let results = self.search_layer(query, curr_ep, self.config.ef_search, 0);

        // Convert to SearchResult
        let k = k.min(results.len());
        Ok(results
            .into_iter()
            .take(k)
            .map(|(dist, idx)| {
                let actual_dist = match self.config.base.distance_type {
                    DistanceType::InnerProduct => -dist,
                    DistanceType::Cosine => 1.0 - dist,
                    DistanceType::L2 => dist,
                };
                SearchResult::new(
                    self.nodes[idx].id.clone(),
                    actual_dist,
                    self.config.base.distance_type,
                )
            })
            .collect())
    }

    fn remove(&mut self, id: &str) -> Result<bool> {
        // Note: Full removal in HNSW is complex. 
        // For production, use soft-delete + periodic rebuild.
        if let Some(&idx) = self.id_to_idx.get(id) {
            // Remove from neighbor lists
            for node in &mut self.nodes {
                for neighbors in &mut node.neighbors {
                    neighbors.remove(&idx);
                }
            }
            self.id_to_idx.remove(id);
            // Mark as deleted (don't actually remove to preserve indices)
            self.nodes[idx].id = String::new();
            self.nodes[idx].vector.clear();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn contains(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }

    fn len(&self) -> usize {
        self.id_to_idx.len()
    }

    fn dimension(&self) -> usize {
        self.config.base.dimension
    }

    fn distance_type(&self) -> DistanceType {
        self.config.base.distance_type
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.id_to_idx.clear();
        self.entry_point = None;
        self.max_level = 0;
    }

    fn memory_usage(&self) -> usize {
        let node_size = self.config.base.dimension * 4 + 64; // vector + overhead
        let neighbor_size = self.config.m * 8 * 2; // avg neighbors * pointer size * 2
        self.nodes.len() * (node_size + neighbor_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> HNSWIndex {
        let config = HNSWConfig::new(4)
            .with_m(4)
            .with_ef_construction(16)
            .with_ef_search(16);
        let mut index = HNSWIndex::new(config);
        
        index.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add("b".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.add("c".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();
        index.add("d".to_string(), &[0.5, 0.5, 0.0, 0.0]).unwrap();
        index.add("e".to_string(), &[0.9, 0.1, 0.0, 0.0]).unwrap();
        
        index
    }

    #[test]
    fn test_add_and_search() {
        let index = create_test_index();
        
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        
        assert!(!results.is_empty());
        // "a" or "e" should be closest
        assert!(results[0].id == "a" || results[0].id == "e");
    }

    #[test]
    fn test_recall() {
        let config = HNSWConfig::new(8).with_m(8).with_ef_search(32);
        let mut index = HNSWIndex::new(config);
        
        // Add 100 random vectors
        for i in 0..100 {
            let vec: Vec<f32> = (0..8).map(|j| ((i * j) % 100) as f32 / 100.0).collect();
            index.add(format!("v{}", i), &vec).unwrap();
        }
        
        // Search should return results
        let results = index.search(&[0.5; 8], 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_duplicate_id() {
        let mut index = create_test_index();
        
        let result = index.add("a".to_string(), &[0.0, 0.0, 0.0, 1.0]);
        assert!(result.is_err());
    }
}
