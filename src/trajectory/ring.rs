//! Ring Topology for Trajectory Memory
//!
//! Provides a circular structure for organizing episodes in a trajectory.
//! Inspired by RCP (Ring Contextual Propagation) which uses ring topology
//! for multi-scale context organization.
//!
//! # Key Properties
//!
//! - **Circular ordering**: Enables wraparound attention
//! - **O(1) neighbor access**: Adjacent elements in constant time
//! - **Ring distance**: Shortest path in either direction
//! - **Cache-friendly**: Contiguous storage in Vec
//!
//! # Usage
//!
//! ```
//! use rag_plusplus_core::trajectory::Ring;
//!
//! let ring = Ring::new(vec![1, 2, 3, 4, 5]);
//!
//! // Get neighbors
//! let neighbors: Vec<_> = ring.neighbors(0, 2).collect();
//!
//! // Ring distance (shortest path)
//! let dist = ring.ring_distance(0, 3); // min(3, 2) = 2
//! ```

use std::ops::Index;

/// A ring topology over a collection of elements.
///
/// Provides circular access patterns and ring-based distance metrics.
/// Elements are stored contiguously for cache efficiency.
#[derive(Debug, Clone)]
pub struct Ring<T> {
    nodes: Vec<T>,
}

impl<T> Ring<T> {
    /// Create a new ring from a vector of nodes.
    ///
    /// # Panics
    ///
    /// Panics if nodes is empty.
    pub fn new(nodes: Vec<T>) -> Self {
        assert!(!nodes.is_empty(), "Ring cannot be empty");
        Self { nodes }
    }

    /// Create a ring with capacity for n nodes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Number of nodes in the ring.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the ring is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a reference to the node at the given index (wrapping).
    #[inline]
    pub fn get(&self, idx: usize) -> &T {
        &self.nodes[idx % self.len()]
    }

    /// Get a mutable reference to the node at the given index (wrapping).
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        let len = self.len();
        &mut self.nodes[idx % len]
    }

    /// Get the next index in the ring (wrapping).
    #[inline]
    pub fn next(&self, idx: usize) -> usize {
        (idx + 1) % self.len()
    }

    /// Get the previous index in the ring (wrapping).
    #[inline]
    pub fn prev(&self, idx: usize) -> usize {
        if idx == 0 {
            self.len() - 1
        } else {
            idx - 1
        }
    }

    /// Compute the ring distance between two indices.
    ///
    /// The ring distance is the shortest path between two nodes,
    /// going either forward or backward around the ring.
    ///
    /// # Example
    ///
    /// For a ring of size 5:
    /// - `ring_distance(0, 1)` = 1 (one step forward)
    /// - `ring_distance(0, 4)` = 1 (one step backward)
    /// - `ring_distance(0, 2)` = 2 (two steps either way)
    #[inline]
    pub fn ring_distance(&self, a: usize, b: usize) -> usize {
        let n = self.len();
        let a = a % n;
        let b = b % n;
        let forward = (b + n - a) % n;
        let backward = (a + n - b) % n;
        forward.min(backward)
    }

    /// Iterate over neighbors within a given radius.
    ///
    /// Returns an iterator yielding references to nodes within `radius`
    /// steps in both directions.
    ///
    /// # Example
    ///
    /// For a ring [0, 1, 2, 3, 4], neighbors(0, 1) yields nodes at indices 4, 1
    /// (previous and next).
    pub fn neighbors(&self, idx: usize, radius: usize) -> impl Iterator<Item = &T> {
        let n = self.len();
        let idx = idx % n;

        (1..=radius).flat_map(move |d| {
            let prev_idx = (idx + n - d) % n;
            let next_idx = (idx + d) % n;

            // Avoid duplicate if prev_idx == next_idx (happens in small rings)
            if prev_idx == next_idx {
                vec![&self.nodes[prev_idx]].into_iter()
            } else {
                vec![&self.nodes[prev_idx], &self.nodes[next_idx]].into_iter()
            }
        })
    }

    /// Get neighbor indices within a given radius.
    ///
    /// Returns indices (not values) of neighbors.
    pub fn neighbor_indices(&self, idx: usize, radius: usize) -> Vec<usize> {
        let n = self.len();
        let idx = idx % n;

        let mut indices = Vec::with_capacity(radius * 2);

        for d in 1..=radius {
            let prev_idx = (idx + n - d) % n;
            let next_idx = (idx + d) % n;

            if prev_idx == next_idx {
                indices.push(prev_idx);
            } else {
                indices.push(prev_idx);
                indices.push(next_idx);
            }
        }

        indices
    }

    /// Iterate over all nodes in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.nodes.iter()
    }

    /// Iterate over all nodes mutably in order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.nodes.iter_mut()
    }

    /// Iterate starting from a given index, wrapping around.
    pub fn iter_from(&self, start: usize) -> impl Iterator<Item = &T> {
        let n = self.len();
        let start = start % n;
        (0..n).map(move |i| &self.nodes[(start + i) % n])
    }

    /// Push a new node to the ring.
    pub fn push(&mut self, node: T) {
        self.nodes.push(node);
    }

    /// Get the underlying nodes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.nodes
    }

    /// Get the underlying nodes as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.nodes
    }

    /// Into the underlying vector.
    pub fn into_inner(self) -> Vec<T> {
        self.nodes
    }
}

impl<T> Index<usize> for Ring<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output {
        self.get(idx)
    }
}

impl<T> FromIterator<T> for Ring<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// A node in a ring with additional metadata.
#[derive(Debug, Clone)]
pub struct RingNode<T> {
    /// The value stored in this node
    pub value: T,
    /// Weight for attention/importance
    pub weight: f32,
    /// Optional connection strength to next node
    pub forward_weight: f32,
    /// Optional connection strength to previous node
    pub backward_weight: f32,
}

impl<T> RingNode<T> {
    /// Create a new ring node with default weights.
    pub fn new(value: T) -> Self {
        Self {
            value,
            weight: 1.0,
            forward_weight: 1.0,
            backward_weight: 1.0,
        }
    }

    /// Create a ring node with custom weight.
    pub fn with_weight(value: T, weight: f32) -> Self {
        Self {
            value,
            weight,
            forward_weight: 1.0,
            backward_weight: 1.0,
        }
    }
}

/// Ring of episodes for multi-scale trajectory organization.
pub type EpisodeRing<E> = Ring<RingNode<E>>;

/// Build a ring from a sequence of episodes with computed weights.
///
/// # Arguments
///
/// * `episodes` - Episode values
/// * `weights` - Weight for each episode (e.g., salience scores)
pub fn build_weighted_ring<T>(episodes: Vec<T>, weights: &[f32]) -> Ring<RingNode<T>> {
    assert_eq!(episodes.len(), weights.len(), "Episode and weight counts must match");

    let nodes: Vec<RingNode<T>> = episodes
        .into_iter()
        .zip(weights.iter())
        .map(|(e, &w)| RingNode::with_weight(e, w))
        .collect();

    Ring::new(nodes)
}

/// Dual Ring Structure for IRCP/RCP
///
/// Represents the same episodes in two different orderings:
/// - **Temporal Ring (RCP)**: Ordered by causal/temporal flow
/// - **Influence Ring (IRCP)**: Ordered by influence/attention weight
///
/// This enables efficient traversal in both directions:
/// - Forward (RCP): "What context led to this response?"
/// - Inverse (IRCP): "What did this response influence?"
///
/// # Example
///
/// ```
/// use rag_plusplus_core::trajectory::{DualRing, DualRingNode};
///
/// // Create nodes with temporal order and influence weights
/// let nodes = vec![
///     DualRingNode::new(0, "episode_a", 0.8),  // temporal_idx=0, influence=0.8
///     DualRingNode::new(1, "episode_b", 0.3),  // temporal_idx=1, influence=0.3
///     DualRingNode::new(2, "episode_c", 0.9),  // temporal_idx=2, influence=0.9
///     DualRingNode::new(3, "episode_d", 0.5),  // temporal_idx=3, influence=0.5
/// ];
///
/// let dual = DualRing::new(nodes);
///
/// // Traverse temporally (RCP direction)
/// let temporal_order: Vec<_> = dual.iter_temporal().map(|n| n.value).collect();
/// // ["episode_a", "episode_b", "episode_c", "episode_d"]
///
/// // Traverse by influence (IRCP direction)
/// let influence_order: Vec<_> = dual.iter_by_influence().map(|n| n.value).collect();
/// // ["episode_c", "episode_a", "episode_d", "episode_b"] (by descending influence)
/// ```
#[derive(Debug, Clone)]
pub struct DualRing<T> {
    /// Nodes stored in temporal order
    nodes: Vec<DualRingNode<T>>,
    /// Indices sorted by influence (descending)
    influence_order: Vec<usize>,
}

/// A node in a dual ring with both temporal position and influence weight.
#[derive(Debug, Clone)]
pub struct DualRingNode<T> {
    /// The value stored in this node
    pub value: T,
    /// Temporal index (position in time)
    pub temporal_idx: usize,
    /// Influence weight (how much this node influenced others)
    pub influence: f32,
    /// Attention received (how much attention this node got from others)
    pub attention_received: f32,
    /// Attention given (how much attention this node gave to others)
    pub attention_given: f32,
}

impl<T> DualRingNode<T> {
    /// Create a new dual ring node.
    pub fn new(temporal_idx: usize, value: T, influence: f32) -> Self {
        Self {
            value,
            temporal_idx,
            influence,
            attention_received: 0.0,
            attention_given: 0.0,
        }
    }

    /// Create with full attention data.
    pub fn with_attention(
        temporal_idx: usize,
        value: T,
        influence: f32,
        attention_received: f32,
        attention_given: f32,
    ) -> Self {
        Self {
            value,
            temporal_idx,
            influence,
            attention_received,
            attention_given,
        }
    }
}

impl<T> DualRing<T> {
    /// Create a dual ring from nodes.
    ///
    /// Nodes should be provided in temporal order.
    pub fn new(nodes: Vec<DualRingNode<T>>) -> Self {
        assert!(!nodes.is_empty(), "DualRing cannot be empty");

        // Build influence order (indices sorted by descending influence)
        let mut influence_order: Vec<usize> = (0..nodes.len()).collect();
        influence_order.sort_by(|&a, &b| {
            nodes[b].influence.partial_cmp(&nodes[a].influence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self { nodes, influence_order }
    }

    /// Number of nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ========== RCP (Temporal) Operations ==========

    /// Get node by temporal index (wrapping).
    #[inline]
    pub fn get_temporal(&self, idx: usize) -> &DualRingNode<T> {
        &self.nodes[idx % self.len()]
    }

    /// Next in temporal order.
    #[inline]
    pub fn temporal_next(&self, idx: usize) -> usize {
        (idx + 1) % self.len()
    }

    /// Previous in temporal order.
    #[inline]
    pub fn temporal_prev(&self, idx: usize) -> usize {
        if idx == 0 { self.len() - 1 } else { idx - 1 }
    }

    /// Temporal ring distance.
    #[inline]
    pub fn temporal_distance(&self, a: usize, b: usize) -> usize {
        let n = self.len();
        let a = a % n;
        let b = b % n;
        let forward = (b + n - a) % n;
        let backward = (a + n - b) % n;
        forward.min(backward)
    }

    /// Iterate in temporal order (RCP direction).
    pub fn iter_temporal(&self) -> impl Iterator<Item = &DualRingNode<T>> {
        self.nodes.iter()
    }

    /// Iterate temporally from a starting point.
    pub fn iter_temporal_from(&self, start: usize) -> impl Iterator<Item = &DualRingNode<T>> {
        let n = self.len();
        let start = start % n;
        (0..n).map(move |i| &self.nodes[(start + i) % n])
    }

    /// Get temporal neighbors within radius.
    pub fn temporal_neighbors(&self, idx: usize, radius: usize) -> impl Iterator<Item = &DualRingNode<T>> {
        let n = self.len();
        let idx = idx % n;

        (1..=radius).flat_map(move |d| {
            let prev_idx = (idx + n - d) % n;
            let next_idx = (idx + d) % n;

            if prev_idx == next_idx {
                vec![&self.nodes[prev_idx]].into_iter()
            } else {
                vec![&self.nodes[prev_idx], &self.nodes[next_idx]].into_iter()
            }
        })
    }

    // ========== IRCP (Influence) Operations ==========

    /// Get node by influence rank (0 = highest influence).
    #[inline]
    pub fn get_by_influence_rank(&self, rank: usize) -> &DualRingNode<T> {
        let idx = self.influence_order[rank % self.len()];
        &self.nodes[idx]
    }

    /// Get influence rank of a temporal index.
    pub fn influence_rank_of(&self, temporal_idx: usize) -> usize {
        self.influence_order.iter()
            .position(|&idx| idx == temporal_idx)
            .unwrap_or(self.len())
    }

    /// Iterate by influence (IRCP direction, highest first).
    pub fn iter_by_influence(&self) -> impl Iterator<Item = &DualRingNode<T>> {
        self.influence_order.iter().map(move |&idx| &self.nodes[idx])
    }

    /// Get top-k most influential nodes.
    pub fn top_influential(&self, k: usize) -> impl Iterator<Item = &DualRingNode<T>> {
        self.influence_order.iter()
            .take(k)
            .map(move |&idx| &self.nodes[idx])
    }

    /// Influence ring distance (distance in influence-sorted order).
    pub fn influence_distance(&self, a_temporal: usize, b_temporal: usize) -> usize {
        let a_rank = self.influence_rank_of(a_temporal);
        let b_rank = self.influence_rank_of(b_temporal);

        let n = self.len();
        let forward = (b_rank + n - a_rank) % n;
        let backward = (a_rank + n - b_rank) % n;
        forward.min(backward)
    }

    /// Get influence neighbors (nodes with similar influence).
    pub fn influence_neighbors(&self, temporal_idx: usize, radius: usize) -> impl Iterator<Item = &DualRingNode<T>> {
        let rank = self.influence_rank_of(temporal_idx);
        let n = self.len();

        (1..=radius).flat_map(move |d| {
            let prev_rank = if rank >= d { rank - d } else { n - (d - rank) };
            let next_rank = (rank + d) % n;

            let prev_idx = self.influence_order[prev_rank];
            let next_idx = self.influence_order[next_rank];

            if prev_idx == next_idx {
                vec![&self.nodes[prev_idx]].into_iter()
            } else {
                vec![&self.nodes[prev_idx], &self.nodes[next_idx]].into_iter()
            }
        })
    }

    // ========== Cross-Ring Operations ==========

    /// Find the temporal distance from high-influence to low-influence nodes.
    ///
    /// This measures how "spread out" influence is temporally.
    /// Low values mean influence clusters in time; high values mean it's distributed.
    pub fn influence_temporal_spread(&self) -> f32 {
        if self.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let count = self.len().min(5); // Top 5 influential

        for i in 0..count {
            for j in (i + 1)..count {
                let idx_i = self.influence_order[i];
                let idx_j = self.influence_order[j];
                total_distance += self.temporal_distance(idx_i, idx_j) as f32;
            }
        }

        let pairs = (count * (count - 1)) / 2;
        if pairs > 0 {
            total_distance / pairs as f32
        } else {
            0.0
        }
    }

    /// Compute attention flow from earlier to later nodes (RCP forward flow).
    ///
    /// Returns total influence flowing from past to future.
    pub fn forward_attention_flow(&self) -> f32 {
        self.nodes.iter()
            .map(|n| n.attention_given)
            .sum()
    }

    /// Compute attention flow from later to earlier nodes (IRCP inverse flow).
    ///
    /// Returns total influence flowing from future to past (attribution).
    pub fn inverse_attention_flow(&self) -> f32 {
        self.nodes.iter()
            .map(|n| n.attention_received)
            .sum()
    }

    /// Update influence weight for a node.
    pub fn update_influence(&mut self, temporal_idx: usize, new_influence: f32) {
        let idx = temporal_idx % self.len();
        self.nodes[idx].influence = new_influence;

        // Re-sort influence order
        self.influence_order.sort_by(|&a, &b| {
            self.nodes[b].influence.partial_cmp(&self.nodes[a].influence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get underlying nodes.
    pub fn as_slice(&self) -> &[DualRingNode<T>] {
        &self.nodes
    }
}

/// Build a dual ring from episodes with temporal positions and influence scores.
pub fn build_dual_ring<T>(
    episodes: Vec<T>,
    influences: &[f32],
) -> DualRing<T> {
    assert_eq!(episodes.len(), influences.len(), "Episode and influence counts must match");

    let nodes: Vec<DualRingNode<T>> = episodes
        .into_iter()
        .enumerate()
        .zip(influences.iter())
        .map(|((idx, value), &influence)| DualRingNode::new(idx, value, influence))
        .collect();

    DualRing::new(nodes)
}

/// Build a dual ring with full attention data.
pub fn build_dual_ring_with_attention<T>(
    episodes: Vec<T>,
    influences: &[f32],
    attention_received: &[f32],
    attention_given: &[f32],
) -> DualRing<T> {
    assert_eq!(episodes.len(), influences.len());
    assert_eq!(episodes.len(), attention_received.len());
    assert_eq!(episodes.len(), attention_given.len());

    let nodes: Vec<DualRingNode<T>> = episodes
        .into_iter()
        .enumerate()
        .zip(influences.iter())
        .zip(attention_received.iter())
        .zip(attention_given.iter())
        .map(|((((idx, value), &inf), &recv), &given)| {
            DualRingNode::with_attention(idx, value, inf, recv, given)
        })
        .collect();

    DualRing::new(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ring() {
        let ring = Ring::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(ring.len(), 5);
        assert!(!ring.is_empty());
    }

    #[test]
    #[should_panic(expected = "Ring cannot be empty")]
    fn test_empty_ring_panics() {
        let _ring: Ring<i32> = Ring::new(vec![]);
    }

    #[test]
    fn test_get_wrapping() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        assert_eq!(*ring.get(0), 0);
        assert_eq!(*ring.get(4), 4);
        assert_eq!(*ring.get(5), 0); // Wraps
        assert_eq!(*ring.get(7), 2); // Wraps
    }

    #[test]
    fn test_next_prev() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        assert_eq!(ring.next(0), 1);
        assert_eq!(ring.next(4), 0); // Wraps
        assert_eq!(ring.prev(0), 4); // Wraps
        assert_eq!(ring.prev(3), 2);
    }

    #[test]
    fn test_ring_distance() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        // Same node
        assert_eq!(ring.ring_distance(0, 0), 0);

        // Adjacent
        assert_eq!(ring.ring_distance(0, 1), 1);
        assert_eq!(ring.ring_distance(0, 4), 1); // Backward is shorter

        // Two steps
        assert_eq!(ring.ring_distance(0, 2), 2);
        assert_eq!(ring.ring_distance(0, 3), 2); // Either direction

        // Wrapping
        assert_eq!(ring.ring_distance(1, 4), 2);
    }

    #[test]
    fn test_neighbors() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        // Neighbors of 0 with radius 1: prev=4, next=1
        let neighbors: Vec<_> = ring.neighbors(0, 1).copied().collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&4));
        assert!(neighbors.contains(&1));

        // Neighbors of 0 with radius 2: prev=3,4 and next=1,2
        let neighbors: Vec<_> = ring.neighbors(0, 2).copied().collect();
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&3));
        assert!(neighbors.contains(&4));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
    }

    #[test]
    fn test_neighbor_indices() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        let indices = ring.neighbor_indices(0, 1);
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&4));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_iter_from() {
        let ring = Ring::new(vec![0, 1, 2, 3, 4]);

        let from_2: Vec<_> = ring.iter_from(2).copied().collect();
        assert_eq!(from_2, vec![2, 3, 4, 0, 1]);

        let from_0: Vec<_> = ring.iter_from(0).copied().collect();
        assert_eq!(from_0, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_index_operator() {
        let ring = Ring::new(vec![10, 20, 30]);

        assert_eq!(ring[0], 10);
        assert_eq!(ring[2], 30);
        assert_eq!(ring[3], 10); // Wraps
    }

    #[test]
    fn test_from_iterator() {
        let ring: Ring<i32> = (0..5).collect();
        assert_eq!(ring.len(), 5);
        assert_eq!(*ring.get(0), 0);
        assert_eq!(*ring.get(4), 4);
    }

    #[test]
    fn test_ring_node() {
        let node = RingNode::new(42);
        assert_eq!(node.value, 42);
        assert!((node.weight - 1.0).abs() < 1e-6);

        let weighted = RingNode::with_weight("hello", 0.5);
        assert_eq!(weighted.value, "hello");
        assert!((weighted.weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_build_weighted_ring() {
        let episodes = vec!["a", "b", "c"];
        let weights = vec![0.1, 0.5, 0.9];

        let ring = build_weighted_ring(episodes, &weights);

        assert_eq!(ring.len(), 3);
        assert_eq!(ring.get(0).value, "a");
        assert!((ring.get(0).weight - 0.1).abs() < 1e-6);
        assert!((ring.get(2).weight - 0.9).abs() < 1e-6);
    }

    // ========== Dual Ring Tests ==========

    #[test]
    fn test_dual_ring_creation() {
        let nodes = vec![
            DualRingNode::new(0, "a", 0.3),
            DualRingNode::new(1, "b", 0.9),
            DualRingNode::new(2, "c", 0.5),
            DualRingNode::new(3, "d", 0.1),
        ];

        let dual = DualRing::new(nodes);
        assert_eq!(dual.len(), 4);
    }

    #[test]
    fn test_dual_ring_temporal_order() {
        let nodes = vec![
            DualRingNode::new(0, "first", 0.3),
            DualRingNode::new(1, "second", 0.9),
            DualRingNode::new(2, "third", 0.5),
        ];

        let dual = DualRing::new(nodes);

        // Temporal iteration should preserve order
        let temporal: Vec<_> = dual.iter_temporal().map(|n| n.value).collect();
        assert_eq!(temporal, vec!["first", "second", "third"]);
    }

    #[test]
    fn test_dual_ring_influence_order() {
        let nodes = vec![
            DualRingNode::new(0, "low", 0.1),
            DualRingNode::new(1, "high", 0.9),
            DualRingNode::new(2, "medium", 0.5),
        ];

        let dual = DualRing::new(nodes);

        // Influence iteration should be sorted by influence (descending)
        let influence: Vec<_> = dual.iter_by_influence().map(|n| n.value).collect();
        assert_eq!(influence, vec!["high", "medium", "low"]);
    }

    #[test]
    fn test_dual_ring_top_influential() {
        let nodes = vec![
            DualRingNode::new(0, "a", 0.2),
            DualRingNode::new(1, "b", 0.8),
            DualRingNode::new(2, "c", 0.5),
            DualRingNode::new(3, "d", 0.9),
            DualRingNode::new(4, "e", 0.1),
        ];

        let dual = DualRing::new(nodes);

        let top2: Vec<_> = dual.top_influential(2).map(|n| n.value).collect();
        assert_eq!(top2, vec!["d", "b"]); // 0.9, 0.8
    }

    #[test]
    fn test_dual_ring_influence_rank() {
        let nodes = vec![
            DualRingNode::new(0, "a", 0.1),
            DualRingNode::new(1, "b", 0.9),
            DualRingNode::new(2, "c", 0.5),
        ];

        let dual = DualRing::new(nodes);

        // Temporal idx 1 has highest influence (0.9) → rank 0
        assert_eq!(dual.influence_rank_of(1), 0);
        // Temporal idx 2 has medium influence (0.5) → rank 1
        assert_eq!(dual.influence_rank_of(2), 1);
        // Temporal idx 0 has lowest influence (0.1) → rank 2
        assert_eq!(dual.influence_rank_of(0), 2);
    }

    #[test]
    fn test_dual_ring_temporal_distance() {
        let nodes: Vec<DualRingNode<i32>> = (0..5)
            .map(|i| DualRingNode::new(i, i as i32, 0.5))
            .collect();

        let dual = DualRing::new(nodes);

        assert_eq!(dual.temporal_distance(0, 1), 1);
        assert_eq!(dual.temporal_distance(0, 4), 1); // Wrap around
        assert_eq!(dual.temporal_distance(0, 2), 2);
    }

    #[test]
    fn test_dual_ring_influence_distance() {
        let nodes = vec![
            DualRingNode::new(0, "a", 0.1), // Rank 2
            DualRingNode::new(1, "b", 0.9), // Rank 0
            DualRingNode::new(2, "c", 0.5), // Rank 1
        ];

        let dual = DualRing::new(nodes);

        // Distance between rank 0 (temporal 1) and rank 1 (temporal 2)
        assert_eq!(dual.influence_distance(1, 2), 1);
        // Distance between rank 0 (temporal 1) and rank 2 (temporal 0)
        assert_eq!(dual.influence_distance(1, 0), 1); // Wrap: 2→0 or 0→2
    }

    #[test]
    fn test_dual_ring_temporal_neighbors() {
        let nodes: Vec<DualRingNode<i32>> = (0..5)
            .map(|i| DualRingNode::new(i, i as i32, 0.5))
            .collect();

        let dual = DualRing::new(nodes);

        // Neighbors of 0 with radius 1: prev=4, next=1
        let neighbors: Vec<_> = dual.temporal_neighbors(0, 1).map(|n| n.value).collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&4));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn test_dual_ring_update_influence() {
        let nodes = vec![
            DualRingNode::new(0, "a", 0.1),
            DualRingNode::new(1, "b", 0.9),
            DualRingNode::new(2, "c", 0.5),
        ];

        let mut dual = DualRing::new(nodes);

        // Initially, b (idx 1) is most influential
        assert_eq!(dual.get_by_influence_rank(0).value, "b");

        // Update a's influence to be highest
        dual.update_influence(0, 0.99);

        // Now a (idx 0) should be most influential
        assert_eq!(dual.get_by_influence_rank(0).value, "a");
    }

    #[test]
    fn test_dual_ring_influence_temporal_spread() {
        // Use 6+ nodes so top 5 influential selection matters
        // Clustered: high influence nodes are adjacent (positions 0, 1, 2)
        let clustered = vec![
            DualRingNode::new(0, "a", 0.95), // high - position 0
            DualRingNode::new(1, "b", 0.90), // high - position 1
            DualRingNode::new(2, "c", 0.85), // high - position 2
            DualRingNode::new(3, "d", 0.80), // high - position 3
            DualRingNode::new(4, "e", 0.75), // top 5 cutoff
            DualRingNode::new(5, "f", 0.10), // excluded from top 5
            DualRingNode::new(6, "g", 0.10), // excluded from top 5
        ];
        let dual_clustered = DualRing::new(clustered);
        // Top 5: positions 0,1,2,3,4 - all adjacent, avg distance ~1.6

        // Spread: high influence nodes are spread apart
        let spread = vec![
            DualRingNode::new(0, "a", 0.95), // high - position 0
            DualRingNode::new(1, "b", 0.10), // low
            DualRingNode::new(2, "c", 0.90), // high - position 2
            DualRingNode::new(3, "d", 0.10), // low
            DualRingNode::new(4, "e", 0.85), // high - position 4
            DualRingNode::new(5, "f", 0.10), // low
            DualRingNode::new(6, "g", 0.80), // high - position 6
        ];
        let dual_spread = DualRing::new(spread);
        // Top 5: positions 0,2,4,6, and one of {1,3,5} - more spread, avg distance ~2.0

        // Spread should have higher value than clustered (more temporal distance between influential nodes)
        assert!(
            dual_spread.influence_temporal_spread() >= dual_clustered.influence_temporal_spread(),
            "spread={}, clustered={}",
            dual_spread.influence_temporal_spread(),
            dual_clustered.influence_temporal_spread()
        );
    }

    #[test]
    fn test_build_dual_ring() {
        let episodes = vec!["a", "b", "c"];
        let influences = vec![0.3, 0.9, 0.5];

        let dual = build_dual_ring(episodes, &influences);

        assert_eq!(dual.len(), 3);
        assert_eq!(dual.get_temporal(0).value, "a");
        assert!((dual.get_temporal(1).influence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_build_dual_ring_with_attention() {
        let episodes = vec!["a", "b"];
        let influences = vec![0.5, 0.8];
        let received = vec![0.3, 0.7];
        let given = vec![0.4, 0.6];

        let dual = build_dual_ring_with_attention(episodes, &influences, &received, &given);

        assert_eq!(dual.len(), 2);
        assert!((dual.get_temporal(0).attention_received - 0.3).abs() < 1e-6);
        assert!((dual.get_temporal(1).attention_given - 0.6).abs() < 1e-6);
    }
}
