//! Coordinate Tree with LCA (Lowest Common Ancestor) Algorithm
//!
//! Provides efficient tree operations for trajectory coordinates using
//! binary lifting for O(log n) LCA queries.
//!
//! # Overview
//!
//! The CoordinateTree stores trajectory nodes in a tree structure and provides:
//! - O(log n) LCA (Lowest Common Ancestor) queries
//! - O(log n) path distance computation
//! - Subtree extraction
//! - Ancestor/descendant queries
//!
//! # Binary Lifting
//!
//! Binary lifting precomputes `ancestor[node][k]` = 2^k-th ancestor of node.
//! This enables O(log n) LCA by:
//! 1. Lift the deeper node to the same depth
//! 2. Binary search for LCA by lifting both nodes
//!
//! # Usage
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::coordinate_tree::CoordinateTree;
//!
//! // Build tree from parent relationships
//! let mut tree = CoordinateTree::new(10);
//! tree.add_node(0, None, coord_0);        // Root
//! tree.add_node(1, Some(0), coord_1);     // Child of root
//! tree.add_node(2, Some(0), coord_2);     // Child of root
//! tree.add_node(3, Some(1), coord_3);     // Grandchild
//! tree.preprocess();                       // Build LCA structure
//!
//! let lca = tree.lca(3, 2);  // LCA of nodes 3 and 2
//! let dist = tree.path_distance(3, 2);  // Distance via LCA
//! ```

use crate::trajectory::TrajectoryCoordinate5D;
use std::collections::HashMap;

/// A node in the coordinate tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node identifier
    pub id: usize,
    /// Parent node ID (None for root)
    pub parent: Option<usize>,
    /// Child node IDs
    pub children: Vec<usize>,
    /// Depth in tree (root = 0)
    pub depth: usize,
    /// Coordinate at this node
    pub coordinate: TrajectoryCoordinate5D,
}

impl TreeNode {
    /// Create a new tree node.
    pub fn new(id: usize, parent: Option<usize>, coordinate: TrajectoryCoordinate5D) -> Self {
        Self {
            id,
            parent,
            children: Vec::new(),
            depth: 0,
            coordinate,
        }
    }

    /// Check if this is the root node.
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Check if this is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Coordinate tree with LCA support.
///
/// Uses binary lifting for O(log n) LCA queries after O(n log n) preprocessing.
#[derive(Debug, Clone)]
pub struct CoordinateTree {
    /// Nodes indexed by ID
    nodes: HashMap<usize, TreeNode>,
    /// Root node ID
    root: Option<usize>,
    /// Maximum depth for binary lifting (log2(n))
    max_log: usize,
    /// Binary lifting table: ancestor[node_id][k] = 2^k ancestor
    /// Uses HashMap for sparse storage
    ancestor: HashMap<usize, Vec<Option<usize>>>,
    /// Whether the tree has been preprocessed
    preprocessed: bool,
    /// Maximum node ID seen
    max_id: usize,
}

impl CoordinateTree {
    /// Create a new coordinate tree with capacity hint.
    pub fn new(capacity: usize) -> Self {
        // Compute max_log = ceil(log2(capacity + 1))
        let max_log = ((capacity + 1) as f64).log2().ceil() as usize;

        Self {
            nodes: HashMap::with_capacity(capacity),
            root: None,
            max_log: max_log.max(1),
            ancestor: HashMap::with_capacity(capacity),
            preprocessed: false,
            max_id: 0,
        }
    }

    /// Add a node to the tree.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique node identifier
    /// * `parent` - Parent node ID (None for root)
    /// * `coordinate` - Trajectory coordinate for this node
    ///
    /// # Returns
    ///
    /// True if node was added, false if ID already exists.
    pub fn add_node(
        &mut self,
        id: usize,
        parent: Option<usize>,
        coordinate: TrajectoryCoordinate5D,
    ) -> bool {
        if self.nodes.contains_key(&id) {
            return false;
        }

        let node = TreeNode::new(id, parent, coordinate);
        self.nodes.insert(id, node);
        self.max_id = self.max_id.max(id);

        // Update parent's children list
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.children.push(id);
            }
        } else {
            // This is a root
            if self.root.is_none() {
                self.root = Some(id);
            }
        }

        // Mark as needing preprocessing
        self.preprocessed = false;

        true
    }

    /// Get a reference to a node.
    pub fn get_node(&self, id: usize) -> Option<&TreeNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node.
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut TreeNode> {
        self.nodes.get_mut(&id)
    }

    /// Get the root node ID.
    pub fn root(&self) -> Option<usize> {
        self.root
    }

    /// Get number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Preprocess the tree for LCA queries.
    ///
    /// Must be called after all nodes are added and before LCA queries.
    /// Complexity: O(n log n)
    pub fn preprocess(&mut self) {
        if self.preprocessed || self.root.is_none() {
            return;
        }

        // Update max_log based on actual size
        self.max_log = ((self.nodes.len() + 1) as f64).log2().ceil() as usize;
        self.max_log = self.max_log.max(1);

        // Compute depths using DFS
        self.compute_depths();

        // Build binary lifting table
        self.build_ancestor_table();

        self.preprocessed = true;
    }

    /// Compute depths using DFS from root.
    fn compute_depths(&mut self) {
        if let Some(root_id) = self.root {
            let mut stack = vec![(root_id, 0usize)];

            while let Some((node_id, depth)) = stack.pop() {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.depth = depth;

                    for &child_id in node.children.clone().iter() {
                        stack.push((child_id, depth + 1));
                    }
                }
            }
        }
    }

    /// Build the binary lifting ancestor table.
    fn build_ancestor_table(&mut self) {
        self.ancestor.clear();

        // Initialize ancestor[node][0] = parent
        for (&id, node) in &self.nodes {
            let mut ancestors = vec![None; self.max_log];
            ancestors[0] = node.parent;
            self.ancestor.insert(id, ancestors);
        }

        // Fill in ancestors[k] = ancestors[ancestors[k-1]][k-1]
        for k in 1..self.max_log {
            let node_ids: Vec<usize> = self.nodes.keys().copied().collect();

            for id in node_ids {
                let prev_ancestor = self.ancestor.get(&id).and_then(|a| a[k - 1]);

                if let Some(prev_id) = prev_ancestor {
                    let kth_ancestor = self.ancestor.get(&prev_id).and_then(|a| a[k - 1]);

                    if let Some(anc) = self.ancestor.get_mut(&id) {
                        anc[k] = kth_ancestor;
                    }
                }
            }
        }
    }

    /// Get the depth of a node.
    pub fn depth(&self, id: usize) -> Option<usize> {
        self.nodes.get(&id).map(|n| n.depth)
    }

    /// Lift a node up by k levels.
    ///
    /// Returns None if we go above the root.
    pub fn lift(&self, mut node: usize, k: usize) -> Option<usize> {
        if !self.preprocessed {
            return None;
        }

        for bit in 0..self.max_log {
            if k & (1 << bit) != 0 {
                node = self.ancestor.get(&node)?.get(bit)?.as_ref().copied()?;
            }
        }

        Some(node)
    }

    /// Get the k-th ancestor of a node (0 = self, 1 = parent, etc.).
    pub fn kth_ancestor(&self, node: usize, k: usize) -> Option<usize> {
        if k == 0 {
            if self.nodes.contains_key(&node) {
                Some(node)
            } else {
                None
            }
        } else {
            self.lift(node, k)
        }
    }

    /// Compute the Lowest Common Ancestor of two nodes.
    ///
    /// Complexity: O(log n)
    ///
    /// # Returns
    ///
    /// The LCA node ID, or None if preprocessing not done or nodes don't exist.
    pub fn lca(&self, a: usize, b: usize) -> Option<usize> {
        if !self.preprocessed {
            return None;
        }

        let depth_a = self.depth(a)?;
        let depth_b = self.depth(b)?;

        // Make a the deeper node
        let (mut a, mut b, depth_a, depth_b) = if depth_a >= depth_b {
            (a, b, depth_a, depth_b)
        } else {
            (b, a, depth_b, depth_a)
        };

        // Lift a to the same depth as b
        let diff = depth_a - depth_b;
        a = self.lift(a, diff)?;

        if a == b {
            return Some(a);
        }

        // Binary search for LCA
        for k in (0..self.max_log).rev() {
            let anc_a = self.ancestor.get(&a).and_then(|v| v.get(k)).and_then(|x| *x);
            let anc_b = self.ancestor.get(&b).and_then(|v| v.get(k)).and_then(|x| *x);

            if anc_a != anc_b {
                if let (Some(new_a), Some(new_b)) = (anc_a, anc_b) {
                    a = new_a;
                    b = new_b;
                }
            }
        }

        // LCA is the parent of where we stopped
        self.ancestor.get(&a).and_then(|v| v[0])
    }

    /// Compute the path distance between two nodes.
    ///
    /// Path distance = depth(a) + depth(b) - 2 * depth(lca)
    ///
    /// # Returns
    ///
    /// Number of edges between a and b, or None if LCA cannot be computed.
    pub fn path_distance(&self, a: usize, b: usize) -> Option<usize> {
        let depth_a = self.depth(a)?;
        let depth_b = self.depth(b)?;
        let lca = self.lca(a, b)?;
        let depth_lca = self.depth(lca)?;

        Some(depth_a + depth_b - 2 * depth_lca)
    }

    /// Check if node `a` is an ancestor of node `b`.
    pub fn is_ancestor(&self, a: usize, b: usize) -> bool {
        self.lca(a, b) == Some(a)
    }

    /// Check if node `a` is a descendant of node `b`.
    pub fn is_descendant(&self, a: usize, b: usize) -> bool {
        self.is_ancestor(b, a)
    }

    /// Get the path from node a to node b.
    ///
    /// Returns the sequence of node IDs from a to b (inclusive).
    pub fn path(&self, a: usize, b: usize) -> Option<Vec<usize>> {
        let lca = self.lca(a, b)?;

        // Path from a to lca
        let mut path_a_to_lca = Vec::new();
        let mut current = a;
        while current != lca {
            path_a_to_lca.push(current);
            current = self.nodes.get(&current)?.parent?;
        }

        // Path from lca to b (reversed)
        let mut path_lca_to_b = Vec::new();
        current = b;
        while current != lca {
            path_lca_to_b.push(current);
            current = self.nodes.get(&current)?.parent?;
        }
        path_lca_to_b.reverse();

        // Combine: a -> ... -> lca -> ... -> b
        path_a_to_lca.push(lca);
        path_a_to_lca.extend(path_lca_to_b);

        Some(path_a_to_lca)
    }

    /// Extract a subtree rooted at the given node.
    ///
    /// Returns a new CoordinateTree containing only the subtree.
    pub fn subtree(&self, root: usize) -> Option<CoordinateTree> {
        if !self.nodes.contains_key(&root) {
            return None;
        }

        let mut subtree = CoordinateTree::new(self.nodes.len());

        // BFS to extract subtree
        let mut queue = vec![root];
        while let Some(node_id) = queue.pop() {
            if let Some(node) = self.nodes.get(&node_id) {
                // In subtree, root has no parent
                let parent = if node_id == root {
                    None
                } else {
                    node.parent
                };

                subtree.add_node(node_id, parent, node.coordinate);

                for &child_id in &node.children {
                    queue.push(child_id);
                }
            }
        }

        subtree.preprocess();
        Some(subtree)
    }

    /// Get all nodes at a specific depth.
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.depth == depth)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get the maximum depth in the tree.
    pub fn max_depth(&self) -> usize {
        self.nodes.values().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Iterate over all nodes.
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &TreeNode)> {
        self.nodes.iter()
    }

    /// Get all leaf nodes.
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_leaf())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get all node IDs in preorder traversal.
    pub fn preorder(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.nodes.len());

        if let Some(root) = self.root {
            let mut stack = vec![root];

            while let Some(node_id) = stack.pop() {
                result.push(node_id);

                if let Some(node) = self.nodes.get(&node_id) {
                    // Push children in reverse order for correct preorder
                    for &child in node.children.iter().rev() {
                        stack.push(child);
                    }
                }
            }
        }

        result
    }

    /// Compute the weighted center of the tree using coordinates.
    pub fn weighted_center(&self) -> Option<TrajectoryCoordinate5D> {
        if self.nodes.is_empty() {
            return None;
        }

        let n = self.nodes.len() as f32;
        let mut sum_depth = 0.0f32;
        let mut sum_sibling = 0.0f32;
        let mut sum_homo = 0.0f32;
        let mut sum_temporal = 0.0f32;
        let mut sum_complexity = 0.0f32;

        for node in self.nodes.values() {
            sum_depth += node.coordinate.depth as f32;
            sum_sibling += node.coordinate.sibling_order as f32;
            sum_homo += node.coordinate.homogeneity;
            sum_temporal += node.coordinate.temporal;
            sum_complexity += node.coordinate.complexity as f32;
        }

        Some(TrajectoryCoordinate5D::new(
            (sum_depth / n).round() as u32,
            (sum_sibling / n).round() as u32,
            sum_homo / n,
            sum_temporal / n,
            (sum_complexity / n).round().max(1.0) as u32,
        ))
    }
}

impl Default for CoordinateTree {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Build a coordinate tree from parent-child relationships.
///
/// # Arguments
///
/// * `nodes` - List of (id, parent_id, coordinate) tuples
///
/// # Returns
///
/// A preprocessed CoordinateTree.
pub fn build_coordinate_tree(
    nodes: impl IntoIterator<Item = (usize, Option<usize>, TrajectoryCoordinate5D)>,
) -> CoordinateTree {
    let nodes: Vec<_> = nodes.into_iter().collect();
    let mut tree = CoordinateTree::new(nodes.len());

    for (id, parent, coord) in nodes {
        tree.add_node(id, parent, coord);
    }

    tree.preprocess();
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coord(depth: u32, temporal: f32) -> TrajectoryCoordinate5D {
        TrajectoryCoordinate5D::new(depth, 0, 0.8, temporal, 1)
    }

    fn build_sample_tree() -> CoordinateTree {
        //       0 (root)
        //      / \
        //     1   2
        //    / \
        //   3   4
        //       |
        //       5

        let mut tree = CoordinateTree::new(6);
        tree.add_node(0, None, make_coord(0, 0.0));
        tree.add_node(1, Some(0), make_coord(1, 0.2));
        tree.add_node(2, Some(0), make_coord(1, 0.3));
        tree.add_node(3, Some(1), make_coord(2, 0.4));
        tree.add_node(4, Some(1), make_coord(2, 0.5));
        tree.add_node(5, Some(4), make_coord(3, 0.6));
        tree.preprocess();
        tree
    }

    #[test]
    fn test_tree_creation() {
        let tree = build_sample_tree();
        assert_eq!(tree.len(), 6);
        assert_eq!(tree.root(), Some(0));
    }

    #[test]
    fn test_node_properties() {
        let tree = build_sample_tree();

        let root = tree.get_node(0).unwrap();
        assert!(root.is_root());
        assert!(!root.is_leaf());
        assert_eq!(root.children.len(), 2);

        let leaf = tree.get_node(3).unwrap();
        assert!(!leaf.is_root());
        assert!(leaf.is_leaf());
    }

    #[test]
    fn test_depths() {
        let tree = build_sample_tree();

        assert_eq!(tree.depth(0), Some(0));
        assert_eq!(tree.depth(1), Some(1));
        assert_eq!(tree.depth(2), Some(1));
        assert_eq!(tree.depth(3), Some(2));
        assert_eq!(tree.depth(4), Some(2));
        assert_eq!(tree.depth(5), Some(3));
    }

    #[test]
    fn test_lca_same_node() {
        let tree = build_sample_tree();
        assert_eq!(tree.lca(3, 3), Some(3));
    }

    #[test]
    fn test_lca_parent_child() {
        let tree = build_sample_tree();
        assert_eq!(tree.lca(1, 3), Some(1)); // Parent-child
        assert_eq!(tree.lca(0, 5), Some(0)); // Root and leaf
    }

    #[test]
    fn test_lca_siblings() {
        let tree = build_sample_tree();
        assert_eq!(tree.lca(3, 4), Some(1)); // Siblings
        assert_eq!(tree.lca(1, 2), Some(0)); // Siblings under root
    }

    #[test]
    fn test_lca_cousins() {
        let tree = build_sample_tree();
        assert_eq!(tree.lca(3, 2), Some(0)); // Different branches
        assert_eq!(tree.lca(5, 2), Some(0)); // Deep leaf and sibling branch
    }

    #[test]
    fn test_path_distance() {
        let tree = build_sample_tree();

        assert_eq!(tree.path_distance(0, 0), Some(0));
        assert_eq!(tree.path_distance(0, 1), Some(1));
        assert_eq!(tree.path_distance(1, 3), Some(1));
        assert_eq!(tree.path_distance(3, 4), Some(2)); // Siblings
        assert_eq!(tree.path_distance(3, 2), Some(3)); // Different branches
        assert_eq!(tree.path_distance(5, 2), Some(4)); // Deep leaf to cousin
    }

    #[test]
    fn test_kth_ancestor() {
        let tree = build_sample_tree();

        assert_eq!(tree.kth_ancestor(5, 0), Some(5));
        assert_eq!(tree.kth_ancestor(5, 1), Some(4));
        assert_eq!(tree.kth_ancestor(5, 2), Some(1));
        assert_eq!(tree.kth_ancestor(5, 3), Some(0));
        assert_eq!(tree.kth_ancestor(5, 4), None); // Above root
    }

    #[test]
    fn test_is_ancestor() {
        let tree = build_sample_tree();

        assert!(tree.is_ancestor(0, 5)); // Root is ancestor of all
        assert!(tree.is_ancestor(1, 5));
        assert!(tree.is_ancestor(4, 5));
        assert!(tree.is_ancestor(5, 5)); // Self
        assert!(!tree.is_ancestor(2, 5)); // Different branch
        assert!(!tree.is_ancestor(5, 0)); // Reversed
    }

    #[test]
    fn test_path() {
        let tree = build_sample_tree();

        let path = tree.path(5, 2).unwrap();
        assert_eq!(path, vec![5, 4, 1, 0, 2]);

        let path = tree.path(3, 4).unwrap();
        assert_eq!(path, vec![3, 1, 4]);

        let path = tree.path(0, 5).unwrap();
        assert_eq!(path, vec![0, 1, 4, 5]);
    }

    #[test]
    fn test_subtree() {
        let tree = build_sample_tree();

        let subtree = tree.subtree(1).unwrap();
        assert_eq!(subtree.len(), 4); // Nodes 1, 3, 4, 5
        assert_eq!(subtree.root(), Some(1));
        assert_eq!(subtree.depth(1), Some(0)); // Root of subtree
        assert_eq!(subtree.depth(5), Some(2));
    }

    #[test]
    fn test_nodes_at_depth() {
        let tree = build_sample_tree();

        let depth_0 = tree.nodes_at_depth(0);
        assert_eq!(depth_0.len(), 1);
        assert!(depth_0.contains(&0));

        let depth_1 = tree.nodes_at_depth(1);
        assert_eq!(depth_1.len(), 2);
        assert!(depth_1.contains(&1));
        assert!(depth_1.contains(&2));

        let depth_2 = tree.nodes_at_depth(2);
        assert_eq!(depth_2.len(), 2);
    }

    #[test]
    fn test_max_depth() {
        let tree = build_sample_tree();
        assert_eq!(tree.max_depth(), 3);
    }

    #[test]
    fn test_leaves() {
        let tree = build_sample_tree();
        let leaves = tree.leaves();

        assert_eq!(leaves.len(), 3);
        assert!(leaves.contains(&2));
        assert!(leaves.contains(&3));
        assert!(leaves.contains(&5));
    }

    #[test]
    fn test_preorder() {
        let tree = build_sample_tree();
        let preorder = tree.preorder();

        assert_eq!(preorder.len(), 6);
        assert_eq!(preorder[0], 0); // Root first

        // Parent should come before children
        let pos_0 = preorder.iter().position(|&x| x == 0).unwrap();
        let pos_1 = preorder.iter().position(|&x| x == 1).unwrap();
        let pos_5 = preorder.iter().position(|&x| x == 5).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_1 < pos_5);
    }

    #[test]
    fn test_weighted_center() {
        let tree = build_sample_tree();
        let center = tree.weighted_center().unwrap();

        // Average depth should be (0+1+1+2+2+3)/6 = 1.5 -> 2
        assert!(center.depth >= 1 && center.depth <= 2);
    }

    #[test]
    fn test_build_coordinate_tree() {
        let nodes = vec![
            (0, None, make_coord(0, 0.0)),
            (1, Some(0), make_coord(1, 0.2)),
            (2, Some(0), make_coord(1, 0.4)),
            (3, Some(1), make_coord(2, 0.6)),
        ];

        let tree = build_coordinate_tree(nodes);

        assert_eq!(tree.len(), 4);
        assert_eq!(tree.root(), Some(0));
        assert_eq!(tree.lca(2, 3), Some(0));
    }

    #[test]
    fn test_empty_tree() {
        let tree = CoordinateTree::new(10);
        assert!(tree.is_empty());
        assert_eq!(tree.root(), None);
        assert_eq!(tree.max_depth(), 0);
    }

    #[test]
    fn test_single_node_tree() {
        let mut tree = CoordinateTree::new(1);
        tree.add_node(0, None, make_coord(0, 0.0));
        tree.preprocess();

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.lca(0, 0), Some(0));
        assert_eq!(tree.path_distance(0, 0), Some(0));
    }

    #[test]
    fn test_linear_tree() {
        // 0 -> 1 -> 2 -> 3 -> 4
        let nodes: Vec<_> = (0..5)
            .map(|i| {
                let parent = if i == 0 { None } else { Some(i - 1) };
                (i, parent, make_coord(i as u32, i as f32 / 5.0))
            })
            .collect();

        let tree = build_coordinate_tree(nodes);

        assert_eq!(tree.max_depth(), 4);
        assert_eq!(tree.lca(0, 4), Some(0));
        assert_eq!(tree.lca(2, 4), Some(2));
        assert_eq!(tree.path_distance(0, 4), Some(4));
    }
}
