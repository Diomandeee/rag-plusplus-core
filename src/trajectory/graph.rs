//! DAG Traversal Algorithms for Trajectory Structure
//!
//! Trajectories are DAGs (Directed Acyclic Graphs) where:
//! - Each node is an episode (unit of experience)
//! - Edges represent parent→child relationships
//! - Multiple children = regenerations or branching
//!
//! This module provides high-performance algorithms for:
//! - Path finding (root to leaf traversal)
//! - Branch detection (identifying decision points)
//! - Primary path selection (choosing the "best" linear path through the DAG)

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a node in the trajectory DAG.
pub type NodeId = u64;

/// Edge in the trajectory DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub parent: NodeId,
    pub child: NodeId,
    pub edge_type: EdgeType,
}

/// Type of edge in the DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EdgeType {
    /// Normal continuation (single child)
    #[default]
    Continuation,
    /// Regeneration (multiple children from same parent)
    Regeneration,
    /// Branch (explicit user-initiated fork)
    Branch,
}

/// An episode in the trajectory DAG.
///
/// An episode represents a unit of experience - a message turn, interaction,
/// or decision point within a trajectory.
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    /// Metadata for primary path selection
    pub weight: f32,
    pub has_thumbs_up: bool,
    pub has_thumbs_down: bool,
    pub content_length: usize,
    pub has_error: bool,
    pub created_at: i64,
}

impl Episode {
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            parent: None,
            children: Vec::new(),
            weight: 1.0,
            has_thumbs_up: false,
            has_thumbs_down: false,
            content_length: 0,
            has_error: false,
            created_at: 0,
        }
    }

    /// Check if this episode is a branching point (multiple children).
    #[inline]
    pub fn is_branch_point(&self) -> bool {
        self.children.len() > 1
    }

    /// Check if this episode is a leaf (no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Check if this episode is a root (no parent).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }
}

/// Information about a branch point in the DAG.
#[derive(Debug, Clone)]
pub struct BranchInfo {
    /// The episode where branching occurs
    pub branch_point: NodeId,
    /// All children at this branch
    pub children: Vec<NodeId>,
    /// Type of branching
    pub branch_type: EdgeType,
    /// Index of selected child for primary path
    pub selected_child_idx: Option<usize>,
}

/// Policy for selecting which child to follow at branch points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathSelectionPolicy {
    /// Prefer child with thumbs_up, then longest content, then first by time
    #[default]
    FeedbackFirst,
    /// Always pick first child by creation time
    FirstByTime,
    /// Always pick child with longest content
    LongestContent,
    /// Pick child with highest weight
    HighestWeight,
}

/// Traversal order for DAG walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraversalOrder {
    /// Depth-first, root to leaves
    #[default]
    DepthFirst,
    /// Breadth-first, level by level
    BreadthFirst,
    /// Topological order (respects dependencies)
    Topological,
    /// Reverse topological (leaves to roots)
    ReverseTopological,
}

/// Result of finding a path through the DAG.
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Ordered list of node IDs from root to leaf
    pub nodes: Vec<NodeId>,
    /// Branch points encountered
    pub branch_points: Vec<BranchInfo>,
    /// Total weight along path
    pub total_weight: f32,
}

/// Trajectory DAG structure optimized for traversal operations.
///
/// A trajectory represents a sequence of experiences (episodes) forming
/// a path through time. The DAG structure captures branching and
/// regeneration points.
#[derive(Debug, Clone)]
pub struct TrajectoryGraph {
    nodes: HashMap<NodeId, Episode>,
    roots: Vec<NodeId>,
    leaves: Vec<NodeId>,
}

impl TrajectoryGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            leaves: Vec::new(),
        }
    }

    /// Build graph from a list of edges.
    ///
    /// # Arguments
    ///
    /// * `edges` - Iterator of (parent_id, child_id) pairs
    ///
    /// # Example
    ///
    /// ```
    /// use rag_plusplus_core::trajectory::{TrajectoryGraph, Edge, EdgeType};
    ///
    /// let edges = vec![
    ///     Edge { parent: 1, child: 2, edge_type: EdgeType::Continuation },
    ///     Edge { parent: 2, child: 3, edge_type: EdgeType::Continuation },
    ///     Edge { parent: 2, child: 4, edge_type: EdgeType::Regeneration }, // Branch
    /// ];
    ///
    /// let graph = TrajectoryGraph::from_edges(edges.iter().copied());
    /// assert_eq!(graph.node_count(), 4);
    /// assert!(graph.is_branch_point(2));
    /// ```
    pub fn from_edges(edges: impl IntoIterator<Item = Edge>) -> Self {
        let mut graph = Self::new();

        for edge in edges {
            // Ensure nodes exist
            graph.nodes.entry(edge.parent).or_insert_with(|| Episode::new(edge.parent));
            graph.nodes.entry(edge.child).or_insert_with(|| Episode::new(edge.child));

            // Add edge
            if let Some(parent) = graph.nodes.get_mut(&edge.parent) {
                if !parent.children.contains(&edge.child) {
                    parent.children.push(edge.child);
                }
            }
            if let Some(child) = graph.nodes.get_mut(&edge.child) {
                child.parent = Some(edge.parent);
            }
        }

        graph.update_roots_and_leaves();
        graph
    }

    /// Add a single episode.
    pub fn add_node(&mut self, node: Episode) {
        self.nodes.insert(node.id, node);
    }

    /// Get an episode by ID.
    #[inline]
    pub fn get_node(&self, id: NodeId) -> Option<&Episode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to an episode.
    #[inline]
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Episode> {
        self.nodes.get_mut(&id)
    }

    /// Number of episodes in the graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get all root episodes (episodes without parents).
    #[inline]
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    /// Get all leaf episodes (episodes without children).
    #[inline]
    pub fn leaves(&self) -> &[NodeId] {
        &self.leaves
    }

    /// Check if an episode is a branch point.
    #[inline]
    pub fn is_branch_point(&self, id: NodeId) -> bool {
        self.nodes.get(&id).map_or(false, |n| n.is_branch_point())
    }

    /// Find all branch points in the graph.
    pub fn find_branch_points(&self) -> Vec<BranchInfo> {
        self.nodes
            .values()
            .filter(|n| n.is_branch_point())
            .map(|n| BranchInfo {
                branch_point: n.id,
                children: n.children.clone(),
                branch_type: if n.children.len() > 1 {
                    EdgeType::Regeneration
                } else {
                    EdgeType::Continuation
                },
                selected_child_idx: None,
            })
            .collect()
    }

    /// Update roots and leaves lists (call after modifications).
    fn update_roots_and_leaves(&mut self) {
        self.roots = self.nodes.values()
            .filter(|n| n.is_root())
            .map(|n| n.id)
            .collect();

        self.leaves = self.nodes.values()
            .filter(|n| n.is_leaf())
            .map(|n| n.id)
            .collect();
    }

    // =========================================================================
    // TRAVERSAL ALGORITHMS
    // =========================================================================

    /// Traverse the graph in specified order, calling visitor for each episode.
    ///
    /// # Arguments
    ///
    /// * `order` - Traversal order
    /// * `visitor` - Callback function for each episode
    pub fn traverse<F>(&self, order: TraversalOrder, mut visitor: F)
    where
        F: FnMut(&Episode),
    {
        match order {
            TraversalOrder::DepthFirst => self.traverse_dfs(&mut visitor),
            TraversalOrder::BreadthFirst => self.traverse_bfs(&mut visitor),
            TraversalOrder::Topological => self.traverse_topological(&mut visitor),
            TraversalOrder::ReverseTopological => self.traverse_reverse_topological(&mut visitor),
        }
    }

    fn traverse_dfs<F>(&self, visitor: &mut F)
    where
        F: FnMut(&Episode),
    {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = self.roots.clone();

        while let Some(id) = stack.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);

            if let Some(node) = self.nodes.get(&id) {
                visitor(node);
                // Push children in reverse order for correct DFS order
                for &child_id in node.children.iter().rev() {
                    if !visited.contains(&child_id) {
                        stack.push(child_id);
                    }
                }
            }
        }
    }

    fn traverse_bfs<F>(&self, visitor: &mut F)
    where
        F: FnMut(&Episode),
    {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<NodeId> = self.roots.iter().copied().collect();

        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);

            if let Some(node) = self.nodes.get(&id) {
                visitor(node);
                for &child_id in &node.children {
                    if !visited.contains(&child_id) {
                        queue.push_back(child_id);
                    }
                }
            }
        }
    }

    fn traverse_topological<F>(&self, visitor: &mut F)
    where
        F: FnMut(&Episode),
    {
        // Kahn's algorithm
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in self.nodes.values() {
            in_degree.entry(node.id).or_insert(0);
            for &child in &node.children {
                *in_degree.entry(child).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(id) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&id) {
                visitor(node);
                for &child in &node.children {
                    if let Some(deg) = in_degree.get_mut(&child) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(child);
                        }
                    }
                }
            }
        }
    }

    fn traverse_reverse_topological<F>(&self, visitor: &mut F)
    where
        F: FnMut(&Episode),
    {
        let mut order = Vec::with_capacity(self.nodes.len());
        self.traverse_topological(&mut |node| order.push(node.id));

        for id in order.into_iter().rev() {
            if let Some(node) = self.nodes.get(&id) {
                visitor(node);
            }
        }
    }

    // =========================================================================
    // PRIMARY PATH SELECTION
    // =========================================================================

    /// Find the primary path through the DAG using the specified selection policy.
    ///
    /// The primary path is a linear sequence from a root to a leaf that represents
    /// the "best" version of the trajectory (e.g., after regenerations).
    ///
    /// # Arguments
    ///
    /// * `policy` - How to choose at branch points
    ///
    /// # Returns
    ///
    /// PathResult containing the selected path and branch information.
    pub fn find_primary_path(&self, policy: PathSelectionPolicy) -> Option<PathResult> {
        if self.roots.is_empty() {
            return None;
        }

        // Start from first root
        let start = self.roots[0];
        let mut path = Vec::new();
        let mut branch_points = Vec::new();
        let mut total_weight = 0.0;
        let mut current = start;

        loop {
            let node = self.nodes.get(&current)?;
            path.push(current);
            total_weight += node.weight;

            if node.children.is_empty() {
                break;
            }

            // Select next node based on policy
            let (next_idx, next) = self.select_child(node, policy)?;

            if node.is_branch_point() {
                branch_points.push(BranchInfo {
                    branch_point: current,
                    children: node.children.clone(),
                    branch_type: EdgeType::Regeneration,
                    selected_child_idx: Some(next_idx),
                });
            }

            current = next;
        }

        Some(PathResult {
            nodes: path,
            branch_points,
            total_weight,
        })
    }

    /// Select which child to follow at a branch point.
    fn select_child(&self, parent: &Episode, policy: PathSelectionPolicy) -> Option<(usize, NodeId)> {
        if parent.children.is_empty() {
            return None;
        }

        let children: Vec<&Episode> = parent.children
            .iter()
            .filter_map(|&id| self.nodes.get(&id))
            .collect();

        if children.is_empty() {
            return Some((0, parent.children[0]));
        }

        let selected_idx = match policy {
            PathSelectionPolicy::FeedbackFirst => {
                // Priority: thumbs_up > no_thumbs_down > longest > first_by_time
                children.iter().enumerate()
                    .max_by(|(_, a), (_, b)| {
                        // First: thumbs_up wins
                        match (a.has_thumbs_up, b.has_thumbs_up) {
                            (true, false) => return std::cmp::Ordering::Greater,
                            (false, true) => return std::cmp::Ordering::Less,
                            _ => {}
                        }
                        // Second: no thumbs_down is better
                        match (a.has_thumbs_down, b.has_thumbs_down) {
                            (false, true) => return std::cmp::Ordering::Greater,
                            (true, false) => return std::cmp::Ordering::Less,
                            _ => {}
                        }
                        // Third: longer content
                        match a.content_length.cmp(&b.content_length) {
                            std::cmp::Ordering::Equal => {}
                            other => return other,
                        }
                        // Fourth: earlier creation time
                        a.created_at.cmp(&b.created_at).reverse()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            PathSelectionPolicy::FirstByTime => {
                children.iter().enumerate()
                    .min_by_key(|(_, n)| n.created_at)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            PathSelectionPolicy::LongestContent => {
                children.iter().enumerate()
                    .max_by_key(|(_, n)| n.content_length)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            PathSelectionPolicy::HighestWeight => {
                children.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        };

        Some((selected_idx, parent.children[selected_idx]))
    }

    // =========================================================================
    // PATH FINDING
    // =========================================================================

    /// Find all paths from an episode to all reachable leaves.
    pub fn find_all_paths_from(&self, start: NodeId) -> Vec<Vec<NodeId>> {
        let mut paths = Vec::new();
        let mut current_path = vec![start];
        self.find_paths_recursive(start, &mut current_path, &mut paths);
        paths
    }

    fn find_paths_recursive(
        &self,
        current: NodeId,
        path: &mut Vec<NodeId>,
        paths: &mut Vec<Vec<NodeId>>,
    ) {
        if let Some(node) = self.nodes.get(&current) {
            if node.is_leaf() {
                paths.push(path.clone());
            } else {
                for &child in &node.children {
                    path.push(child);
                    self.find_paths_recursive(child, path, paths);
                    path.pop();
                }
            }
        }
    }

    /// Find the path from root to a specific episode.
    pub fn find_path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        let mut path = Vec::new();
        let mut current = target;

        loop {
            path.push(current);
            match self.nodes.get(&current)?.parent {
                Some(parent) => current = parent,
                None => break,
            }
        }

        path.reverse();
        Some(path)
    }

    /// Compute the depth of an episode (distance from root).
    pub fn depth(&self, node: NodeId) -> Option<usize> {
        self.find_path_to(node).map(|p| p.len() - 1)
    }

    /// Find the lowest common ancestor of two episodes.
    pub fn lowest_common_ancestor(&self, a: NodeId, b: NodeId) -> Option<NodeId> {
        let path_a = self.find_path_to(a)?;
        let path_b = self.find_path_to(b)?;

        let path_a_set: HashSet<_> = path_a.iter().copied().collect();

        // Walk up from b until we find a node in a's path
        for &node in path_b.iter().rev() {
            if path_a_set.contains(&node) {
                return Some(node);
            }
        }

        None
    }
}

impl Default for TrajectoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_graph() -> TrajectoryGraph {
        // 1 -> 2 -> 3 -> 4
        let edges = vec![
            Edge { parent: 1, child: 2, edge_type: EdgeType::Continuation },
            Edge { parent: 2, child: 3, edge_type: EdgeType::Continuation },
            Edge { parent: 3, child: 4, edge_type: EdgeType::Continuation },
        ];
        TrajectoryGraph::from_edges(edges.into_iter())
    }

    fn make_branching_graph() -> TrajectoryGraph {
        // 1 -> 2 -> 3
        //        -> 4 (regeneration)
        //   -> 5 (separate branch)
        let edges = vec![
            Edge { parent: 1, child: 2, edge_type: EdgeType::Continuation },
            Edge { parent: 2, child: 3, edge_type: EdgeType::Regeneration },
            Edge { parent: 2, child: 4, edge_type: EdgeType::Regeneration },
            Edge { parent: 1, child: 5, edge_type: EdgeType::Branch },
        ];
        TrajectoryGraph::from_edges(edges.into_iter())
    }

    #[test]
    fn test_linear_graph() {
        let graph = make_linear_graph();
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.roots(), &[1]);
        assert_eq!(graph.leaves(), &[4]);
        assert!(!graph.is_branch_point(1));
    }

    #[test]
    fn test_branching_graph() {
        let graph = make_branching_graph();
        assert_eq!(graph.node_count(), 5);
        assert!(graph.is_branch_point(1));
        assert!(graph.is_branch_point(2));

        let branches = graph.find_branch_points();
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn test_find_path_to() {
        let graph = make_linear_graph();
        let path = graph.find_path_to(4).unwrap();
        assert_eq!(path, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_primary_path() {
        let graph = make_linear_graph();
        let result = graph.find_primary_path(PathSelectionPolicy::FirstByTime).unwrap();
        assert_eq!(result.nodes, vec![1, 2, 3, 4]);
        assert!(result.branch_points.is_empty());
    }

    #[test]
    fn test_dfs_traversal() {
        let graph = make_linear_graph();
        let mut visited = Vec::new();
        graph.traverse(TraversalOrder::DepthFirst, |node| {
            visited.push(node.id);
        });
        assert_eq!(visited, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_depth() {
        let graph = make_linear_graph();
        assert_eq!(graph.depth(1), Some(0));
        assert_eq!(graph.depth(4), Some(3));
    }
}
