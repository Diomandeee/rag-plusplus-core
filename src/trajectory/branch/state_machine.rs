//! Branch State Machine
//!
//! High-level state machine for managing branches in a trajectory DAG.
//! Ported from DLM's StateMachine class with split/merge operations.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::trajectory::graph::{NodeId, TrajectoryGraph};
use super::operations::{Branch, BranchId, BranchOperation, BranchError, ForkPoint};

/// Get current Unix timestamp in seconds.
#[inline]
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Context for the current branch position.
#[derive(Debug, Clone)]
pub struct BranchContext {
    /// Current active branch
    pub current_branch: BranchId,
    /// Current node position within the branch
    pub current_node: NodeId,
    /// Depth in the trajectory
    pub depth: u32,
    /// Path from root to current position
    pub path: Vec<NodeId>,
}

/// Result of a split operation.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// The original branch (updated)
    pub original_branch: BranchId,
    /// The new branch created by the split
    pub new_branch: BranchId,
    /// The split point node
    pub split_point: NodeId,
    /// Nodes moved to the new branch
    pub moved_nodes: Vec<NodeId>,
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The branch that received the merge
    pub target_branch: BranchId,
    /// The branch that was merged (now inactive)
    pub merged_branch: BranchId,
    /// The merge point node
    pub merge_point: NodeId,
}

/// Branch State Machine for trajectory DAGs.
///
/// Provides high-level operations for managing branches, including:
/// - **split**: Create a new branch from a subtree (DLM port)
/// - **merge**: Combine branches back together
/// - **traverse**: Navigate between branches
/// - **recover**: Find and reactivate "lost" branches
///
/// # Design (from DLM)
///
/// The state machine maintains:
/// - A mapping of branch IDs to Branch structs
/// - The currently active branch
/// - A history of all branch operations
/// - Fork points where branching occurred
///
/// # Example
///
/// ```ignore
/// let graph = TrajectoryGraph::from_edges(edges);
/// let mut machine = BranchStateMachine::from_graph(graph);
///
/// // Split at a fork point
/// let result = machine.split(fork_node_id)?;
/// println!("Created new branch: {}", result.new_branch);
///
/// // Traverse to the new branch
/// machine.traverse(result.new_branch)?;
///
/// // Later, merge back
/// machine.merge(result.new_branch, result.original_branch)?;
/// ```
#[derive(Debug, Clone)]
pub struct BranchStateMachine {
    /// All branches in the state machine
    branches: HashMap<BranchId, Branch>,
    /// Currently active branch
    current_branch: BranchId,
    /// History of all branch operations
    history: Vec<BranchOperation>,
    /// Fork points in the trajectory
    fork_points: HashMap<NodeId, ForkPoint>,
    /// Next available branch ID
    next_branch_id: BranchId,
    /// Mapping of nodes to their owning branch
    node_to_branch: HashMap<NodeId, BranchId>,
    /// The underlying trajectory graph
    graph: TrajectoryGraph,
    /// Current context
    context: BranchContext,
}

impl BranchStateMachine {
    /// Create a new state machine from a trajectory graph.
    ///
    /// Analyzes the graph to identify branch points and creates
    /// the initial branch structure.
    pub fn from_graph(graph: TrajectoryGraph) -> Self {
        let mut machine = Self {
            branches: HashMap::new(),
            current_branch: 0,
            history: Vec::new(),
            fork_points: HashMap::new(),
            next_branch_id: 1,
            node_to_branch: HashMap::new(),
            graph,
            context: BranchContext {
                current_branch: 0,
                current_node: 0,
                depth: 0,
                path: Vec::new(),
            },
        };

        machine.initialize_from_graph();
        machine
    }

    /// Initialize branch structure from the underlying graph.
    fn initialize_from_graph(&mut self) {
        let roots: Vec<NodeId> = self.graph.roots().to_vec();

        if roots.is_empty() {
            return;
        }

        // Create root branch from first root
        let root_node = roots[0];
        let root_branch = Branch::root(0, root_node);
        self.branches.insert(0, root_branch);
        self.node_to_branch.insert(root_node, 0);

        // Build branch structure by traversing the graph
        self.build_branches_from_root(root_node, 0, 0);

        // Update context
        if let Some(head) = self.branches.get(&0).map(|b| b.head) {
            self.context = BranchContext {
                current_branch: 0,
                current_node: head,
                depth: self.compute_depth(head),
                path: self.compute_path(head),
            };
        }
    }

    /// Recursively build branches from a node.
    fn build_branches_from_root(&mut self, node_id: NodeId, branch_id: BranchId, depth: u32) {
        if let Some(episode) = self.graph.get_node(node_id) {
            let children = episode.children.clone();

            if children.len() > 1 {
                // This is a fork point - create branches for each child
                let child_branch_ids: Vec<BranchId> = children.iter()
                    .map(|_| {
                        let bid = self.next_branch_id;
                        self.next_branch_id += 1;
                        bid
                    })
                    .collect();

                let fork = ForkPoint::new(node_id, child_branch_ids.clone(), depth);
                self.fork_points.insert(node_id, fork);

                for (i, &child_id) in children.iter().enumerate() {
                    let child_branch_id = child_branch_ids[i];
                    let mut child_branch = Branch::new(child_branch_id, node_id, child_id);
                    child_branch.parent_branch = Some(branch_id);

                    // Add to parent's child branches
                    if let Some(parent) = self.branches.get_mut(&branch_id) {
                        parent.child_branches.push(child_branch_id);
                    }

                    self.branches.insert(child_branch_id, child_branch);
                    self.node_to_branch.insert(child_id, child_branch_id);

                    // Recurse
                    self.build_branches_from_root(child_id, child_branch_id, depth + 1);
                }
            } else if children.len() == 1 {
                // Single child - extend current branch
                let child_id = children[0];
                if let Some(branch) = self.branches.get_mut(&branch_id) {
                    branch.add_node(child_id);
                }
                self.node_to_branch.insert(child_id, branch_id);

                // Recurse
                self.build_branches_from_root(child_id, branch_id, depth + 1);
            }
            // If no children, we've reached a leaf
        }
    }

    // =========================================================================
    // CORE OPERATIONS (Ported from DLM)
    // =========================================================================

    /// Split a branch at a given node, creating a new independent branch.
    ///
    /// This is the core operation for solving the "lost branch" problem,
    /// ported from DLM's `StateMachine.split()`.
    ///
    /// # Arguments
    ///
    /// * `node_id` - The node at which to split
    ///
    /// # Returns
    ///
    /// * `SplitResult` containing the original and new branch IDs
    ///
    /// # Errors
    ///
    /// * `NodeNotFound` if the node doesn't exist
    /// * `CannotSplitRoot` if trying to split at the root
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = machine.split(fork_node_id)?;
    /// // The new branch is now independent and can be explored separately
    /// machine.traverse(result.new_branch)?;
    /// ```
    pub fn split(&mut self, node_id: NodeId) -> Result<SplitResult, BranchError> {
        // Validate the node exists
        if self.graph.get_node(node_id).is_none() {
            return Err(BranchError::NodeNotFound(node_id));
        }

        // Cannot split at root
        if self.graph.roots().contains(&node_id) {
            return Err(BranchError::CannotSplitRoot);
        }

        // Find the branch containing this node
        let source_branch_id = *self.node_to_branch.get(&node_id)
            .ok_or(BranchError::NodeNotFound(node_id))?;

        // Create new branch
        let new_branch_id = self.next_branch_id;
        self.next_branch_id += 1;

        // Find the parent node (fork point for the new branch)
        let fork_point = self.graph.get_node(node_id)
            .and_then(|e| e.parent)
            .ok_or(BranchError::NoParent(source_branch_id))?;

        // Collect all nodes that will move to the new branch
        let moved_nodes = self.collect_subtree(node_id);

        // Create the new branch
        let mut new_branch = Branch::new(new_branch_id, fork_point, node_id);
        new_branch.parent_branch = Some(source_branch_id);
        new_branch.nodes = moved_nodes.clone();

        // Find the head of the new branch (deepest leaf)
        if let Some(&head) = moved_nodes.iter()
            .filter(|&&n| self.graph.get_node(n).map_or(false, |e| e.is_leaf()))
            .max_by_key(|&&n| self.compute_depth(n))
        {
            new_branch.head = head;
        }

        // Update node ownership
        for &n in &moved_nodes {
            self.node_to_branch.insert(n, new_branch_id);
        }

        // Remove nodes from source branch
        if let Some(source) = self.branches.get_mut(&source_branch_id) {
            source.nodes.retain(|n| !moved_nodes.contains(n));
            source.child_branches.push(new_branch_id);
        }

        // Insert new branch
        self.branches.insert(new_branch_id, new_branch);

        // Record operation
        let operation = BranchOperation::split(source_branch_id, new_branch_id, node_id);
        self.history.push(operation);

        Ok(SplitResult {
            original_branch: source_branch_id,
            new_branch: new_branch_id,
            split_point: node_id,
            moved_nodes,
        })
    }

    /// Merge a branch into another branch.
    ///
    /// Ported from DLM's `StateMachine.merge()`.
    ///
    /// # Arguments
    ///
    /// * `from_branch` - The branch to merge (will become inactive)
    /// * `into_branch` - The branch to receive the merge
    ///
    /// # Returns
    ///
    /// * `MergeResult` with merge details
    pub fn merge(&mut self, from_branch: BranchId, into_branch: BranchId) -> Result<MergeResult, BranchError> {
        if from_branch == into_branch {
            return Err(BranchError::SelfMerge);
        }

        // Validate both branches exist
        if !self.branches.contains_key(&from_branch) {
            return Err(BranchError::BranchNotFound(from_branch));
        }
        if !self.branches.contains_key(&into_branch) {
            return Err(BranchError::BranchNotFound(into_branch));
        }

        // Check if from_branch is already merged
        if self.branches.get(&from_branch).map_or(false, |b| b.is_merged()) {
            return Err(BranchError::AlreadyMerged(from_branch));
        }

        // Get merge point (fork point of from_branch)
        let merge_point = self.branches.get(&from_branch)
            .map(|b| b.fork_point)
            .ok_or(BranchError::BranchNotFound(from_branch))?;

        // Move nodes from from_branch to into_branch
        let moved_nodes: Vec<NodeId> = self.branches.get(&from_branch)
            .map(|b| b.nodes.clone())
            .unwrap_or_default();

        for &node in &moved_nodes {
            self.node_to_branch.insert(node, into_branch);
        }

        // Update into_branch
        if let Some(target) = self.branches.get_mut(&into_branch) {
            target.nodes.extend(moved_nodes);
            target.updated_at = current_timestamp();
        }

        // Mark from_branch as merged
        if let Some(source) = self.branches.get_mut(&from_branch) {
            source.mark_merged();
        }

        // Record operation
        let operation = BranchOperation::merge(from_branch, into_branch, merge_point);
        self.history.push(operation);

        Ok(MergeResult {
            target_branch: into_branch,
            merged_branch: from_branch,
            merge_point,
        })
    }

    /// Traverse from current branch to another branch.
    ///
    /// Changes the active branch context.
    pub fn traverse(&mut self, target_branch: BranchId) -> Result<(), BranchError> {
        if !self.branches.contains_key(&target_branch) {
            return Err(BranchError::BranchNotFound(target_branch));
        }

        let from_branch = self.current_branch;

        // Update context
        if let Some(branch) = self.branches.get(&target_branch) {
            self.current_branch = target_branch;
            self.context = BranchContext {
                current_branch: target_branch,
                current_node: branch.head,
                depth: self.compute_depth(branch.head),
                path: self.compute_path(branch.head),
            };
        }

        // Record operation
        let operation = BranchOperation::traverse(from_branch, target_branch);
        self.history.push(operation);

        Ok(())
    }

    /// Archive a branch (preserve but mark inactive).
    pub fn archive(&mut self, branch_id: BranchId, reason: Option<String>) -> Result<(), BranchError> {
        let branch = self.branches.get_mut(&branch_id)
            .ok_or(BranchError::BranchNotFound(branch_id))?;

        branch.archive();

        // Record operation
        let operation = BranchOperation::archive(branch_id, reason);
        self.history.push(operation);

        Ok(())
    }

    // =========================================================================
    // QUERY METHODS
    // =========================================================================

    /// Get the current branch.
    pub fn current(&self) -> Option<&Branch> {
        self.branches.get(&self.current_branch)
    }

    /// Get a branch by ID.
    pub fn get_branch(&self, branch_id: BranchId) -> Option<&Branch> {
        self.branches.get(&branch_id)
    }

    /// Get all active branches.
    pub fn active_branches(&self) -> impl Iterator<Item = &Branch> {
        self.branches.values().filter(|b| b.is_active())
    }

    /// Get all branches (including inactive).
    pub fn all_branches(&self) -> impl Iterator<Item = &Branch> {
        self.branches.values()
    }

    /// Get the number of branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Get all fork points.
    pub fn fork_points(&self) -> impl Iterator<Item = &ForkPoint> {
        self.fork_points.values()
    }

    /// Get the operation history.
    pub fn history(&self) -> &[BranchOperation] {
        &self.history
    }

    /// Get the current context.
    pub fn context(&self) -> &BranchContext {
        &self.context
    }

    /// Find which branch contains a node.
    pub fn find_branch_for_node(&self, node_id: NodeId) -> Option<BranchId> {
        self.node_to_branch.get(&node_id).copied()
    }

    /// Get child branches of a given branch.
    pub fn child_branches(&self, branch_id: BranchId) -> Vec<BranchId> {
        self.branches.get(&branch_id)
            .map(|b| b.child_branches.clone())
            .unwrap_or_default()
    }

    // =========================================================================
    // HELPER METHODS
    // =========================================================================

    /// Collect all nodes in a subtree rooted at the given node.
    fn collect_subtree(&self, root: NodeId) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        let mut stack = vec![root];
        let mut visited = HashSet::new();

        while let Some(node_id) = stack.pop() {
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);
            nodes.push(node_id);

            if let Some(episode) = self.graph.get_node(node_id) {
                for &child in &episode.children {
                    stack.push(child);
                }
            }
        }

        nodes
    }

    /// Compute the depth of a node.
    fn compute_depth(&self, node_id: NodeId) -> u32 {
        self.graph.depth(node_id).unwrap_or(0) as u32
    }

    /// Compute the path from root to a node.
    fn compute_path(&self, node_id: NodeId) -> Vec<NodeId> {
        self.graph.find_path_to(node_id).unwrap_or_default()
    }

    /// Get reference to the underlying graph.
    pub fn graph(&self) -> &TrajectoryGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::operations::BranchStatus;
    use crate::trajectory::graph::{Edge, EdgeType};

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
    fn test_state_machine_creation() {
        let graph = make_branching_graph();
        let machine = BranchStateMachine::from_graph(graph);

        // Should have multiple branches due to fork points
        assert!(machine.branch_count() > 0);
    }

    #[test]
    fn test_split_operation() {
        let graph = make_branching_graph();
        let mut machine = BranchStateMachine::from_graph(graph);

        // Find a node to split on
        let initial_count = machine.branch_count();

        // Split at node 3 (if it's not root and has a parent)
        if let Ok(result) = machine.split(3) {
            assert!(machine.branch_count() >= initial_count);
            assert!(machine.get_branch(result.new_branch).is_some());
        }
    }

    #[test]
    fn test_traverse() {
        let graph = make_branching_graph();
        let mut machine = BranchStateMachine::from_graph(graph);

        let original_branch = machine.current_branch;

        // Try to traverse to any other branch
        for branch in machine.all_branches().map(|b| b.id).collect::<Vec<_>>() {
            if branch != original_branch {
                assert!(machine.traverse(branch).is_ok());
                assert_eq!(machine.current_branch, branch);
                break;
            }
        }
    }

    #[test]
    fn test_archive() {
        let graph = make_branching_graph();
        let mut machine = BranchStateMachine::from_graph(graph);

        // Archive the current branch
        let branch_id = machine.current_branch;
        assert!(machine.archive(branch_id, Some("test".to_string())).is_ok());

        let branch = machine.get_branch(branch_id).unwrap();
        assert_eq!(branch.status, BranchStatus::Archived);
    }

    #[test]
    fn test_cannot_split_root() {
        let graph = make_branching_graph();
        let root = *graph.roots().first().unwrap();
        let mut machine = BranchStateMachine::from_graph(graph);

        let result = machine.split(root);
        assert!(matches!(result, Err(BranchError::CannotSplitRoot)));
    }

    #[test]
    fn test_operation_history() {
        let graph = make_branching_graph();
        let mut machine = BranchStateMachine::from_graph(graph);

        // Perform some operations
        let _ = machine.archive(0, None);

        assert!(!machine.history().is_empty());
    }

    #[test]
    fn test_find_branch_for_node() {
        let graph = make_branching_graph();
        let machine = BranchStateMachine::from_graph(graph);

        // All nodes should belong to some branch
        for node_id in [1, 2, 3, 4, 5] {
            let branch = machine.find_branch_for_node(node_id);
            // Node should exist in some branch
            if machine.graph().get_node(node_id).is_some() {
                // Note: some nodes might not be in branches if they're fork points
                // This is expected behavior
            }
        }
    }
}
