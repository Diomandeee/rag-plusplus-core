//! Branch Operations and Types
//!
//! Core types for branch management in trajectory DAGs.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::trajectory::graph::NodeId;

/// Get current Unix timestamp in seconds.
#[inline]
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Unique identifier for a branch.
pub type BranchId = u64;

/// Status of a branch in the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchStatus {
    /// Currently active branch (being explored)
    Active,
    /// Archived branch (not active but preserved)
    Archived,
    /// Merged into another branch
    Merged,
    /// Recovered from "lost" state
    Recovered,
    /// Orphaned (parent was deleted)
    Orphaned,
}

impl Default for BranchStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// A fork point where branching occurred.
///
/// Represents a node in the DAG where multiple children exist,
/// creating a decision point in the trajectory.
#[derive(Debug, Clone)]
pub struct ForkPoint {
    /// The node ID where forking occurred
    pub node_id: NodeId,
    /// All child branches at this fork
    pub children: Vec<BranchId>,
    /// The selected/active child branch (if any)
    pub selected_child: Option<BranchId>,
    /// Timestamp when fork was created
    pub created_at: i64,
    /// Depth in the trajectory graph
    pub depth: u32,
}

impl ForkPoint {
    /// Create a new fork point.
    pub fn new(node_id: NodeId, children: Vec<BranchId>, depth: u32) -> Self {
        Self {
            node_id,
            children,
            selected_child: None,
            created_at: current_timestamp(),
            depth,
        }
    }

    /// Check if this fork has multiple paths (is actually a branch point).
    #[inline]
    pub fn is_branch_point(&self) -> bool {
        self.children.len() > 1
    }

    /// Get the number of child branches.
    #[inline]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Select a child branch as the active path.
    pub fn select(&mut self, branch_id: BranchId) -> Result<(), BranchError> {
        if self.children.contains(&branch_id) {
            self.selected_child = Some(branch_id);
            Ok(())
        } else {
            Err(BranchError::InvalidBranch(branch_id))
        }
    }
}

/// A branch in the trajectory state machine.
///
/// Represents a linear path through the DAG from a fork point to a leaf
/// (or to another fork point). Branches can be split off into independent
/// state machines and later merged back.
#[derive(Debug, Clone)]
pub struct Branch {
    /// Unique identifier for this branch
    pub id: BranchId,
    /// Where this branch diverged from its parent
    pub fork_point: NodeId,
    /// Current tip (head) of the branch
    pub head: NodeId,
    /// Current status of the branch
    pub status: BranchStatus,
    /// Parent branch (if this is a child branch)
    pub parent_branch: Option<BranchId>,
    /// Child branches that forked from this one
    pub child_branches: Vec<BranchId>,
    /// Nodes that belong exclusively to this branch
    pub nodes: Vec<NodeId>,
    /// Metadata associated with this branch
    pub metadata: HashMap<String, String>,
    /// Timestamp when branch was created
    pub created_at: i64,
    /// Timestamp when branch was last updated
    pub updated_at: i64,
    /// Human-readable label (optional)
    pub label: Option<String>,
}

impl Branch {
    /// Create a new branch.
    pub fn new(id: BranchId, fork_point: NodeId, head: NodeId) -> Self {
        let now = current_timestamp();
        Self {
            id,
            fork_point,
            head,
            status: BranchStatus::Active,
            parent_branch: None,
            child_branches: Vec::new(),
            nodes: vec![head],
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            label: None,
        }
    }

    /// Create a root branch (no fork point).
    pub fn root(id: BranchId, root_node: NodeId) -> Self {
        let now = current_timestamp();
        Self {
            id,
            fork_point: root_node,
            head: root_node,
            status: BranchStatus::Active,
            parent_branch: None,
            child_branches: Vec::new(),
            nodes: vec![root_node],
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            label: Some("main".to_string()),
        }
    }

    /// Check if this branch is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.status == BranchStatus::Active
    }

    /// Check if this branch is the main/root branch.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent_branch.is_none()
    }

    /// Check if this branch has been merged.
    #[inline]
    pub fn is_merged(&self) -> bool {
        self.status == BranchStatus::Merged
    }

    /// Archive this branch (preserve but mark inactive).
    pub fn archive(&mut self) {
        self.status = BranchStatus::Archived;
        self.updated_at = current_timestamp();
    }

    /// Mark this branch as merged.
    pub fn mark_merged(&mut self) {
        self.status = BranchStatus::Merged;
        self.updated_at = current_timestamp();
    }

    /// Recover this branch (reactivate from archived/orphaned).
    pub fn recover(&mut self) {
        self.status = BranchStatus::Recovered;
        self.updated_at = current_timestamp();
    }

    /// Add a node to this branch.
    pub fn add_node(&mut self, node_id: NodeId) {
        self.nodes.push(node_id);
        self.head = node_id;
        self.updated_at = current_timestamp();
    }

    /// Get the length of this branch (number of nodes).
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if branch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Set a label for this branch.
    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = Some(label.into());
        self.updated_at = current_timestamp();
    }

    /// Add metadata to this branch.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
        self.updated_at = current_timestamp();
    }
}

/// Operations that can be performed on branches.
///
/// These operations form the core of the "lost branch" solution,
/// ported from DLM's StateMachine operations.
#[derive(Debug, Clone)]
pub enum BranchOperation {
    /// Split a branch at a node, creating a new independent branch.
    /// Ported from DLM `StateMachine.split()`.
    Split {
        /// Source branch being split
        from: BranchId,
        /// New branch created by the split
        to: BranchId,
        /// Node where the split occurred
        at: NodeId,
        /// Timestamp of the operation
        timestamp: i64,
    },
    /// Merge a branch back into another.
    /// Ported from DLM `StateMachine.merge()`.
    Merge {
        /// Branch being merged (will become inactive)
        from: BranchId,
        /// Branch receiving the merge
        into: BranchId,
        /// Node where merge occurred
        at: NodeId,
        /// Timestamp of the operation
        timestamp: i64,
    },
    /// Recover a "lost" branch, reactivating it.
    Recover {
        /// Branch being recovered
        branch: BranchId,
        /// How it was recovered
        strategy: String,
        /// Timestamp of the operation
        timestamp: i64,
    },
    /// Traverse from one branch to another.
    Traverse {
        /// Branch being left
        from: BranchId,
        /// Branch being entered
        to: BranchId,
        /// Timestamp of the operation
        timestamp: i64,
    },
    /// Archive a branch (preserve but deactivate).
    Archive {
        /// Branch being archived
        branch: BranchId,
        /// Reason for archiving
        reason: Option<String>,
        /// Timestamp of the operation
        timestamp: i64,
    },
    /// Create a new branch at a fork point.
    Fork {
        /// Parent branch
        parent: BranchId,
        /// New branch created
        child: BranchId,
        /// Fork point node
        at: NodeId,
        /// Timestamp of the operation
        timestamp: i64,
    },
}

impl BranchOperation {
    /// Create a new split operation.
    pub fn split(from: BranchId, to: BranchId, at: NodeId) -> Self {
        Self::Split {
            from,
            to,
            at,
            timestamp: current_timestamp(),
        }
    }

    /// Create a new merge operation.
    pub fn merge(from: BranchId, into: BranchId, at: NodeId) -> Self {
        Self::Merge {
            from,
            into,
            at,
            timestamp: current_timestamp(),
        }
    }

    /// Create a new recover operation.
    pub fn recover(branch: BranchId, strategy: impl Into<String>) -> Self {
        Self::Recover {
            branch,
            strategy: strategy.into(),
            timestamp: current_timestamp(),
        }
    }

    /// Create a new traverse operation.
    pub fn traverse(from: BranchId, to: BranchId) -> Self {
        Self::Traverse {
            from,
            to,
            timestamp: current_timestamp(),
        }
    }

    /// Create a new archive operation.
    pub fn archive(branch: BranchId, reason: Option<String>) -> Self {
        Self::Archive {
            branch,
            reason,
            timestamp: current_timestamp(),
        }
    }

    /// Create a new fork operation.
    pub fn fork(parent: BranchId, child: BranchId, at: NodeId) -> Self {
        Self::Fork {
            parent,
            child,
            at,
            timestamp: current_timestamp(),
        }
    }

    /// Get the timestamp of this operation.
    pub fn timestamp(&self) -> i64 {
        match self {
            Self::Split { timestamp, .. } => *timestamp,
            Self::Merge { timestamp, .. } => *timestamp,
            Self::Recover { timestamp, .. } => *timestamp,
            Self::Traverse { timestamp, .. } => *timestamp,
            Self::Archive { timestamp, .. } => *timestamp,
            Self::Fork { timestamp, .. } => *timestamp,
        }
    }
}

/// Errors that can occur during branch operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchError {
    /// Branch with given ID was not found
    BranchNotFound(BranchId),
    /// Node with given ID was not found
    NodeNotFound(NodeId),
    /// Operation would create a cycle
    CycleDetected,
    /// Cannot split the root node
    CannotSplitRoot,
    /// Branch is already merged
    AlreadyMerged(BranchId),
    /// Invalid branch ID for operation
    InvalidBranch(BranchId),
    /// Cannot merge branch with itself
    SelfMerge,
    /// Branch has no parent
    NoParent(BranchId),
    /// Operation not allowed in current state
    InvalidState(String),
}

impl std::fmt::Display for BranchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BranchNotFound(id) => write!(f, "Branch {} not found", id),
            Self::NodeNotFound(id) => write!(f, "Node {} not found", id),
            Self::CycleDetected => write!(f, "Operation would create a cycle"),
            Self::CannotSplitRoot => write!(f, "Cannot split the root node"),
            Self::AlreadyMerged(id) => write!(f, "Branch {} is already merged", id),
            Self::InvalidBranch(id) => write!(f, "Invalid branch ID: {}", id),
            Self::SelfMerge => write!(f, "Cannot merge branch with itself"),
            Self::NoParent(id) => write!(f, "Branch {} has no parent", id),
            Self::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for BranchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_creation() {
        let branch = Branch::new(1, 100, 101);
        assert_eq!(branch.id, 1);
        assert_eq!(branch.fork_point, 100);
        assert_eq!(branch.head, 101);
        assert!(branch.is_active());
        // Note: parent_branch is None by default, so is_root() returns true
        // This is expected for a newly created branch without explicit parent
        assert!(branch.parent_branch.is_none());
    }

    #[test]
    fn test_root_branch() {
        let branch = Branch::root(0, 1);
        assert!(branch.is_root());
        assert_eq!(branch.label, Some("main".to_string()));
    }

    #[test]
    fn test_branch_archive() {
        let mut branch = Branch::new(1, 100, 101);
        assert!(branch.is_active());
        branch.archive();
        assert!(!branch.is_active());
        assert_eq!(branch.status, BranchStatus::Archived);
    }

    #[test]
    fn test_branch_recover() {
        let mut branch = Branch::new(1, 100, 101);
        branch.archive();
        branch.recover();
        assert_eq!(branch.status, BranchStatus::Recovered);
    }

    #[test]
    fn test_fork_point() {
        let mut fork = ForkPoint::new(100, vec![1, 2, 3], 5);
        assert!(fork.is_branch_point());
        assert_eq!(fork.child_count(), 3);

        assert!(fork.select(2).is_ok());
        assert_eq!(fork.selected_child, Some(2));

        assert!(fork.select(99).is_err());
    }

    #[test]
    fn test_branch_operations() {
        let split_op = BranchOperation::split(1, 2, 100);
        assert!(split_op.timestamp() > 0);

        let merge_op = BranchOperation::merge(2, 1, 100);
        assert!(merge_op.timestamp() > 0);

        let recover_op = BranchOperation::recover(1, "manual");
        assert!(recover_op.timestamp() > 0);
    }

    #[test]
    fn test_branch_add_node() {
        let mut branch = Branch::new(1, 100, 101);
        assert_eq!(branch.len(), 1);

        branch.add_node(102);
        assert_eq!(branch.len(), 2);
        assert_eq!(branch.head, 102);
        assert!(branch.nodes.contains(&102));
    }

    #[test]
    fn test_branch_metadata() {
        let mut branch = Branch::new(1, 100, 101);
        branch.add_metadata("source", "regeneration");
        branch.set_label("experiment-1");

        assert_eq!(branch.metadata.get("source"), Some(&"regeneration".to_string()));
        assert_eq!(branch.label, Some("experiment-1".to_string()));
    }
}
