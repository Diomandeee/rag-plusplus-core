//! Branch Management Module
//!
//! Provides branch operations for trajectory DAGs to solve the "lost branch" problem.
//!
//! # Problem Statement
//!
//! In conversation systems like ChatGPT, users often explore multiple paths:
//! - Regenerating responses creates sibling branches
//! - Editing earlier messages creates divergent branches
//! - Exploring "what if" scenarios creates parallel investigations
//!
//! Without proper branch management, these valuable exploration paths are "lost" -
//! they exist in the data but are inaccessible or forgotten.
//!
//! # Solution: BranchStateMachine
//!
//! The `BranchStateMachine` provides:
//! - **Split**: Extract a branch into an independent state machine (from DLM)
//! - **Merge**: Combine branches back together
//! - **Recover**: Find and reactivate "lost" branches
//! - **Traverse**: Navigate between branches while preserving context
//!
//! # Example
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::branch::{
//!     BranchStateMachine, Branch, BranchOperation,
//! };
//!
//! // Create a branch state machine from a trajectory graph
//! let mut machine = BranchStateMachine::new(graph);
//!
//! // Split at a branch point to create a new independent branch
//! let new_branch = machine.split(fork_point_id)?;
//!
//! // Find all recoverable branches (branches that were "lost")
//! let lost_branches = machine.find_recoverable_branches();
//!
//! // Recover a lost branch
//! machine.recover(lost_branch_id)?;
//! ```
//!
//! # Architecture (from DLM legacy)
//!
//! The design is ported from the DLM (Dialogue Lifecycle Management) package's
//! `StateMachine.split()` operation, which creates a new StateMachine from
//! a subtree of the conversation DAG.

mod operations;
mod resolution;
mod state_machine;

pub use operations::{
    Branch, BranchId, BranchOperation, BranchStatus, ForkPoint, BranchError,
};
pub use resolution::{
    BranchResolver, RecoverableBranch, RecoveryStrategy, LostReason,
};
pub use state_machine::{
    BranchStateMachine, BranchContext, SplitResult, MergeResult,
};
