//! Chain Management Module
//!
//! Manages multiple conversation chains (state machines) and provides
//! cross-chain operations. Ported from DLM's ChainManager.
//!
//! # Concept
//!
//! A "chain" is a single conversation trajectory managed by a `BranchStateMachine`.
//! The `ChainManager` coordinates multiple chains, enabling:
//!
//! - Multi-conversation tracking
//! - Cross-chain merging and splitting
//! - Cleanup of inactive chains
//! - Serialization/deserialization of chain state
//!
//! # Example
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::chain::ChainManager;
//!
//! let mut manager = ChainManager::new();
//!
//! // Create a new conversation chain
//! let chain_id = manager.create_chain(None);
//!
//! // Get a chain for operations
//! let chain = manager.get_chain_mut(&chain_id)?;
//! chain.split(some_node_id)?;
//!
//! // Merge two chains
//! manager.merge_chains(&chain_id_1, &chain_id_2)?;
//!
//! // Cleanup old chains
//! manager.cleanup_inactive(Duration::from_secs(3600));
//! ```

mod manager;
mod links;

pub use manager::{
    ChainManager, ChainId, ChainMetadata, ChainManagerConfig,
    ChainManagerError, ChainManagerStats,
};
pub use links::{
    CrossChainLink, CrossChainLinkType, LinkStrength,
    find_cross_chain_links, detect_knowledge_transfer,
    KNOWLEDGE_TRANSFER_PATTERNS,
};
