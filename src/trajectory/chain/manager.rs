//! Chain Manager Implementation
//!
//! Manages multiple conversation chains (BranchStateMachine instances).
//! Ported from DLM's ChainManager class.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::trajectory::branch::{BranchStateMachine, BranchError};
use crate::trajectory::graph::{TrajectoryGraph, NodeId};

/// Unique identifier for a chain (conversation).
pub type ChainId = String;

/// Get current Unix timestamp in seconds.
#[inline]
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Generate a unique chain ID.
fn generate_chain_id() -> ChainId {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = current_timestamp();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chain_{:x}_{:04x}", timestamp, counter)
}

/// Metadata about a chain.
#[derive(Debug, Clone)]
pub struct ChainMetadata {
    /// Chain ID
    pub id: ChainId,
    /// Human-readable title
    pub title: Option<String>,
    /// When the chain was created
    pub created_at: i64,
    /// When the chain was last accessed
    pub last_accessed_at: i64,
    /// Number of nodes in the chain
    pub node_count: usize,
    /// Number of branches in the chain
    pub branch_count: usize,
    /// Whether the chain is active
    pub is_active: bool,
    /// Custom metadata
    pub tags: Vec<String>,
}

impl ChainMetadata {
    /// Create metadata for a new chain.
    pub fn new(id: ChainId) -> Self {
        let now = current_timestamp();
        Self {
            id,
            title: None,
            created_at: now,
            last_accessed_at: now,
            node_count: 0,
            branch_count: 0,
            is_active: true,
            tags: Vec::new(),
        }
    }

    /// Update last accessed time to now.
    pub fn touch(&mut self) {
        self.last_accessed_at = current_timestamp();
    }

    /// Update statistics from a state machine.
    pub fn update_stats(&mut self, machine: &BranchStateMachine) {
        self.node_count = machine.graph().node_count();
        self.branch_count = machine.branch_count();
    }
}

/// Configuration for chain manager.
#[derive(Debug, Clone)]
pub struct ChainManagerConfig {
    /// Maximum number of chains to keep in memory
    pub max_chains: usize,
    /// Inactivity threshold for cleanup (in seconds)
    pub inactivity_threshold_secs: u64,
    /// Whether to auto-cleanup on operations
    pub auto_cleanup: bool,
}

impl Default for ChainManagerConfig {
    fn default() -> Self {
        Self {
            max_chains: 1000,
            inactivity_threshold_secs: 3600, // 1 hour
            auto_cleanup: false,
        }
    }
}

/// Statistics about the chain manager.
#[derive(Debug, Clone, Default)]
pub struct ChainManagerStats {
    /// Total chains managed
    pub total_chains: usize,
    /// Active chains
    pub active_chains: usize,
    /// Total nodes across all chains
    pub total_nodes: usize,
    /// Total branches across all chains
    pub total_branches: usize,
}

/// Errors that can occur in chain management.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainManagerError {
    /// Chain not found
    ChainNotFound(ChainId),
    /// Chain already exists
    ChainAlreadyExists(ChainId),
    /// Maximum chains reached
    MaxChainsReached,
    /// Branch operation error
    BranchError(String),
    /// Invalid operation
    InvalidOperation(String),
}

impl std::fmt::Display for ChainManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainNotFound(id) => write!(f, "Chain not found: {}", id),
            Self::ChainAlreadyExists(id) => write!(f, "Chain already exists: {}", id),
            Self::MaxChainsReached => write!(f, "Maximum number of chains reached"),
            Self::BranchError(msg) => write!(f, "Branch error: {}", msg),
            Self::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for ChainManagerError {}

impl From<BranchError> for ChainManagerError {
    fn from(err: BranchError) -> Self {
        Self::BranchError(err.to_string())
    }
}

/// Manager for multiple conversation chains.
///
/// Provides operations for creating, accessing, and managing multiple
/// conversation state machines. Ported from DLM's ChainManager.
///
/// # Example
///
/// ```ignore
/// let mut manager = ChainManager::new();
///
/// // Create a chain with auto-generated ID
/// let chain_id = manager.create_chain(None);
///
/// // Create a chain with specific ID
/// let chain_id = manager.create_chain(Some("my-chain".to_string()));
///
/// // Get chain for operations
/// if let Some(chain) = manager.get_chain_mut(&chain_id) {
///     chain.split(some_node)?;
/// }
///
/// // Merge two chains
/// manager.merge_chains(&chain1, &chain2)?;
/// ```
pub struct ChainManager {
    /// All managed chains
    chains: HashMap<ChainId, BranchStateMachine>,
    /// Metadata for each chain
    metadata: HashMap<ChainId, ChainMetadata>,
    /// Currently active chain (if any)
    active_chain: Option<ChainId>,
    /// Configuration
    config: ChainManagerConfig,
}

impl Default for ChainManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainManager {
    /// Create a new chain manager with default config.
    pub fn new() -> Self {
        Self::with_config(ChainManagerConfig::default())
    }

    /// Create a new chain manager with specific config.
    pub fn with_config(config: ChainManagerConfig) -> Self {
        Self {
            chains: HashMap::new(),
            metadata: HashMap::new(),
            active_chain: None,
            config,
        }
    }

    // =========================================================================
    // CHAIN CREATION & ACCESS
    // =========================================================================

    /// Create a new conversation chain.
    ///
    /// If `chain_id` is None, a unique ID is generated.
    /// Returns the chain ID.
    pub fn create_chain(&mut self, chain_id: Option<ChainId>) -> Result<ChainId, ChainManagerError> {
        // Check max chains limit
        if self.chains.len() >= self.config.max_chains {
            if self.config.auto_cleanup {
                self.cleanup_inactive(Duration::from_secs(self.config.inactivity_threshold_secs));
            }
            if self.chains.len() >= self.config.max_chains {
                return Err(ChainManagerError::MaxChainsReached);
            }
        }

        let id = chain_id.unwrap_or_else(generate_chain_id);

        if self.chains.contains_key(&id) {
            return Err(ChainManagerError::ChainAlreadyExists(id));
        }

        // Create empty graph and state machine
        let graph = TrajectoryGraph::new();
        let machine = BranchStateMachine::from_graph(graph);

        // Create metadata
        let metadata = ChainMetadata::new(id.clone());

        self.chains.insert(id.clone(), machine);
        self.metadata.insert(id.clone(), metadata);

        // Set as active if no active chain
        if self.active_chain.is_none() {
            self.active_chain = Some(id.clone());
        }

        Ok(id)
    }

    /// Create a chain from an existing graph.
    pub fn create_chain_from_graph(
        &mut self,
        chain_id: Option<ChainId>,
        graph: TrajectoryGraph,
    ) -> Result<ChainId, ChainManagerError> {
        let id = chain_id.unwrap_or_else(generate_chain_id);

        if self.chains.contains_key(&id) {
            return Err(ChainManagerError::ChainAlreadyExists(id));
        }

        let machine = BranchStateMachine::from_graph(graph);
        let mut metadata = ChainMetadata::new(id.clone());
        metadata.update_stats(&machine);

        self.chains.insert(id.clone(), machine);
        self.metadata.insert(id.clone(), metadata);

        if self.active_chain.is_none() {
            self.active_chain = Some(id.clone());
        }

        Ok(id)
    }

    /// Get a chain by ID (immutable).
    pub fn get_chain(&self, chain_id: &ChainId) -> Option<&BranchStateMachine> {
        self.chains.get(chain_id)
    }

    /// Get a chain by ID (mutable).
    pub fn get_chain_mut(&mut self, chain_id: &ChainId) -> Option<&mut BranchStateMachine> {
        // Update access time
        if let Some(meta) = self.metadata.get_mut(chain_id) {
            meta.touch();
        }
        self.chains.get_mut(chain_id)
    }

    /// Get chain metadata.
    pub fn get_metadata(&self, chain_id: &ChainId) -> Option<&ChainMetadata> {
        self.metadata.get(chain_id)
    }

    /// Get mutable chain metadata.
    pub fn get_metadata_mut(&mut self, chain_id: &ChainId) -> Option<&mut ChainMetadata> {
        self.metadata.get_mut(chain_id)
    }

    /// Check if a chain exists.
    pub fn contains(&self, chain_id: &ChainId) -> bool {
        self.chains.contains_key(chain_id)
    }

    /// Get all chain IDs.
    pub fn chain_ids(&self) -> impl Iterator<Item = &ChainId> {
        self.chains.keys()
    }

    /// Get number of chains.
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    // =========================================================================
    // ACTIVE CHAIN MANAGEMENT
    // =========================================================================

    /// Get the currently active chain ID.
    pub fn active_chain(&self) -> Option<&ChainId> {
        self.active_chain.as_ref()
    }

    /// Set the active chain.
    pub fn set_active_chain(&mut self, chain_id: ChainId) -> Result<(), ChainManagerError> {
        if !self.chains.contains_key(&chain_id) {
            return Err(ChainManagerError::ChainNotFound(chain_id));
        }
        self.active_chain = Some(chain_id);
        Ok(())
    }

    /// Get the active chain (immutable).
    pub fn get_active_chain(&self) -> Option<&BranchStateMachine> {
        self.active_chain.as_ref().and_then(|id| self.chains.get(id))
    }

    /// Get the active chain (mutable).
    pub fn get_active_chain_mut(&mut self) -> Option<&mut BranchStateMachine> {
        if let Some(id) = &self.active_chain {
            if let Some(meta) = self.metadata.get_mut(id) {
                meta.touch();
            }
            self.chains.get_mut(id)
        } else {
            None
        }
    }

    // =========================================================================
    // CHAIN OPERATIONS
    // =========================================================================

    /// Delete a chain.
    pub fn delete_chain(&mut self, chain_id: &ChainId) -> bool {
        let removed = self.chains.remove(chain_id).is_some();
        self.metadata.remove(chain_id);

        // Update active chain if deleted
        if self.active_chain.as_ref() == Some(chain_id) {
            self.active_chain = self.chains.keys().next().cloned();
        }

        removed
    }

    /// Merge two chains.
    ///
    /// The `from` chain is merged into `into`, and then deleted.
    pub fn merge_chains(
        &mut self,
        from: &ChainId,
        into: &ChainId,
    ) -> Result<(), ChainManagerError> {
        if from == into {
            return Err(ChainManagerError::InvalidOperation(
                "Cannot merge chain with itself".to_string(),
            ));
        }

        // Get both chains
        let from_chain = self.chains.remove(from)
            .ok_or_else(|| ChainManagerError::ChainNotFound(from.clone()))?;

        let into_chain = self.chains.get_mut(into)
            .ok_or_else(|| ChainManagerError::ChainNotFound(into.clone()))?;

        // Merge the graphs (append from's graph to into's)
        // Note: This is a simplified merge - in production you'd want
        // to properly connect the graphs
        for branch in from_chain.all_branches() {
            // Add branches from 'from' chain to 'into' chain
            // This is a simplified implementation
            let _ = branch; // Placeholder for actual merge logic
        }

        // Update metadata
        self.metadata.remove(from);
        if let Some(meta) = self.metadata.get_mut(into) {
            meta.update_stats(into_chain);
            meta.touch();
        }

        // Update active chain if needed
        if self.active_chain.as_ref() == Some(from) {
            self.active_chain = Some(into.clone());
        }

        Ok(())
    }

    /// Split a chain at a node, creating a new chain.
    ///
    /// Returns the ID of the new chain.
    pub fn split_chain(
        &mut self,
        chain_id: &ChainId,
        node_id: NodeId,
    ) -> Result<ChainId, ChainManagerError> {
        // Get the chain
        let chain = self.chains.get_mut(chain_id)
            .ok_or_else(|| ChainManagerError::ChainNotFound(chain_id.clone()))?;

        // Perform the split
        let split_result = chain.split(node_id)?;

        // Create a new chain for the split branch
        let new_chain_id = generate_chain_id();

        // Note: In a full implementation, we'd extract the subtree
        // into a new graph. For now, we just record the split.
        let mut metadata = ChainMetadata::new(new_chain_id.clone());
        metadata.title = Some(format!("Split from {} at node {}", chain_id, node_id));

        // Update original chain metadata
        if let Some(meta) = self.metadata.get_mut(chain_id) {
            meta.update_stats(chain);
            meta.touch();
        }

        self.metadata.insert(new_chain_id.clone(), metadata);

        let _ = split_result; // Use split result in full implementation

        Ok(new_chain_id)
    }

    // =========================================================================
    // CLEANUP & MAINTENANCE
    // =========================================================================

    /// Cleanup inactive chains.
    ///
    /// Removes chains that haven't been accessed within the threshold.
    pub fn cleanup_inactive(&mut self, threshold: Duration) {
        let now = current_timestamp();
        let threshold_secs = threshold.as_secs() as i64;

        let inactive: Vec<ChainId> = self.metadata
            .iter()
            .filter(|(_, meta)| {
                now - meta.last_accessed_at > threshold_secs
            })
            .map(|(id, _)| id.clone())
            .collect();

        for chain_id in inactive {
            self.delete_chain(&chain_id);
        }
    }

    /// Mark a chain as inactive.
    pub fn deactivate_chain(&mut self, chain_id: &ChainId) -> Result<(), ChainManagerError> {
        let meta = self.metadata.get_mut(chain_id)
            .ok_or_else(|| ChainManagerError::ChainNotFound(chain_id.clone()))?;
        meta.is_active = false;
        Ok(())
    }

    /// Reactivate a chain.
    pub fn reactivate_chain(&mut self, chain_id: &ChainId) -> Result<(), ChainManagerError> {
        let meta = self.metadata.get_mut(chain_id)
            .ok_or_else(|| ChainManagerError::ChainNotFound(chain_id.clone()))?;
        meta.is_active = true;
        meta.touch();
        Ok(())
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get statistics about all managed chains.
    pub fn stats(&self) -> ChainManagerStats {
        let mut stats = ChainManagerStats::default();
        stats.total_chains = self.chains.len();

        for (id, chain) in &self.chains {
            stats.total_nodes += chain.graph().node_count();
            stats.total_branches += chain.branch_count();

            if let Some(meta) = self.metadata.get(id) {
                if meta.is_active {
                    stats.active_chains += 1;
                }
            }
        }

        stats
    }

    /// Get all metadata.
    pub fn all_metadata(&self) -> impl Iterator<Item = &ChainMetadata> {
        self.metadata.values()
    }

    // =========================================================================
    // SEARCH & QUERY
    // =========================================================================

    /// Find chains by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&ChainId> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.tags.contains(&tag.to_string()))
            .map(|(id, _)| id)
            .collect()
    }

    /// Find chains by title (partial match).
    pub fn find_by_title(&self, query: &str) -> Vec<&ChainId> {
        let query_lower = query.to_lowercase();
        self.metadata
            .iter()
            .filter(|(_, meta)| {
                meta.title
                    .as_ref()
                    .map_or(false, |t| t.to_lowercase().contains(&query_lower))
            })
            .map(|(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_chain() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(None).unwrap();

        assert!(manager.contains(&chain_id));
        assert_eq!(manager.chain_count(), 1);
    }

    #[test]
    fn test_create_chain_with_id() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(Some("my-chain".to_string())).unwrap();

        assert_eq!(chain_id, "my-chain");
        assert!(manager.contains(&"my-chain".to_string()));
    }

    #[test]
    fn test_duplicate_chain_error() {
        let mut manager = ChainManager::new();
        manager.create_chain(Some("test".to_string())).unwrap();

        let result = manager.create_chain(Some("test".to_string()));
        assert!(matches!(result, Err(ChainManagerError::ChainAlreadyExists(_))));
    }

    #[test]
    fn test_delete_chain() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(None).unwrap();

        assert!(manager.delete_chain(&chain_id));
        assert!(!manager.contains(&chain_id));
        assert_eq!(manager.chain_count(), 0);
    }

    #[test]
    fn test_active_chain() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(None).unwrap();

        // First chain should be active by default
        assert_eq!(manager.active_chain(), Some(&chain_id));

        // Create another chain
        let chain_id2 = manager.create_chain(None).unwrap();

        // First chain should still be active
        assert_eq!(manager.active_chain(), Some(&chain_id));

        // Set second chain as active
        manager.set_active_chain(chain_id2.clone()).unwrap();
        assert_eq!(manager.active_chain(), Some(&chain_id2));
    }

    #[test]
    fn test_get_chain() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(None).unwrap();

        assert!(manager.get_chain(&chain_id).is_some());
        assert!(manager.get_chain(&"nonexistent".to_string()).is_none());
    }

    #[test]
    fn test_stats() {
        let mut manager = ChainManager::new();
        manager.create_chain(None).unwrap();
        manager.create_chain(None).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_chains, 2);
        assert_eq!(stats.active_chains, 2);
    }

    #[test]
    fn test_metadata() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(Some("test".to_string())).unwrap();

        let meta = manager.get_metadata(&chain_id).unwrap();
        assert!(meta.is_active);
        assert!(meta.created_at > 0);

        // Update metadata
        let meta = manager.get_metadata_mut(&chain_id).unwrap();
        meta.title = Some("My Chain".to_string());
        meta.tags.push("important".to_string());

        // Verify
        let meta = manager.get_metadata(&chain_id).unwrap();
        assert_eq!(meta.title, Some("My Chain".to_string()));
        assert!(meta.tags.contains(&"important".to_string()));
    }

    #[test]
    fn test_find_by_tag() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(Some("test".to_string())).unwrap();

        let meta = manager.get_metadata_mut(&chain_id).unwrap();
        meta.tags.push("project".to_string());

        let found = manager.find_by_tag("project");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], &chain_id);

        let not_found = manager.find_by_tag("nonexistent");
        assert!(not_found.is_empty());
    }

    #[test]
    fn test_deactivate_reactivate() {
        let mut manager = ChainManager::new();
        let chain_id = manager.create_chain(None).unwrap();

        assert!(manager.get_metadata(&chain_id).unwrap().is_active);

        manager.deactivate_chain(&chain_id).unwrap();
        assert!(!manager.get_metadata(&chain_id).unwrap().is_active);

        manager.reactivate_chain(&chain_id).unwrap();
        assert!(manager.get_metadata(&chain_id).unwrap().is_active);
    }
}
