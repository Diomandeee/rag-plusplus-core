//! Cross-Chain Links
//!
//! Support for linking related content across different conversation chains.
//! Enables knowledge transfer detection and cross-conversation navigation.

use crate::trajectory::graph::NodeId;
use super::manager::ChainId;

/// Strength of a cross-chain link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkStrength {
    /// Semantic similarity (0.0 - 1.0)
    pub semantic: f32,
    /// Temporal proximity (0.0 - 1.0)
    pub temporal: f32,
    /// Thematic overlap (0.0 - 1.0)
    pub thematic: f32,
    /// Combined score
    pub combined: f32,
}

impl LinkStrength {
    /// Create a new link strength with all factors.
    pub fn new(semantic: f32, temporal: f32, thematic: f32) -> Self {
        // Default weighting: semantic 0.5, temporal 0.2, thematic 0.3
        let combined = 0.5 * semantic + 0.2 * temporal + 0.3 * thematic;
        Self {
            semantic,
            temporal,
            thematic,
            combined,
        }
    }

    /// Create link strength from semantic similarity only.
    pub fn from_semantic(similarity: f32) -> Self {
        Self::new(similarity, 0.5, 0.5)
    }

    /// Check if this is a strong link (above threshold).
    pub fn is_strong(&self, threshold: f32) -> bool {
        self.combined >= threshold
    }
}

impl Default for LinkStrength {
    fn default() -> Self {
        Self {
            semantic: 0.0,
            temporal: 0.0,
            thematic: 0.0,
            combined: 0.0,
        }
    }
}

/// A link between nodes in different chains.
///
/// Represents semantic, temporal, or thematic relationships
/// between content in different conversations.
#[derive(Debug, Clone)]
pub struct CrossChainLink {
    /// Source chain
    pub source_chain: ChainId,
    /// Source node
    pub source_node: NodeId,
    /// Target chain
    pub target_chain: ChainId,
    /// Target node
    pub target_node: NodeId,
    /// Link strength
    pub strength: LinkStrength,
    /// Link type (semantic, reference, continuation, etc.)
    pub link_type: CrossChainLinkType,
    /// Optional description
    pub description: Option<String>,
}

/// Types of cross-chain links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossChainLinkType {
    /// Semantic similarity (similar content)
    Semantic,
    /// Explicit reference ("as we discussed before")
    Reference,
    /// Continuation of topic
    Continuation,
    /// Related concept
    Related,
    /// Same topic, different approach
    Alternative,
    /// Knowledge transfer ("similar to project X")
    KnowledgeTransfer,
}

impl CrossChainLink {
    /// Create a new cross-chain link.
    pub fn new(
        source_chain: ChainId,
        source_node: NodeId,
        target_chain: ChainId,
        target_node: NodeId,
        strength: LinkStrength,
        link_type: CrossChainLinkType,
    ) -> Self {
        Self {
            source_chain,
            source_node,
            target_chain,
            target_node,
            strength,
            link_type,
            description: None,
        }
    }

    /// Create a semantic link.
    pub fn semantic(
        source_chain: ChainId,
        source_node: NodeId,
        target_chain: ChainId,
        target_node: NodeId,
        similarity: f32,
    ) -> Self {
        Self::new(
            source_chain,
            source_node,
            target_chain,
            target_node,
            LinkStrength::from_semantic(similarity),
            CrossChainLinkType::Semantic,
        )
    }

    /// Add a description to this link.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Check if this link is bidirectional.
    pub fn is_bidirectional(&self) -> bool {
        // Semantic and related links are typically bidirectional
        matches!(
            self.link_type,
            CrossChainLinkType::Semantic | CrossChainLinkType::Related
        )
    }

    /// Get the reverse link (swap source and target).
    pub fn reverse(&self) -> Self {
        Self {
            source_chain: self.target_chain.clone(),
            source_node: self.target_node,
            target_chain: self.source_chain.clone(),
            target_node: self.source_node,
            strength: self.strength,
            link_type: self.link_type,
            description: self.description.clone(),
        }
    }
}

/// Find cross-chain links between nodes based on embeddings.
///
/// This is a placeholder for the actual implementation which would:
/// 1. Compute embedding similarities between nodes across chains
/// 2. Apply thresholds to filter weak links
/// 3. Detect knowledge transfer patterns
///
/// # Arguments
///
/// * `chain_embeddings` - Map of chain_id -> (node_id, embedding)
/// * `threshold` - Minimum similarity threshold
///
/// # Returns
///
/// Vector of cross-chain links above the threshold.
pub fn find_cross_chain_links(
    _chain_embeddings: &[(ChainId, Vec<(NodeId, Vec<f32>)>)],
    _threshold: f32,
) -> Vec<CrossChainLink> {
    // Placeholder implementation
    // In production, this would:
    // 1. Build an index of all embeddings
    // 2. For each embedding, find k-nearest neighbors from other chains
    // 3. Create links for those above threshold
    Vec::new()
}

/// Knowledge transfer patterns to detect.
///
/// These patterns indicate when a user is referencing prior knowledge
/// from another conversation.
pub const KNOWLEDGE_TRANSFER_PATTERNS: &[&str] = &[
    "as we discussed",
    "like we did before",
    "similar to what we",
    "remember when we",
    "building on the",
    "like in the",
    "same as before",
    "just like last time",
    "from our previous",
    "we already covered",
];

/// Detect knowledge transfer patterns in text.
pub fn detect_knowledge_transfer(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    for pattern in KNOWLEDGE_TRANSFER_PATTERNS {
        if lower.contains(pattern) {
            return Some(pattern);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_strength() {
        let strength = LinkStrength::new(0.8, 0.5, 0.6);
        assert!(strength.combined > 0.0);
        assert!(strength.is_strong(0.5));
        assert!(!strength.is_strong(0.9));
    }

    #[test]
    fn test_cross_chain_link() {
        let link = CrossChainLink::semantic(
            "chain1".to_string(),
            1,
            "chain2".to_string(),
            2,
            0.85,
        );

        assert_eq!(link.source_chain, "chain1");
        assert_eq!(link.target_chain, "chain2");
        assert!(link.is_bidirectional());

        let reverse = link.reverse();
        assert_eq!(reverse.source_chain, "chain2");
        assert_eq!(reverse.target_chain, "chain1");
    }

    #[test]
    fn test_knowledge_transfer_detection() {
        assert!(detect_knowledge_transfer("As we discussed before, this should work").is_some());
        assert!(detect_knowledge_transfer("Similar to what we did in the auth project").is_some());
        assert!(detect_knowledge_transfer("This is completely new").is_none());
    }

    #[test]
    fn test_link_with_description() {
        let link = CrossChainLink::semantic(
            "chain1".to_string(),
            1,
            "chain2".to_string(),
            2,
            0.85,
        ).with_description("Both discuss authentication");

        assert_eq!(link.description, Some("Both discuss authentication".to_string()));
    }
}
