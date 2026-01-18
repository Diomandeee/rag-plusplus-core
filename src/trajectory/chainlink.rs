//! ChainLink Interaction Framework
//!
//! Implements the ChainLink estimator from the DLM package for computing
//! relationship strength between elements in a response chain.
//!
//! # Overview
//!
//! ChainLink represents a link in a response chain with associated metadata:
//! - Coordinates (position in trajectory)
//! - Embeddings (semantic representation)
//! - Link type (continuation, elaboration, contradiction, etc.)
//! - Attention weights (forward, inverse, cross)
//!
//! # 4-Component Estimation
//!
//! The ChainLink estimator uses 4 components:
//!
//! | Component | Weight | Description |
//! |-----------|--------|-------------|
//! | Baseline | 0.20 | Base semantic similarity |
//! | Relationship | 0.30 | Relationship strength (coordinate + attention) |
//! | Type-based | 0.20 | Weight based on link type compatibility |
//! | Context-weighted | 0.30 | Context-aware weighted similarity |
//!
//! # Usage
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::chainlink::{ChainLink, ChainLinkEstimator};
//!
//! let link_a = ChainLink::new(coord_a, embedding_a, LinkType::Continuation);
//! let link_b = ChainLink::new(coord_b, embedding_b, LinkType::Elaboration);
//!
//! let estimator = ChainLinkEstimator::default();
//! let strength = estimator.estimate(&link_a, &link_b);
//! ```

use crate::distance::cosine_similarity_fast;
use crate::trajectory::{TrajectoryCoordinate5D, DLMWeights, AttentionWeights};

/// Types of links in a response chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinkType {
    /// Direct continuation of previous content
    Continuation,
    /// Elaboration or expansion of previous content
    Elaboration,
    /// Summarization or compression
    Summary,
    /// Contradiction or correction
    Contradiction,
    /// New topic or tangent
    Tangent,
    /// Question about previous content
    Question,
    /// Answer to a previous question
    Answer,
    /// Code implementation
    Code,
    /// Error or problem statement
    Error,
    /// Solution or fix
    Solution,
    /// Meta-commentary or reflection
    Meta,
    /// Unknown or unclassified
    Unknown,
}

impl LinkType {
    /// Get numeric value for compatibility matrix lookup.
    pub fn as_index(&self) -> usize {
        match self {
            Self::Continuation => 0,
            Self::Elaboration => 1,
            Self::Summary => 2,
            Self::Contradiction => 3,
            Self::Tangent => 4,
            Self::Question => 5,
            Self::Answer => 6,
            Self::Code => 7,
            Self::Error => 8,
            Self::Solution => 9,
            Self::Meta => 10,
            Self::Unknown => 11,
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "continuation" | "continue" => Self::Continuation,
            "elaboration" | "elaborate" | "expand" => Self::Elaboration,
            "summary" | "summarize" | "compress" => Self::Summary,
            "contradiction" | "contradict" | "correct" => Self::Contradiction,
            "tangent" | "aside" | "digression" => Self::Tangent,
            "question" | "ask" | "query" => Self::Question,
            "answer" | "reply" | "respond" => Self::Answer,
            "code" | "implementation" | "impl" => Self::Code,
            "error" | "problem" | "issue" | "bug" => Self::Error,
            "solution" | "fix" | "resolve" => Self::Solution,
            "meta" | "reflection" | "comment" => Self::Meta,
            _ => Self::Unknown,
        }
    }

    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Continuation => "continuation",
            Self::Elaboration => "elaboration",
            Self::Summary => "summary",
            Self::Contradiction => "contradiction",
            Self::Tangent => "tangent",
            Self::Question => "question",
            Self::Answer => "answer",
            Self::Code => "code",
            Self::Error => "error",
            Self::Solution => "solution",
            Self::Meta => "meta",
            Self::Unknown => "unknown",
        }
    }

    /// Check if this is a "constructive" link type.
    pub fn is_constructive(&self) -> bool {
        matches!(
            self,
            Self::Continuation | Self::Elaboration | Self::Answer | Self::Solution
        )
    }

    /// Check if this is a "questioning" link type.
    pub fn is_questioning(&self) -> bool {
        matches!(self, Self::Question | Self::Error | Self::Contradiction)
    }
}

impl Default for LinkType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// A link in a response chain with coordinates, embedding, and metadata.
#[derive(Debug, Clone)]
pub struct ChainLink {
    /// Position in trajectory
    pub coordinate: TrajectoryCoordinate5D,

    /// Semantic embedding
    pub embedding: Vec<f32>,

    /// Type of link
    pub link_type: LinkType,

    /// Influence weight (0-1)
    pub influence: f32,

    /// Whether this is a user message
    pub is_user: bool,

    /// Attention weights (if computed)
    pub attention: Option<AttentionWeights>,

    /// Optional identifier
    pub id: Option<String>,
}

impl ChainLink {
    /// Create a new chain link.
    pub fn new(
        coordinate: TrajectoryCoordinate5D,
        embedding: Vec<f32>,
        link_type: LinkType,
    ) -> Self {
        Self {
            coordinate,
            embedding,
            link_type,
            influence: 0.5,
            is_user: false,
            attention: None,
            id: None,
        }
    }

    /// Create with full metadata.
    pub fn with_metadata(
        coordinate: TrajectoryCoordinate5D,
        embedding: Vec<f32>,
        link_type: LinkType,
        influence: f32,
        is_user: bool,
    ) -> Self {
        Self {
            coordinate,
            embedding,
            link_type,
            influence: influence.clamp(0.0, 1.0),
            is_user,
            attention: None,
            id: None,
        }
    }

    /// Set attention weights.
    pub fn with_attention(mut self, attention: AttentionWeights) -> Self {
        self.attention = Some(attention);
        self
    }

    /// Set identifier.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Compute semantic similarity to another link.
    pub fn semantic_similarity(&self, other: &Self) -> f32 {
        cosine_similarity_fast(&self.embedding, &other.embedding)
    }

    /// Compute coordinate distance to another link.
    pub fn coordinate_distance(&self, other: &Self, weights: &DLMWeights) -> f32 {
        self.coordinate.dlm_distance(&other.coordinate, weights)
    }
}

/// Configuration for the ChainLink estimator.
#[derive(Debug, Clone)]
pub struct ChainLinkEstimatorConfig {
    /// Weight for baseline similarity component
    pub baseline_weight: f32,

    /// Weight for relationship strength component
    pub relationship_weight: f32,

    /// Weight for type-based component
    pub type_weight: f32,

    /// Weight for context-weighted component
    pub context_weight: f32,

    /// DLM weights for coordinate distance
    pub coord_weights: DLMWeights,

    /// Type compatibility matrix (12x12)
    /// Values represent compatibility between link types [0, 1]
    pub type_compatibility: [[f32; 12]; 12],
}

impl Default for ChainLinkEstimatorConfig {
    fn default() -> Self {
        Self {
            baseline_weight: 0.20,
            relationship_weight: 0.30,
            type_weight: 0.20,
            context_weight: 0.30,
            coord_weights: DLMWeights::default(),
            type_compatibility: Self::default_compatibility_matrix(),
        }
    }
}

impl ChainLinkEstimatorConfig {
    /// Create the default type compatibility matrix.
    ///
    /// Higher values mean the types work well together.
    fn default_compatibility_matrix() -> [[f32; 12]; 12] {
        // Types: Continuation, Elaboration, Summary, Contradiction, Tangent,
        //        Question, Answer, Code, Error, Solution, Meta, Unknown
        let mut matrix = [[0.5; 12]; 12];

        // Self-compatibility (same type)
        for i in 0..12 {
            matrix[i][i] = 0.8;
        }

        // High compatibility pairs
        let high_compat = [
            (0, 1), // Continuation <-> Elaboration
            (1, 0),
            (0, 2), // Continuation <-> Summary
            (2, 0),
            (5, 6), // Question <-> Answer
            (6, 5),
            (7, 8), // Code <-> Error
            (8, 7),
            (8, 9), // Error <-> Solution
            (9, 8),
            (7, 9), // Code <-> Solution
            (9, 7),
        ];
        for (i, j) in high_compat {
            matrix[i][j] = 0.9;
        }

        // Medium compatibility
        let medium_compat = [
            (1, 2), // Elaboration <-> Summary
            (2, 1),
            (3, 9), // Contradiction <-> Solution
            (9, 3),
            (4, 5), // Tangent <-> Question
            (5, 4),
        ];
        for (i, j) in medium_compat {
            matrix[i][j] = 0.7;
        }

        // Low compatibility
        let low_compat = [
            (0, 4), // Continuation <-> Tangent
            (4, 0),
            (3, 0), // Contradiction <-> Continuation
            (0, 3),
        ];
        for (i, j) in low_compat {
            matrix[i][j] = 0.3;
        }

        matrix
    }

    /// Get type compatibility score.
    pub fn get_type_compatibility(&self, type_a: LinkType, type_b: LinkType) -> f32 {
        self.type_compatibility[type_a.as_index()][type_b.as_index()]
    }

    /// Create config with custom component weights.
    pub fn with_weights(
        baseline: f32,
        relationship: f32,
        type_weight: f32,
        context: f32,
    ) -> Self {
        // Normalize weights
        let total = baseline + relationship + type_weight + context;
        Self {
            baseline_weight: baseline / total,
            relationship_weight: relationship / total,
            type_weight: type_weight / total,
            context_weight: context / total,
            ..Default::default()
        }
    }
}

/// ChainLink estimator for computing relationship strength.
#[derive(Debug, Clone)]
pub struct ChainLinkEstimator {
    config: ChainLinkEstimatorConfig,
}

impl ChainLinkEstimator {
    /// Create a new estimator with configuration.
    pub fn new(config: ChainLinkEstimatorConfig) -> Self {
        Self { config }
    }

    /// Compute baseline similarity component.
    ///
    /// Pure semantic similarity between embeddings.
    #[inline]
    fn compute_baseline(&self, link_a: &ChainLink, link_b: &ChainLink) -> f32 {
        // Normalize from [-1, 1] to [0, 1]
        (1.0 + link_a.semantic_similarity(link_b)) / 2.0
    }

    /// Compute relationship strength component.
    ///
    /// Combines coordinate distance and influence weights.
    #[inline]
    fn compute_relationship(&self, link_a: &ChainLink, link_b: &ChainLink) -> f32 {
        // Coordinate-based proximity
        let coord_dist = link_a.coordinate_distance(link_b, &self.config.coord_weights);
        let coord_sim = (-coord_dist).exp(); // [0, 1]

        // Influence combination
        let influence_avg = (link_a.influence + link_b.influence) / 2.0;

        // Attention-based (if available)
        let attention_score = match (&link_a.attention, &link_b.attention) {
            (Some(att_a), Some(att_b)) => {
                // Cross-attention if different roles
                if link_a.is_user != link_b.is_user {
                    (att_a.total_mass + att_b.total_mass) / 2.0
                } else {
                    // Forward attention otherwise
                    let entropy_a = att_a.forward_entropy();
                    let entropy_b = att_b.forward_entropy();
                    // Lower entropy = more focused = higher score
                    1.0 / (1.0 + (entropy_a + entropy_b) / 2.0)
                }
            }
            _ => 0.5, // Neutral if no attention
        };

        // Weighted combination
        0.4 * coord_sim + 0.3 * influence_avg + 0.3 * attention_score
    }

    /// Compute type-based weight component.
    ///
    /// Uses the compatibility matrix for link types.
    #[inline]
    fn compute_type_based(&self, link_a: &ChainLink, link_b: &ChainLink) -> f32 {
        self.config
            .get_type_compatibility(link_a.link_type, link_b.link_type)
    }

    /// Compute context-weighted component.
    ///
    /// Combines semantic similarity with positional context.
    #[inline]
    fn compute_context_weighted(&self, link_a: &ChainLink, link_b: &ChainLink) -> f32 {
        let semantic = (1.0 + link_a.semantic_similarity(link_b)) / 2.0;

        // Temporal proximity bonus
        let temporal_diff = (link_a.coordinate.temporal - link_b.coordinate.temporal).abs();
        let temporal_bonus = 1.0 - temporal_diff; // Higher when close in time

        // Homogeneity alignment
        let homo_diff = (link_a.coordinate.homogeneity - link_b.coordinate.homogeneity).abs();
        let homo_bonus = 1.0 - homo_diff;

        // Role alignment (same role = slightly higher)
        let role_bonus = if link_a.is_user == link_b.is_user {
            0.1
        } else {
            0.0
        };

        // Weighted combination
        0.5 * semantic + 0.2 * temporal_bonus + 0.2 * homo_bonus + 0.1 * (0.5 + role_bonus)
    }

    /// Estimate relationship strength between two links.
    ///
    /// Returns a value in [0, 1] where higher = stronger relationship.
    pub fn estimate(&self, link_a: &ChainLink, link_b: &ChainLink) -> f32 {
        let baseline = self.compute_baseline(link_a, link_b);
        let relationship = self.compute_relationship(link_a, link_b);
        let type_based = self.compute_type_based(link_a, link_b);
        let context = self.compute_context_weighted(link_a, link_b);

        self.config.baseline_weight * baseline
            + self.config.relationship_weight * relationship
            + self.config.type_weight * type_based
            + self.config.context_weight * context
    }

    /// Estimate relationship with detailed breakdown.
    pub fn estimate_detailed(&self, link_a: &ChainLink, link_b: &ChainLink) -> ChainLinkEstimate {
        let baseline = self.compute_baseline(link_a, link_b);
        let relationship = self.compute_relationship(link_a, link_b);
        let type_based = self.compute_type_based(link_a, link_b);
        let context = self.compute_context_weighted(link_a, link_b);

        let total = self.config.baseline_weight * baseline
            + self.config.relationship_weight * relationship
            + self.config.type_weight * type_based
            + self.config.context_weight * context;

        ChainLinkEstimate {
            total,
            baseline,
            relationship,
            type_based,
            context,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &ChainLinkEstimatorConfig {
        &self.config
    }
}

impl Default for ChainLinkEstimator {
    fn default() -> Self {
        Self::new(ChainLinkEstimatorConfig::default())
    }
}

/// Detailed estimate breakdown.
#[derive(Debug, Clone, Copy)]
pub struct ChainLinkEstimate {
    /// Total estimated relationship strength [0, 1]
    pub total: f32,
    /// Baseline semantic similarity component
    pub baseline: f32,
    /// Relationship strength component
    pub relationship: f32,
    /// Type-based compatibility component
    pub type_based: f32,
    /// Context-weighted component
    pub context: f32,
}

impl ChainLinkEstimate {
    /// Check if the estimate indicates a strong relationship.
    pub fn is_strong(&self) -> bool {
        self.total > 0.7
    }

    /// Check if the estimate indicates a weak relationship.
    pub fn is_weak(&self) -> bool {
        self.total < 0.3
    }

    /// Get the dominant component.
    pub fn dominant_component(&self) -> &'static str {
        let components = [
            (self.baseline, "baseline"),
            (self.relationship, "relationship"),
            (self.type_based, "type_based"),
            (self.context, "context"),
        ];

        components
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("unknown")
    }
}

/// Compute pairwise relationship matrix for a chain of links.
pub fn compute_chain_matrix(
    estimator: &ChainLinkEstimator,
    links: &[ChainLink],
) -> Vec<Vec<f32>> {
    let n = links.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[i][j] = 1.0; // Self-relationship
            } else {
                matrix[i][j] = estimator.estimate(&links[i], &links[j]);
            }
        }
    }

    matrix
}

/// Find the strongest link for each position.
pub fn find_strongest_links(
    estimator: &ChainLinkEstimator,
    links: &[ChainLink],
) -> Vec<Option<(usize, f32)>> {
    let n = links.len();
    let mut strongest = vec![None; n];

    for i in 0..n {
        let mut best: Option<(usize, f32)> = None;

        for j in 0..n {
            if i == j {
                continue;
            }

            let strength = estimator.estimate(&links[i], &links[j]);

            match best {
                Some((_, best_strength)) if strength > best_strength => {
                    best = Some((j, strength));
                }
                None => {
                    best = Some((j, strength));
                }
                _ => {}
            }
        }

        strongest[i] = best;
    }

    strongest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_link(
        depth: u32,
        temporal: f32,
        embedding_seed: f32,
        link_type: LinkType,
    ) -> ChainLink {
        let coord = TrajectoryCoordinate5D::new(depth, 0, 0.8, temporal, 1);
        let embedding: Vec<f32> = (0..8)
            .map(|i| (embedding_seed + i as f32 * 0.1).sin())
            .collect();

        ChainLink::new(coord, embedding, link_type)
    }

    #[test]
    fn test_link_type_parsing() {
        assert_eq!(LinkType::from_str("continuation"), LinkType::Continuation);
        assert_eq!(LinkType::from_str("QUESTION"), LinkType::Question);
        assert_eq!(LinkType::from_str("code"), LinkType::Code);
        assert_eq!(LinkType::from_str("unknown_type"), LinkType::Unknown);
    }

    #[test]
    fn test_link_type_properties() {
        assert!(LinkType::Continuation.is_constructive());
        assert!(LinkType::Answer.is_constructive());
        assert!(!LinkType::Question.is_constructive());

        assert!(LinkType::Question.is_questioning());
        assert!(LinkType::Error.is_questioning());
        assert!(!LinkType::Answer.is_questioning());
    }

    #[test]
    fn test_chain_link_creation() {
        let coord = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.5, 1);
        let embedding = vec![0.5; 8];
        let link = ChainLink::new(coord, embedding, LinkType::Continuation);

        assert!((link.influence - 0.5).abs() < 1e-6);
        assert!(!link.is_user);
        assert!(link.attention.is_none());
    }

    #[test]
    fn test_chain_link_similarity() {
        let link_a = make_test_link(1, 0.2, 1.0, LinkType::Continuation);
        let link_b = make_test_link(2, 0.4, 1.1, LinkType::Elaboration);

        let sim = link_a.semantic_similarity(&link_b);
        assert!(sim > 0.9); // Similar embeddings

        let link_c = make_test_link(3, 0.6, 5.0, LinkType::Tangent);
        let sim_c = link_a.semantic_similarity(&link_c);
        assert!(sim_c < sim); // Less similar
    }

    #[test]
    fn test_estimator_config_default() {
        let config = ChainLinkEstimatorConfig::default();

        // Weights should sum to 1.0
        let total = config.baseline_weight
            + config.relationship_weight
            + config.type_weight
            + config.context_weight;
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_type_compatibility() {
        let config = ChainLinkEstimatorConfig::default();

        // Question-Answer should be highly compatible
        let qa_compat = config.get_type_compatibility(LinkType::Question, LinkType::Answer);
        assert!(qa_compat > 0.8);

        // Error-Solution should be highly compatible
        let es_compat = config.get_type_compatibility(LinkType::Error, LinkType::Solution);
        assert!(es_compat > 0.8);

        // Continuation-Tangent should be low compatibility
        let ct_compat = config.get_type_compatibility(LinkType::Continuation, LinkType::Tangent);
        assert!(ct_compat < 0.5);
    }

    #[test]
    fn test_estimator_estimate() {
        let estimator = ChainLinkEstimator::default();

        let link_a = make_test_link(1, 0.2, 1.0, LinkType::Question);
        let link_b = make_test_link(2, 0.4, 1.1, LinkType::Answer);

        let strength = estimator.estimate(&link_a, &link_b);

        // Should be positive and reasonable
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }

    #[test]
    fn test_estimator_self_similarity() {
        let estimator = ChainLinkEstimator::default();

        let link = make_test_link(2, 0.5, 1.0, LinkType::Continuation);
        let strength = estimator.estimate(&link, &link);

        // Self-similarity should be very high
        assert!(strength > 0.8);
    }

    #[test]
    fn test_estimator_detailed() {
        let estimator = ChainLinkEstimator::default();

        let link_a = make_test_link(1, 0.2, 1.0, LinkType::Error);
        let link_b = make_test_link(2, 0.4, 1.2, LinkType::Solution);

        let estimate = estimator.estimate_detailed(&link_a, &link_b);

        assert!(estimate.total > 0.0);
        assert!(estimate.baseline > 0.0);
        assert!(estimate.relationship > 0.0);
        assert!(estimate.type_based > 0.0);
        assert!(estimate.context > 0.0);

        // Type-based should be high for Error-Solution
        assert!(estimate.type_based > 0.8);
    }

    #[test]
    fn test_estimate_dominant_component() {
        let estimate = ChainLinkEstimate {
            total: 0.75,
            baseline: 0.9,
            relationship: 0.6,
            type_based: 0.7,
            context: 0.5,
        };

        assert_eq!(estimate.dominant_component(), "baseline");
        assert!(estimate.is_strong());
        assert!(!estimate.is_weak());
    }

    #[test]
    fn test_compute_chain_matrix() {
        let estimator = ChainLinkEstimator::default();

        let links = vec![
            make_test_link(1, 0.1, 1.0, LinkType::Question),
            make_test_link(2, 0.3, 1.2, LinkType::Answer),
            make_test_link(3, 0.5, 1.4, LinkType::Elaboration),
        ];

        let matrix = compute_chain_matrix(&estimator, &links);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be 1.0 (self-relationship)
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);
        assert!((matrix[2][2] - 1.0).abs() < 1e-6);

        // Matrix should be symmetric for same links
        // (not guaranteed for all cases due to role differences)
    }

    #[test]
    fn test_find_strongest_links() {
        let estimator = ChainLinkEstimator::default();

        let links = vec![
            make_test_link(1, 0.1, 1.0, LinkType::Question),
            make_test_link(2, 0.3, 1.1, LinkType::Answer), // Close to Q
            make_test_link(3, 0.8, 5.0, LinkType::Tangent), // Far from others
        ];

        let strongest = find_strongest_links(&estimator, &links);

        assert_eq!(strongest.len(), 3);

        // Question's strongest should be Answer (index 1)
        assert!(strongest[0].is_some());

        // All should have a strongest link
        for (i, s) in strongest.iter().enumerate() {
            assert!(s.is_some(), "Link {} should have a strongest connection", i);
        }
    }

    #[test]
    fn test_chain_link_with_metadata() {
        let coord = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.5, 1);
        let embedding = vec![0.5; 8];
        let link = ChainLink::with_metadata(
            coord,
            embedding,
            LinkType::Continuation,
            0.8,
            true, // is_user
        );

        assert!((link.influence - 0.8).abs() < 1e-6);
        assert!(link.is_user);
    }

    #[test]
    fn test_chain_link_fluent_api() {
        let coord = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.5, 1);
        let embedding = vec![0.5; 8];
        let attention = AttentionWeights::uniform(3);

        let link = ChainLink::new(coord, embedding, LinkType::Continuation)
            .with_attention(attention)
            .with_id("test_link_1");

        assert!(link.attention.is_some());
        assert_eq!(link.id, Some("test_link_1".to_string()));
    }
}
