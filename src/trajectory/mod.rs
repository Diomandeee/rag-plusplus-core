//! Trajectory Memory Module
//!
//! High-performance algorithms for trajectory-structured memory, implementing
//! concepts from IRCP (Inverse Ring Contextual Propagation) and RCP (Ring
//! Contextual Propagation).
//!
//! # Components
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`graph`] | DAG traversal, path finding, branch detection |
//! | [`phase`] | Phase inference from episode patterns |
//! | [`salience`] | Importance weighting for bounded forgetting |
//! | [`ring`] | Circular topology for multi-scale context (includes DualRing) |
//! | [`coordinate`] | 5D positioning system (DLM coordinates with complexity) |
//! | [`coordinate_tree`] | Tree operations with O(log n) LCA via binary lifting |
//! | [`conservation`] | Conservation metrics for forgetting validation |
//! | [`ircp`] | Inverse Ring Contextual Propagation attention |
//! | [`chainlink`] | ChainLink interaction framework with 4-component estimation |
//! | [`branch`] | Branch state machine for split/merge/recover operations |
//! | [`chain`] | Multi-chain management for conversation tracking |
//!
//! # Conceptual Framework
//!
//! - **Trajectory**: A sequence of experiences (episodes) forming a path through time
//! - **Episode**: A unit of experience (message turn, interaction, decision point)
//! - **Phase**: A qualitative state within a trajectory:
//!   - `Exploration`: Initial inquiry, questions, topic discovery
//!   - `Consolidation`: Building understanding, code implementation
//!   - `Synthesis`: Summarization, decisions, conclusions
//!   - `Debugging`: Error resolution, troubleshooting
//!   - `Planning`: Roadmaps, structured plans
//! - **Salience**: Importance weight [0, 1] for bounded forgetting
//! - **Conservation**: Metrics ensuring memory operations preserve invariants
//!
//! # Coordinate System (5D DLM)
//!
//! Each episode has a 5D position (extending TPO with complexity):
//!
//! | Dimension | Type | Meaning |
//! |-----------|------|---------|
//! | `depth` | u32 | Distance from root (0 = root) |
//! | `sibling_order` | u32 | Position among siblings |
//! | `homogeneity` | f32 | Semantic similarity to parent [0, 1] |
//! | `temporal` | f32 | Normalized timestamp [0, 1] |
//! | `complexity` | u32 | Message complexity (n_parts from DLM) |
//!
//! # Dual Ring Structure (IRCP/RCP)
//!
//! The `DualRing` provides two simultaneous orderings over the same episodes:
//!
//! | Ring | Direction | Ordering | Use Case |
//! |------|-----------|----------|----------|
//! | **Temporal** (RCP) | Forward | Time/causal sequence | "What context led here?" |
//! | **Influence** (IRCP) | Inverse | By influence weight | "What had most impact?" |
//!
//! ```text
//! Temporal Ring (RCP):     [E₀] → [E₁] → [E₂] → [E₃] → [E₄] ─┐
//!                            ↑                                 │
//!                            └─────────────────────────────────┘
//!
//! Influence Ring (IRCP):   [E₂] → [E₄] → [E₀] → [E₃] → [E₁] ─┐
//!                            ↑     (sorted by influence)      │
//!                            └────────────────────────────────┘
//! ```
//!
//! # Conservation Laws
//!
//! Per RCP, memory operations should preserve:
//!
//! | Law | Formula | Meaning |
//! |-----|---------|---------|
//! | Magnitude | Σ aᵢ‖eᵢ‖ = const | Context doesn't disappear |
//! | Energy | ½Σᵢⱼ aᵢaⱼ cos(eᵢ,eⱼ) | Attention capacity conserved |
//! | Information | -Σ aᵢ log(aᵢ) | Shannon entropy of attention |
//!
//! # Performance
//!
//! These algorithms are implemented in Rust for performance because they run
//! repeatedly during retrieval and corpus maintenance. Key optimizations:
//!
//! - SIMD-accelerated distance computations (via `distance` module)
//! - Cache-friendly contiguous storage in Ring topology
//! - O(1) neighbor access in ring structure
//! - Efficient hash-based DAG traversal
//!
//! # Example Usage
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::{
//!     TrajectoryGraph, Edge, EdgeType, PathSelectionPolicy,
//!     PhaseInferencer, TurnFeatures, TrajectoryPhase,
//!     SalienceScorer, SalienceFactors, Feedback,
//!     Ring, TrajectoryCoordinate,
//!     ConservationMetrics,
//! };
//!
//! // Build trajectory graph from edges
//! let edges = vec![
//!     Edge { parent: 1, child: 2, edge_type: EdgeType::Continuation },
//!     Edge { parent: 2, child: 3, edge_type: EdgeType::Continuation },
//! ];
//! let graph = TrajectoryGraph::from_edges(edges.iter().copied());
//!
//! // Find primary path through DAG
//! let primary = graph.find_primary_path(PathSelectionPolicy::FeedbackFirst);
//!
//! // Infer phases from episode features
//! let inferencer = PhaseInferencer::new();
//! let features = TurnFeatures::from_content(1, "user", "What is this?");
//! let (phase, confidence) = inferencer.infer_single(&features).unwrap();
//!
//! // Compute salience scores
//! let scorer = SalienceScorer::new();
//! let factors = SalienceFactors {
//!     turn_id: 1,
//!     feedback: Some(Feedback::ThumbsUp),
//!     ..Default::default()
//! };
//! let salience = scorer.score_single(&factors, None);
//!
//! // Build ring structure
//! let ring = Ring::new(vec![1, 2, 3, 4, 5]);
//! let distance = ring.ring_distance(0, 3); // Shortest path: 2
//!
//! // Validate conservation
//! let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
//! let refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
//! let attention = vec![0.5, 0.5];
//! let metrics = ConservationMetrics::compute(&refs, &attention);
//! ```

pub mod branch;
pub mod chain;
pub mod chainlink;
pub mod conservation;
pub mod coordinate;
pub mod coordinate_tree;
pub mod graph;
pub mod ircp;
pub mod path_quality;
pub mod phase;
pub mod ring;
pub mod salience;

// Re-exports
pub use conservation::{
    ConservationMetrics, ConservationViolation, ConservationConfig,
    ConservationTracker, weighted_centroid, weighted_covariance,
};
pub use coordinate::{
    TrajectoryCoordinate, NormalizedCoordinate,
    // 5D coordinates (TPO extension with complexity dimension)
    TrajectoryCoordinate5D, NormalizedCoordinate5D,
    // DLM weight configuration for distance calculations
    DLMWeights,
    compute_trajectory_coordinates,
};
pub use graph::{
    TrajectoryGraph, Episode, Edge, EdgeType, NodeId,
    PathSelectionPolicy, TraversalOrder, BranchInfo, PathResult,
};
pub use phase::{
    TrajectoryPhase, PhaseInferencer, PhaseConfig,
    TurnFeatures, PhaseTransition,
};

// Backward compatibility - ConversationPhase is deprecated in favor of TrajectoryPhase
#[allow(deprecated)]
pub use phase::ConversationPhase;
pub use ring::{
    Ring, RingNode, EpisodeRing, build_weighted_ring,
    // Dual ring structure for IRCP/RCP
    DualRing, DualRingNode, build_dual_ring, build_dual_ring_with_attention,
};
pub use path_quality::{
    PathQuality, PathQualityWeights, PathQualityFactors,
};
pub use salience::{
    SalienceScorer, SalienceConfig, SalienceFactors,
    TurnSalience, CorpusSalienceStats, Feedback,
};
pub use ircp::{
    IRCPPropagator, IRCPConfig, AttentionWeights,
    batch_compute_attention, compute_attention_matrix,
};
pub use chainlink::{
    ChainLink, ChainLinkEstimator, ChainLinkEstimatorConfig,
    ChainLinkEstimate, LinkType,
    compute_chain_matrix, find_strongest_links,
};
pub use coordinate_tree::{
    CoordinateTree, TreeNode, build_coordinate_tree,
};
pub use branch::{
    // Branch state machine for split/merge/recover operations
    BranchStateMachine, BranchContext, SplitResult, MergeResult,
    // Core branch types
    Branch, BranchId, BranchOperation, BranchStatus, ForkPoint, BranchError,
    // Branch resolution for "lost branch" recovery
    BranchResolver, RecoverableBranch, RecoveryStrategy, LostReason,
};
pub use chain::{
    // Chain manager for multi-conversation tracking
    ChainManager, ChainId, ChainMetadata, ChainManagerConfig,
    ChainManagerError, ChainManagerStats,
    // Cross-chain links
    CrossChainLink, CrossChainLinkType, LinkStrength,
    find_cross_chain_links, detect_knowledge_transfer,
    KNOWLEDGE_TRANSFER_PATTERNS,
};
