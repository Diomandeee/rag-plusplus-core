//! I-RCP (Inverse Ring Contextual Propagation) Module
//!
//! Implements the I-RCP propagation algorithm for computing attention weights
//! in a dual-ring trajectory structure. Based on the DLM package analysis.
//!
//! # Key Concepts
//!
//! - **Forward Attention (A_F)**: Attention from earlier nodes to later nodes (RCP)
//! - **Inverse Attention (A_I)**: Inferred attention from context that produced a response (IRCP)
//! - **Cross Attention (A_C)**: Attention flow between user and assistant turns
//!
//! # Algorithm
//!
//! Attention is computed as:
//!
//! ```text
//! raw_score[i] = spatial_weight(query_coord, context_coord[i]) * semantic_weight(query_emb, context_emb[i])
//! attention = softmax(raw_scores / temperature)
//! ```
//!
//! Where:
//! - `spatial_weight = exp(-distance(query_coord, context_coord))`
//! - `semantic_weight = (1 + cosine_similarity) / 2` (normalized to [0, 1])
//!
//! # Usage
//!
//! ```ignore
//! use rag_plusplus_core::trajectory::ircp::{IRCPPropagator, IRCPConfig, AttentionWeights};
//! use rag_plusplus_core::trajectory::{TrajectoryCoordinate5D, DLMWeights};
//!
//! let config = IRCPConfig::default();
//! let propagator = IRCPPropagator::new(config);
//!
//! let query_coord = TrajectoryCoordinate5D::new(3, 0, 0.9, 0.5, 1);
//! let context_coords = vec![
//!     TrajectoryCoordinate5D::new(1, 0, 0.8, 0.2, 1),
//!     TrajectoryCoordinate5D::new(2, 0, 0.85, 0.4, 2),
//! ];
//! let query_emb = vec![0.5; 768];
//! let context_embs = vec![vec![0.5; 768], vec![0.6; 768]];
//!
//! let weights = propagator.compute_attention(
//!     &query_coord,
//!     &context_coords,
//!     &query_emb,
//!     &context_embs.iter().map(|e| e.as_slice()).collect::<Vec<_>>(),
//! );
//! ```

use crate::distance::cosine_similarity_fast;
use crate::trajectory::{TrajectoryCoordinate5D, DLMWeights};

/// Configuration for I-RCP propagation.
#[derive(Debug, Clone)]
pub struct IRCPConfig {
    /// Temperature for softmax (lower = sharper attention)
    pub temperature: f32,

    /// Weight configuration for coordinate distance
    pub coord_weights: DLMWeights,

    /// Relative weight of spatial vs semantic components [0, 1]
    /// 0 = pure semantic, 1 = pure spatial
    pub spatial_weight: f32,

    /// Whether to use cosine distance (true) or coordinate distance (false) for spatial
    pub use_coordinate_cosine: bool,

    /// Minimum attention weight (prevents division by zero)
    pub min_attention: f32,

    /// Whether to apply causal masking (future nodes get zero attention)
    pub causal_mask: bool,
}

impl Default for IRCPConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            coord_weights: DLMWeights::default(),
            spatial_weight: 0.3, // Favor semantic similarity
            use_coordinate_cosine: false,
            min_attention: 1e-10,
            causal_mask: false,
        }
    }
}

impl IRCPConfig {
    /// Configuration that heavily weights semantic similarity.
    pub fn semantic_focused() -> Self {
        Self {
            spatial_weight: 0.1,
            coord_weights: DLMWeights::semantic_focused(),
            ..Default::default()
        }
    }

    /// Configuration that heavily weights coordinate distance.
    pub fn spatial_focused() -> Self {
        Self {
            spatial_weight: 0.7,
            coord_weights: DLMWeights::structural_focused(),
            ..Default::default()
        }
    }

    /// Configuration for causal attention (no looking at future).
    pub fn causal() -> Self {
        Self {
            causal_mask: true,
            ..Default::default()
        }
    }

    /// Sharp attention (low temperature).
    pub fn sharp() -> Self {
        Self {
            temperature: 0.1,
            ..Default::default()
        }
    }

    /// Diffuse attention (high temperature).
    pub fn diffuse() -> Self {
        Self {
            temperature: 3.0,
            ..Default::default()
        }
    }
}

/// Computed attention weights from I-RCP propagation.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Forward attention weights (A_F): query → context
    pub forward: Vec<f32>,

    /// Inverse attention weights (A_I): context → query (inferred)
    pub inverse: Vec<f32>,

    /// Cross attention weights (A_C): between user/assistant turns
    pub cross: Vec<f32>,

    /// Raw scores before softmax (for debugging)
    pub raw_scores: Vec<f32>,

    /// Total attention mass (should be ~1.0 after softmax)
    pub total_mass: f32,
}

impl AttentionWeights {
    /// Create empty attention weights.
    pub fn empty() -> Self {
        Self {
            forward: Vec::new(),
            inverse: Vec::new(),
            cross: Vec::new(),
            raw_scores: Vec::new(),
            total_mass: 0.0,
        }
    }

    /// Create uniform attention weights.
    pub fn uniform(n: usize) -> Self {
        if n == 0 {
            return Self::empty();
        }

        let weight = 1.0 / n as f32;
        Self {
            forward: vec![weight; n],
            inverse: vec![weight; n],
            cross: vec![weight; n],
            raw_scores: vec![1.0; n],
            total_mass: 1.0,
        }
    }

    /// Get the index with highest forward attention.
    pub fn top_forward(&self) -> Option<usize> {
        self.forward
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Get indices sorted by forward attention (descending).
    pub fn sorted_forward_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.forward.len()).collect();
        indices.sort_by(|&a, &b| {
            self.forward[b]
                .partial_cmp(&self.forward[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get top-k indices by forward attention.
    pub fn top_k_forward(&self, k: usize) -> Vec<usize> {
        self.sorted_forward_indices().into_iter().take(k).collect()
    }

    /// Compute entropy of forward attention distribution.
    pub fn forward_entropy(&self) -> f32 {
        -self
            .forward
            .iter()
            .filter(|&&w| w > 1e-10)
            .map(|w| w * w.ln())
            .sum::<f32>()
    }

    /// Check if attention is concentrated (low entropy).
    pub fn is_concentrated(&self, threshold: f32) -> bool {
        self.forward_entropy() < threshold
    }
}

/// I-RCP propagation engine.
///
/// Computes attention weights for queries over context sets using
/// both spatial (coordinate) and semantic (embedding) similarity.
#[derive(Debug, Clone)]
pub struct IRCPPropagator {
    config: IRCPConfig,
}

impl IRCPPropagator {
    /// Create a new propagator with configuration.
    pub fn new(config: IRCPConfig) -> Self {
        Self { config }
    }

    /// Compute spatial weight between two coordinates.
    ///
    /// Higher weight = closer in coordinate space.
    #[inline]
    fn spatial_weight(&self, query: &TrajectoryCoordinate5D, context: &TrajectoryCoordinate5D) -> f32 {
        if self.config.use_coordinate_cosine {
            // Use cosine similarity of coordinates (direction-based)
            (1.0 + query.cosine_similarity(context)) / 2.0
        } else {
            // Use exponential decay of coordinate distance
            let dist = query.dlm_distance(context, &self.config.coord_weights);
            (-dist).exp()
        }
    }

    /// Compute semantic weight between two embeddings.
    ///
    /// Higher weight = more semantically similar.
    #[inline]
    fn semantic_weight(&self, query_emb: &[f32], context_emb: &[f32]) -> f32 {
        // Normalize cosine similarity from [-1, 1] to [0, 1]
        (1.0 + cosine_similarity_fast(query_emb, context_emb)) / 2.0
    }

    /// Apply softmax to raw scores.
    fn softmax(&self, scores: &[f32]) -> Vec<f32> {
        if scores.is_empty() {
            return Vec::new();
        }

        // Find max for numerical stability
        let max_score = scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Compute exp((x - max) / temperature)
        let exps: Vec<f32> = scores
            .iter()
            .map(|&s| ((s - max_score) / self.config.temperature).exp())
            .collect();

        // Normalize
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            exps.iter().map(|e| (e / sum).max(self.config.min_attention)).collect()
        } else {
            vec![1.0 / scores.len() as f32; scores.len()]
        }
    }

    /// Compute forward attention weights (A_F).
    ///
    /// Forward attention: how much should the query attend to each context item?
    pub fn compute_forward_attention(
        &self,
        query_coord: &TrajectoryCoordinate5D,
        context_coords: &[TrajectoryCoordinate5D],
        query_emb: &[f32],
        context_embs: &[&[f32]],
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            context_coords.len(),
            context_embs.len(),
            "Coordinate and embedding counts must match"
        );

        if context_coords.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let sw = self.config.spatial_weight;
        let raw_scores: Vec<f32> = context_coords
            .iter()
            .zip(context_embs.iter())
            .enumerate()
            .map(|(_i, (coord, emb))| {
                // Apply causal mask if enabled
                if self.config.causal_mask && coord.temporal > query_coord.temporal {
                    return 0.0;
                }

                let spatial = self.spatial_weight(query_coord, coord);
                let semantic = self.semantic_weight(query_emb, emb);

                // Weighted combination
                sw * spatial + (1.0 - sw) * semantic
            })
            .collect();

        let attention = self.softmax(&raw_scores);
        (attention, raw_scores)
    }

    /// Compute inverse attention weights (A_I).
    ///
    /// Inverse attention: given a response, infer what context produced it.
    /// This is the "inverse" of forward attention.
    ///
    /// The inverse is computed by normalizing attention received:
    /// A_I[i] = A_F[i] * influence[i] / sum(A_F * influence)
    pub fn compute_inverse_attention(
        &self,
        forward_attention: &[f32],
        influences: &[f32],
    ) -> Vec<f32> {
        assert_eq!(
            forward_attention.len(),
            influences.len(),
            "Attention and influence counts must match"
        );

        if forward_attention.is_empty() {
            return Vec::new();
        }

        // Weight attention by influence
        let weighted: Vec<f32> = forward_attention
            .iter()
            .zip(influences.iter())
            .map(|(&a, &inf)| a * inf)
            .collect();

        // Normalize
        let sum: f32 = weighted.iter().sum();
        if sum > 0.0 {
            weighted.iter().map(|w| w / sum).collect()
        } else {
            vec![1.0 / weighted.len() as f32; weighted.len()]
        }
    }

    /// Compute cross attention weights (A_C).
    ///
    /// Cross attention captures flow between user and assistant turns.
    /// User turns receive attention from assistant context, and vice versa.
    pub fn compute_cross_attention(
        &self,
        query_coord: &TrajectoryCoordinate5D,
        context_coords: &[TrajectoryCoordinate5D],
        query_emb: &[f32],
        context_embs: &[&[f32]],
        query_is_user: bool,
        context_is_user: &[bool],
    ) -> Vec<f32> {
        assert_eq!(context_coords.len(), context_is_user.len());

        if context_coords.is_empty() {
            return Vec::new();
        }

        let sw = self.config.spatial_weight;

        // Cross attention only applies to opposite roles
        let raw_scores: Vec<f32> = context_coords
            .iter()
            .zip(context_embs.iter())
            .zip(context_is_user.iter())
            .map(|((coord, emb), &is_user)| {
                // Only attend to opposite role
                if is_user == query_is_user {
                    return 0.0;
                }

                // Apply causal mask if enabled
                if self.config.causal_mask && coord.temporal > query_coord.temporal {
                    return 0.0;
                }

                let spatial = self.spatial_weight(query_coord, coord);
                let semantic = self.semantic_weight(query_emb, emb);

                sw * spatial + (1.0 - sw) * semantic
            })
            .collect();

        self.softmax(&raw_scores)
    }

    /// Compute full I-RCP attention weights.
    ///
    /// Returns forward, inverse, and cross attention in one call.
    pub fn compute_attention(
        &self,
        query_coord: &TrajectoryCoordinate5D,
        context_coords: &[TrajectoryCoordinate5D],
        query_emb: &[f32],
        context_embs: &[&[f32]],
    ) -> AttentionWeights {
        let (forward, raw_scores) =
            self.compute_forward_attention(query_coord, context_coords, query_emb, context_embs);

        if forward.is_empty() {
            return AttentionWeights::empty();
        }

        // For inverse, use forward attention as influence proxy
        let inverse = self.compute_inverse_attention(&forward, &forward);

        // For cross, we'd need role information - use forward as placeholder
        let cross = forward.clone();

        let total_mass = forward.iter().sum();

        AttentionWeights {
            forward,
            inverse,
            cross,
            raw_scores,
            total_mass,
        }
    }

    /// Compute attention with role information for cross attention.
    pub fn compute_attention_with_roles(
        &self,
        query_coord: &TrajectoryCoordinate5D,
        context_coords: &[TrajectoryCoordinate5D],
        query_emb: &[f32],
        context_embs: &[&[f32]],
        query_is_user: bool,
        context_is_user: &[bool],
        influences: &[f32],
    ) -> AttentionWeights {
        let (forward, raw_scores) =
            self.compute_forward_attention(query_coord, context_coords, query_emb, context_embs);

        if forward.is_empty() {
            return AttentionWeights::empty();
        }

        let inverse = self.compute_inverse_attention(&forward, influences);

        let cross = self.compute_cross_attention(
            query_coord,
            context_coords,
            query_emb,
            context_embs,
            query_is_user,
            context_is_user,
        );

        let total_mass = forward.iter().sum();

        AttentionWeights {
            forward,
            inverse,
            cross,
            raw_scores,
            total_mass,
        }
    }

    /// Propagate attention through a sequence of queries.
    ///
    /// Returns attention weights for each query position.
    pub fn propagate_sequence(
        &self,
        coords: &[TrajectoryCoordinate5D],
        embeddings: &[&[f32]],
    ) -> Vec<AttentionWeights> {
        assert_eq!(coords.len(), embeddings.len());

        let n = coords.len();
        if n == 0 {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            // Context is all nodes before current position
            let context_coords: Vec<_> = coords[..i].to_vec();
            let context_embs: Vec<_> = embeddings[..i].iter().copied().collect();

            if context_coords.is_empty() {
                results.push(AttentionWeights::empty());
            } else {
                let weights = self.compute_attention(
                    &coords[i],
                    &context_coords,
                    embeddings[i],
                    &context_embs,
                );
                results.push(weights);
            }
        }

        results
    }

    /// Get the config.
    pub fn config(&self) -> &IRCPConfig {
        &self.config
    }

    /// Update config.
    pub fn set_config(&mut self, config: IRCPConfig) {
        self.config = config;
    }
}

impl Default for IRCPPropagator {
    fn default() -> Self {
        Self::new(IRCPConfig::default())
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch compute attention for multiple queries against shared context.
///
/// More efficient than calling compute_attention repeatedly when context is shared.
pub fn batch_compute_attention(
    propagator: &IRCPPropagator,
    query_coords: &[TrajectoryCoordinate5D],
    query_embs: &[&[f32]],
    context_coords: &[TrajectoryCoordinate5D],
    context_embs: &[&[f32]],
) -> Vec<AttentionWeights> {
    query_coords
        .iter()
        .zip(query_embs.iter())
        .map(|(coord, emb)| propagator.compute_attention(coord, context_coords, emb, context_embs))
        .collect()
}

/// Compute attention matrix between all pairs.
///
/// Returns n x n matrix where element [i][j] is attention from i to j.
pub fn compute_attention_matrix(
    propagator: &IRCPPropagator,
    coords: &[TrajectoryCoordinate5D],
    embeddings: &[&[f32]],
) -> Vec<Vec<f32>> {
    let n = coords.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        let weights = propagator.compute_attention(&coords[i], coords, embeddings[i], embeddings);
        matrix[i] = weights.forward;
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_coord(depth: u32, temporal: f32) -> TrajectoryCoordinate5D {
        TrajectoryCoordinate5D::new(depth, 0, 0.8, temporal, 1)
    }

    fn make_test_embedding(seed: f32) -> Vec<f32> {
        (0..8).map(|i| (seed + i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_ircp_config_default() {
        let config = IRCPConfig::default();
        assert!((config.temperature - 1.0).abs() < 1e-6);
        assert!((config.spatial_weight - 0.3).abs() < 1e-6);
        assert!(!config.causal_mask);
    }

    #[test]
    fn test_ircp_config_presets() {
        let semantic = IRCPConfig::semantic_focused();
        assert!(semantic.spatial_weight < 0.2);

        let spatial = IRCPConfig::spatial_focused();
        assert!(spatial.spatial_weight > 0.5);

        let causal = IRCPConfig::causal();
        assert!(causal.causal_mask);

        let sharp = IRCPConfig::sharp();
        assert!(sharp.temperature < 0.5);

        let diffuse = IRCPConfig::diffuse();
        assert!(diffuse.temperature > 2.0);
    }

    #[test]
    fn test_attention_weights_uniform() {
        let weights = AttentionWeights::uniform(5);
        assert_eq!(weights.forward.len(), 5);
        assert!((weights.forward[0] - 0.2).abs() < 1e-6);
        assert!((weights.total_mass - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_weights_empty() {
        let weights = AttentionWeights::empty();
        assert!(weights.forward.is_empty());
        assert!(weights.total_mass < 1e-6);
    }

    #[test]
    fn test_attention_weights_top_k() {
        let weights = AttentionWeights {
            forward: vec![0.1, 0.5, 0.2, 0.15, 0.05],
            inverse: vec![0.2; 5],
            cross: vec![0.2; 5],
            raw_scores: vec![1.0; 5],
            total_mass: 1.0,
        };

        let top1 = weights.top_forward();
        assert_eq!(top1, Some(1)); // Index 1 has 0.5

        let top3 = weights.top_k_forward(3);
        assert_eq!(top3, vec![1, 2, 3]); // 0.5, 0.2, 0.15
    }

    #[test]
    fn test_propagator_empty_context() {
        let propagator = IRCPPropagator::default();
        let query = make_test_coord(3, 0.5);
        let query_emb = make_test_embedding(1.0);

        let weights = propagator.compute_attention(&query, &[], &query_emb, &[]);

        assert!(weights.forward.is_empty());
        assert!(weights.total_mass < 1e-6);
    }

    #[test]
    fn test_propagator_single_context() {
        let propagator = IRCPPropagator::default();
        let query = make_test_coord(3, 0.5);
        let context = vec![make_test_coord(1, 0.2)];
        let query_emb = make_test_embedding(1.0);
        let context_emb = make_test_embedding(1.1);
        let context_embs: Vec<&[f32]> = vec![&context_emb];

        let weights = propagator.compute_attention(&query, &context, &query_emb, &context_embs);

        assert_eq!(weights.forward.len(), 1);
        assert!((weights.forward[0] - 1.0).abs() < 1e-6); // Single element gets all attention
    }

    #[test]
    fn test_propagator_multiple_context() {
        let propagator = IRCPPropagator::default();
        let query = make_test_coord(3, 0.5);
        let context = vec![
            make_test_coord(1, 0.1),
            make_test_coord(2, 0.3),
            make_test_coord(4, 0.6),
        ];
        let query_emb = make_test_embedding(1.0);
        let context_emb1 = make_test_embedding(0.5);
        let context_emb2 = make_test_embedding(0.9); // More similar to query
        let context_emb3 = make_test_embedding(2.0);
        let context_embs: Vec<&[f32]> = vec![&context_emb1, &context_emb2, &context_emb3];

        let weights = propagator.compute_attention(&query, &context, &query_emb, &context_embs);

        assert_eq!(weights.forward.len(), 3);

        // All attention weights should sum to 1.0
        let sum: f32 = weights.forward.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Second context should have higher attention (more similar embedding)
        assert!(weights.forward[1] > weights.forward[0]);
    }

    #[test]
    fn test_propagator_causal_mask() {
        let mut config = IRCPConfig::default();
        config.causal_mask = true;
        let propagator = IRCPPropagator::new(config);

        let query = make_test_coord(2, 0.5); // temporal = 0.5
        let context = vec![
            make_test_coord(1, 0.2), // Before query (should attend)
            make_test_coord(3, 0.8), // After query (should be masked)
        ];
        let query_emb = make_test_embedding(1.0);
        let context_emb1 = make_test_embedding(1.0);
        let context_emb2 = make_test_embedding(1.0);
        let context_embs: Vec<&[f32]> = vec![&context_emb1, &context_emb2];

        let weights = propagator.compute_attention(&query, &context, &query_emb, &context_embs);

        // Future context should have near-zero attention
        assert!(weights.forward[0] > weights.forward[1]);
    }

    #[test]
    fn test_propagator_inverse_attention() {
        let propagator = IRCPPropagator::default();
        let forward = vec![0.2, 0.5, 0.3];
        let influences = vec![1.0, 0.5, 1.5]; // Different influences

        let inverse = propagator.compute_inverse_attention(&forward, &influences);

        assert_eq!(inverse.len(), 3);
        let sum: f32 = inverse.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // High influence should boost attention
        // forward[2] * influence[2] = 0.3 * 1.5 = 0.45
        // forward[1] * influence[1] = 0.5 * 0.5 = 0.25
        // So inverse[2] should be higher than inverse[1]
        assert!(inverse[2] > inverse[1]);
    }

    #[test]
    fn test_propagator_cross_attention() {
        let propagator = IRCPPropagator::default();

        let query = make_test_coord(2, 0.5);
        let query_is_user = true;

        let context = vec![
            make_test_coord(1, 0.2),
            make_test_coord(2, 0.4),
            make_test_coord(3, 0.6),
        ];
        let context_is_user = vec![false, true, false]; // User only attends to assistant

        let query_emb = make_test_embedding(1.0);
        let context_emb1 = make_test_embedding(1.0);
        let context_emb2 = make_test_embedding(1.0);
        let context_emb3 = make_test_embedding(1.0);
        let context_embs: Vec<&[f32]> = vec![&context_emb1, &context_emb2, &context_emb3];

        let cross = propagator.compute_cross_attention(
            &query,
            &context,
            &query_emb,
            &context_embs,
            query_is_user,
            &context_is_user,
        );

        assert_eq!(cross.len(), 3);

        // Same-role context should have zero/minimal attention
        // context[1] is also user, so should be masked
        assert!(cross[0] > cross[1]); // Assistant > User
        assert!(cross[2] > cross[1]); // Assistant > User
    }

    #[test]
    fn test_propagate_sequence() {
        let propagator = IRCPPropagator::default();

        let coords = vec![
            make_test_coord(0, 0.0),
            make_test_coord(1, 0.25),
            make_test_coord(2, 0.5),
            make_test_coord(3, 0.75),
        ];

        let emb0 = make_test_embedding(0.0);
        let emb1 = make_test_embedding(0.5);
        let emb2 = make_test_embedding(1.0);
        let emb3 = make_test_embedding(1.5);
        let embeddings: Vec<&[f32]> = vec![&emb0, &emb1, &emb2, &emb3];

        let results = propagator.propagate_sequence(&coords, &embeddings);

        assert_eq!(results.len(), 4);

        // First position has no context
        assert!(results[0].forward.is_empty());

        // Second position has 1 context
        assert_eq!(results[1].forward.len(), 1);

        // Third position has 2 contexts
        assert_eq!(results[2].forward.len(), 2);

        // Fourth position has 3 contexts
        assert_eq!(results[3].forward.len(), 3);
    }

    #[test]
    fn test_batch_compute_attention() {
        let propagator = IRCPPropagator::default();

        let query_coords = vec![make_test_coord(3, 0.5), make_test_coord(4, 0.7)];

        let context_coords = vec![make_test_coord(1, 0.1), make_test_coord(2, 0.3)];

        let query_emb1 = make_test_embedding(1.0);
        let query_emb2 = make_test_embedding(1.5);
        let query_embs: Vec<&[f32]> = vec![&query_emb1, &query_emb2];

        let context_emb1 = make_test_embedding(0.5);
        let context_emb2 = make_test_embedding(1.0);
        let context_embs: Vec<&[f32]> = vec![&context_emb1, &context_emb2];

        let results = batch_compute_attention(
            &propagator,
            &query_coords,
            &query_embs,
            &context_coords,
            &context_embs,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].forward.len(), 2);
        assert_eq!(results[1].forward.len(), 2);
    }

    #[test]
    fn test_compute_attention_matrix() {
        let propagator = IRCPPropagator::default();

        let coords = vec![
            make_test_coord(0, 0.0),
            make_test_coord(1, 0.5),
            make_test_coord(2, 1.0),
        ];

        let emb0 = make_test_embedding(0.0);
        let emb1 = make_test_embedding(0.5);
        let emb2 = make_test_embedding(1.0);
        let embeddings: Vec<&[f32]> = vec![&emb0, &emb1, &emb2];

        let matrix = compute_attention_matrix(&propagator, &coords, &embeddings);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Each row should sum to 1.0
        for row in &matrix {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_entropy() {
        // Uniform attention has high entropy
        let uniform = AttentionWeights::uniform(4);
        let uniform_entropy = uniform.forward_entropy();

        // Concentrated attention has low entropy
        let concentrated = AttentionWeights {
            forward: vec![0.97, 0.01, 0.01, 0.01],
            inverse: vec![0.25; 4],
            cross: vec![0.25; 4],
            raw_scores: vec![1.0; 4],
            total_mass: 1.0,
        };
        let concentrated_entropy = concentrated.forward_entropy();

        assert!(uniform_entropy > concentrated_entropy);
        assert!(concentrated.is_concentrated(0.5));
        assert!(!uniform.is_concentrated(0.5));
    }
}
