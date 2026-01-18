//! Integration tests for rag_plusplus_core crate
//!
//! Tests HNSW index, IRCP propagation, trajectory coordinates, and salience scoring.

use rag_plusplus_core::{
    // Index types
    HNSWConfig, HNSWIndex, VectorIndex,
    // Trajectory types
    TrajectoryCoordinate5D,
    SalienceScorer, SalienceFactors, Feedback, TurnSalience,
    // Cache
    QueryCache, CacheConfig,
};
use rag_plusplus_core::trajectory::{
    DLMWeights,
    IRCPConfig, IRCPPropagator, AttentionWeights,
};

// === HNSW Index Tests ===

#[test]
fn test_hnsw_index_creation() {
    let config = HNSWConfig::new(128);
    let index = HNSWIndex::new(config);
    
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
}

#[test]
fn test_hnsw_config_new() {
    let config = HNSWConfig::new(64);
    assert_eq!(config.base.dimension, 64);
    assert!(config.m > 0);
    assert!(config.ef_construction > 0);
}

#[test]
fn test_hnsw_config_builder() {
    let config = HNSWConfig::new(64)
        .with_m(32)
        .with_ef_construction(100)
        .with_ef_search(50);
    
    assert_eq!(config.m, 32);
    assert_eq!(config.ef_construction, 100);
    assert_eq!(config.ef_search, 50);
}

#[test]
fn test_hnsw_index_add_single() {
    let config = HNSWConfig::new(64);
    let mut index = HNSWIndex::new(config);
    
    let embedding: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    
    index.add("test_id_1".to_string(), &embedding).expect("add vector");
    
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn test_hnsw_index_add_multiple() {
    let config = HNSWConfig::new(32);
    let mut index = HNSWIndex::new(config);
    
    for i in 0..100 {
        let embedding: Vec<f32> = (0..32).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
        let id = format!("id_{}", i);
        index.add(id, &embedding).expect("add vector");
    }
    
    assert_eq!(index.len(), 100);
}

#[test]
fn test_hnsw_index_search() {
    let config = HNSWConfig::new(16);
    let mut index = HNSWIndex::new(config);
    
    // Add some vectors
    for i in 0..10 {
        let embedding: Vec<f32> = (0..16).map(|j| ((i * 10 + j) as f32 * 0.1).sin()).collect();
        let id = format!("doc_{}", i);
        index.add(id, &embedding).expect("add vector");
    }
    
    // Search for similar
    let query: Vec<f32> = (0..16).map(|j| (j as f32 * 0.1).sin()).collect(); // Similar to doc_0
    let results = index.search(&query, 3).expect("search");
    
    assert!(!results.is_empty());
    assert!(results.len() <= 3);
    
    // First result should be closest
    assert!(results[0].distance < 1.0);
}

#[test]
fn test_hnsw_index_search_empty() {
    let config = HNSWConfig::new(32);
    let index = HNSWIndex::new(config);
    
    let query: Vec<f32> = vec![0.0; 32];
    let results = index.search(&query, 5).expect("search");
    
    assert!(results.is_empty());
}

#[test]
fn test_hnsw_result_ordering() {
    let config = HNSWConfig::new(8);
    let mut index = HNSWIndex::new(config);
    
    // Add vectors with increasing distance from origin
    for i in 0..5 {
        let embedding: Vec<f32> = vec![(i + 1) as f32; 8];
        let id = format!("dist_{}", i + 1);
        index.add(id, &embedding).expect("add vector");
    }
    
    // Search for origin
    let query: Vec<f32> = vec![0.0; 8];
    let results = index.search(&query, 5).expect("search");
    
    // Results should be ordered by distance (ascending)
    for i in 1..results.len() {
        assert!(results[i].distance >= results[i-1].distance);
    }
}

// === Trajectory Coordinate Tests ===

#[test]
fn test_trajectory_5d_creation() {
    // TrajectoryCoordinate5D::new(depth, sibling_order, homogeneity, temporal, complexity)
    let coord = TrajectoryCoordinate5D::new(3, 0, 0.8, 0.5, 1);
    
    assert_eq!(coord.depth, 3);
    assert_eq!(coord.sibling_order, 0);
    assert!((coord.homogeneity - 0.8).abs() < 0.001);
    assert!((coord.temporal - 0.5).abs() < 0.001);
    assert_eq!(coord.complexity, 1);
}

#[test]
fn test_trajectory_5d_distance() {
    // complexity must be at least 1
    let coord1 = TrajectoryCoordinate5D::new(0, 0, 0.0, 0.0, 1);
    let coord2 = TrajectoryCoordinate5D::new(5, 2, 1.0, 1.0, 10);
    
    let distance = coord1.distance(&coord2);
    assert!(distance > 0.0);
    
    // Self distance should be 0
    let self_distance = coord1.distance(&coord1);
    assert!(self_distance.abs() < 0.001);
}

#[test]
fn test_trajectory_5d_weighted_distance() {
    // complexity must be at least 1
    let coord1 = TrajectoryCoordinate5D::new(0, 0, 0.0, 0.0, 1);
    let coord2 = TrajectoryCoordinate5D::new(5, 0, 0.0, 0.0, 1);
    
    // With default weights
    let weights = DLMWeights::default();
    let distance_default = coord1.dlm_distance(&coord2, &weights);
    
    // With structural-focused weights (emphasize depth)
    let structural_weights = DLMWeights::structural_focused();
    let distance_structural = coord1.dlm_distance(&coord2, &structural_weights);
    
    // Both should give positive distance for depth difference
    assert!(distance_structural > 0.0);
    assert!(distance_default > 0.0);
}

#[test]
fn test_dlm_weights_normalization() {
    // Weights should auto-normalize to sum to 1.0
    let weights = DLMWeights::new(1.0, 1.0, 1.0, 1.0, 1.0);
    let sum = weights.depth + weights.sibling + weights.homogeneity + weights.temporal + weights.complexity;
    assert!((sum - 1.0).abs() < 0.001);
}

#[test]
fn test_dlm_weight_presets() {
    let semantic = DLMWeights::semantic_focused();
    let structural = DLMWeights::structural_focused();
    let temporal = DLMWeights::temporal_focused();
    
    // Semantic-focused should emphasize homogeneity
    assert!(semantic.homogeneity > semantic.depth);
    
    // Structural-focused should emphasize depth
    assert!(structural.depth > structural.homogeneity);
    
    // Temporal-focused should emphasize temporal
    assert!(temporal.temporal > temporal.depth);
}

// === Salience Scoring Tests ===

#[test]
fn test_salience_scorer_creation() {
    let scorer = SalienceScorer::new();
    
    // Default scorer should give base score (0.5) for neutral factors
    let factors = SalienceFactors::default();
    let result = scorer.score_single(&factors, None);
    
    assert!(result.score >= 0.0 && result.score <= 1.0);
}

#[test]
fn test_salience_factors_default() {
    let factors = SalienceFactors::default();
    
    assert_eq!(factors.turn_id, 0);
    assert!(factors.feedback.is_none());
    assert!(!factors.is_phase_transition);
    assert_eq!(factors.reference_count, 0);
    assert!(factors.embedding.is_none());
}

#[test]
fn test_salience_feedback_boost() {
    let scorer = SalienceScorer::new();
    
    // ThumbsUp should boost score
    let mut factors_up = SalienceFactors::default();
    factors_up.feedback = Some(Feedback::ThumbsUp);
    let result_up = scorer.score_single(&factors_up, None);
    
    // ThumbsDown should reduce score
    let mut factors_down = SalienceFactors::default();
    factors_down.feedback = Some(Feedback::ThumbsDown);
    let result_down = scorer.score_single(&factors_down, None);
    
    // Neutral
    let factors_neutral = SalienceFactors::default();
    let result_neutral = scorer.score_single(&factors_neutral, None);
    
    assert!(result_up.score > result_neutral.score);
    assert!(result_neutral.score > result_down.score);
}

#[test]
fn test_salience_phase_transition_boost() {
    let scorer = SalienceScorer::new();
    
    let mut factors = SalienceFactors::default();
    factors.is_phase_transition = true;
    let result = scorer.score_single(&factors, None);
    
    let baseline = scorer.score_single(&SalienceFactors::default(), None);
    
    // Phase transition should boost score
    assert!(result.phase_contribution > 0.0);
    assert!(result.score > baseline.score);
}

#[test]
fn test_salience_reference_boost_capped() {
    let scorer = SalienceScorer::new();
    
    let mut factors = SalienceFactors::default();
    factors.reference_count = 100; // Many references
    let result = scorer.score_single(&factors, None);
    
    // Reference contribution should be capped
    assert!(result.reference_contribution <= 0.15);
}

#[test]
fn test_salience_batch_scoring() {
    let scorer = SalienceScorer::new();
    
    let turns: Vec<SalienceFactors> = (0..10)
        .map(|i| {
            let mut f = SalienceFactors::default();
            f.turn_id = i as u64;
            f.reference_count = i;
            f
        })
        .collect();
    
    let scores = scorer.score_corpus(&turns);
    
    assert_eq!(scores.len(), 10);
    
    // All scores should be valid
    for score in &scores {
        assert!(score.score >= 0.0);
        assert!(score.score <= 1.0);
    }
}

#[test]
fn test_salience_stats() {
    let scorer = SalienceScorer::new();
    
    let saliences = vec![
        TurnSalience {
            turn_id: 1,
            score: 0.2,
            feedback_contribution: 0.0,
            phase_contribution: 0.0,
            reference_contribution: 0.0,
            novelty_contribution: 0.0,
        },
        TurnSalience {
            turn_id: 2,
            score: 0.5,
            feedback_contribution: 0.0,
            phase_contribution: 0.0,
            reference_contribution: 0.0,
            novelty_contribution: 0.0,
        },
        TurnSalience {
            turn_id: 3,
            score: 0.8,
            feedback_contribution: 0.0,
            phase_contribution: 0.0,
            reference_contribution: 0.0,
            novelty_contribution: 0.0,
        },
    ];
    
    let stats = scorer.compute_stats(&saliences);
    
    assert_eq!(stats.total_turns, 3);
    assert!((stats.mean_salience - 0.5).abs() < 0.01);
    assert!(stats.min_salience < 0.3);
    assert!(stats.max_salience > 0.7);
}

// === IRCP Propagation Tests ===

#[test]
fn test_ircp_config_default() {
    let config = IRCPConfig::default();
    
    assert!(config.temperature > 0.0);
    assert!(config.spatial_weight >= 0.0 && config.spatial_weight <= 1.0);
    assert!(config.min_attention > 0.0);
}

#[test]
fn test_ircp_config_presets() {
    let semantic = IRCPConfig::semantic_focused();
    let spatial = IRCPConfig::spatial_focused();
    let causal = IRCPConfig::causal();
    let sharp = IRCPConfig::sharp();
    let diffuse = IRCPConfig::diffuse();
    
    // Semantic-focused should have low spatial weight
    assert!(semantic.spatial_weight < 0.5);
    
    // Spatial-focused should have high spatial weight
    assert!(spatial.spatial_weight > 0.5);
    
    // Causal should have causal mask enabled
    assert!(causal.causal_mask);
    
    // Sharp should have low temperature
    assert!(sharp.temperature < 1.0);
    
    // Diffuse should have high temperature
    assert!(diffuse.temperature > 1.0);
}

#[test]
fn test_ircp_propagator_creation() {
    let config = IRCPConfig::default();
    let propagator = IRCPPropagator::new(config);
    
    // Propagator should be created successfully (no panic)
    let _ = propagator;
}

#[test]
fn test_attention_weights_empty() {
    let empty = AttentionWeights::empty();
    
    assert!(empty.forward.is_empty());
    assert!(empty.inverse.is_empty());
    assert!(empty.cross.is_empty());
    assert!(empty.total_mass == 0.0);
}

#[test]
fn test_attention_weights_uniform() {
    let uniform = AttentionWeights::uniform(5);
    
    assert_eq!(uniform.forward.len(), 5);
    
    // All weights should be equal
    for w in &uniform.forward {
        assert!((*w - 0.2).abs() < 0.01);
    }
    
    // Total mass should be 1.0
    assert!((uniform.total_mass - 1.0).abs() < 0.01);
}

#[test]
fn test_attention_weights_top_forward() {
    let mut weights = AttentionWeights::uniform(5);
    weights.forward[2] = 0.5; // Make index 2 highest
    
    let top = weights.top_forward();
    assert_eq!(top, Some(2));
}

#[test]
fn test_attention_weights_sorted_indices() {
    let mut weights = AttentionWeights::uniform(3);
    weights.forward = vec![0.1, 0.5, 0.3];
    
    let sorted = weights.sorted_forward_indices();
    assert_eq!(sorted, vec![1, 2, 0]); // Descending order
}

#[test]
fn test_attention_weights_entropy() {
    let uniform = AttentionWeights::uniform(5);
    let entropy = uniform.forward_entropy();
    
    // Uniform distribution has maximum entropy
    assert!(entropy > 0.0);
    
    // Concentrated distribution has low entropy
    let mut concentrated = AttentionWeights::uniform(5);
    concentrated.forward = vec![0.9, 0.025, 0.025, 0.025, 0.025];
    let conc_entropy = concentrated.forward_entropy();
    
    assert!(conc_entropy < entropy);
}

// === Cache Tests ===

#[test]
fn test_cache_config_default() {
    let config = CacheConfig::default();
    assert!(config.max_entries > 0);
}

#[test]
fn test_query_cache_creation() {
    let config = CacheConfig::default();
    let cache = QueryCache::new(config);
    
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

// === Integration Tests ===

#[test]
fn test_full_pipeline_simulation() {
    // Simulate a RAG++ query pipeline
    
    // 1. Create HNSW index
    let config = HNSWConfig::new(16);
    let mut index = HNSWIndex::new(config);
    
    // 2. Add some "memory turns"
    for i in 0..20 {
        let id = format!("turn_{}", i);
        let embedding: Vec<f32> = (0..16).map(|j| ((i + j) as f32 * 0.2).sin()).collect();
        index.add(id, &embedding).expect("add turn");
    }
    
    // 3. Query
    let query_embedding: Vec<f32> = (0..16).map(|j| (j as f32 * 0.2).sin()).collect();
    let neighbors = index.search(&query_embedding, 5).expect("search");
    
    assert!(!neighbors.is_empty());
    
    // 4. Compute trajectory coordinates for neighbors
    let coords: Vec<TrajectoryCoordinate5D> = neighbors.iter().enumerate()
        .map(|(i, n)| TrajectoryCoordinate5D::new(
            i as u32,           // depth
            0,                  // sibling_order
            1.0 - n.distance.min(1.0), // homogeneity from similarity
            i as f32 / 5.0,     // temporal
            1,                  // complexity
        ))
        .collect();
    
    // 5. Create IRCP propagator
    let ircp_config = IRCPConfig::default();
    let _propagator = IRCPPropagator::new(ircp_config);
    
    // 6. Compute salience
    let scorer = SalienceScorer::new();
    
    let factors: Vec<SalienceFactors> = (0..neighbors.len())
        .map(|i| {
            let mut f = SalienceFactors::default();
            f.turn_id = i as u64;
            f.reference_count = i;
            f
        })
        .collect();
    
    let saliences = scorer.score_corpus(&factors);
    
    // All steps completed successfully
    assert!(!saliences.is_empty());
    assert_eq!(coords.len(), neighbors.len());
}

#[test]
fn test_coordinate_as_vector() {
    let coord = TrajectoryCoordinate5D::new(3, 1, 0.8, 0.5, 2);
    let vec = coord.as_vector();
    
    // Should have 5 elements
    assert_eq!(vec.len(), 5);
}

#[test]
fn test_trajectory_cosine_similarity() {
    let coord1 = TrajectoryCoordinate5D::new(3, 0, 0.8, 0.5, 1);
    let coord2 = TrajectoryCoordinate5D::new(3, 0, 0.8, 0.5, 1);
    
    let similarity = coord1.cosine_similarity(&coord2);
    
    // Identical coordinates should have similarity close to 1.0
    assert!((similarity - 1.0).abs() < 0.01);
}

#[test]
fn test_dlm_weights_from_array() {
    let weights = DLMWeights::from_array([0.2, 0.2, 0.2, 0.2, 0.2]);
    let arr = weights.to_array();
    
    assert_eq!(arr.len(), 5);
    for w in arr {
        assert!((w - 0.2_f32).abs() < 0.001);
    }
}

#[test]
fn test_coordinate_default() {
    let coord = TrajectoryCoordinate5D::default();
    
    assert_eq!(coord.depth, 0);
    assert_eq!(coord.sibling_order, 0);
    // Default has sensible values, just verify they're in valid range
    assert!(coord.homogeneity >= 0.0 && coord.homogeneity <= 1.0);
    assert!(coord.temporal >= 0.0 && coord.temporal <= 1.0);
    // Default complexity is at least 1
    assert!(coord.complexity >= 1);
}
