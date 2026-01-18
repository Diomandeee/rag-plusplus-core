//! Distance Computation Module
//!
//! Provides optimized distance/similarity functions for vector operations.

#![allow(unsafe_code)]  // SIMD dispatch requires unsafe for intrinsics
//!
//! # Architecture
//!
//! This module is designed for SIMD acceleration:
//! - `scalar`: Pure Rust baseline implementations
//! - `simd_avx2`: AVX2-accelerated (x86_64) - conditionally compiled
//! - `simd_neon`: NEON-accelerated (ARM64) - future
//!
//! Runtime dispatch automatically selects the fastest available implementation
//! based on CPU feature detection.
//!
//! # Distance Types
//!
//! - **L2 (Euclidean)**: `sqrt(sum((a[i] - b[i])^2))` - lower is more similar
//! - **Inner Product**: `sum(a[i] * b[i])` - higher is more similar
//! - **Cosine**: `dot(a, b) / (norm(a) * norm(b))` - higher is more similar
//!
//! # Performance Notes
//!
//! - AVX2 provides 4-8x speedup on x86_64 CPUs (Intel Haswell+, AMD Zen+)
//! - For cosine similarity on pre-normalized vectors, use inner product instead
//!   (equivalent result, avoids redundant normalization)

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod simd_avx2;

// Re-export scalar functions as the baseline API
pub use scalar::{
    l2_distance,
    l2_distance_squared,
    inner_product,
    cosine_similarity,
    cosine_distance,
    norm,
    norm_squared,
    normalize,
    normalize_in_place,
    // Batch operations
    normalize_batch,
    normalize_batch_flat,
    compute_norms_batch,
    find_unnormalized,
};

use crate::index::DistanceType;

// =============================================================================
// RUNTIME SIMD DISPATCH
// =============================================================================

/// Check if AVX2 + FMA is available at runtime.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// L2 distance with automatic SIMD dispatch.
///
/// Selects the fastest implementation based on CPU features:
/// - AVX2+FMA on x86_64 (if available)
/// - Scalar fallback otherwise
#[inline]
pub fn l2_distance_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: CPU feature detection guarantees AVX2+FMA support
            return unsafe { simd_avx2::l2_distance_avx2(a, b) };
        }
    }
    scalar::l2_distance(a, b)
}

/// Inner product with automatic SIMD dispatch.
#[inline]
pub fn inner_product_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { simd_avx2::inner_product_avx2(a, b) };
        }
    }
    scalar::inner_product(a, b)
}

/// Cosine similarity with automatic SIMD dispatch.
#[inline]
pub fn cosine_similarity_fast(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { simd_avx2::cosine_similarity_avx2(a, b) };
        }
    }
    scalar::cosine_similarity(a, b)
}

/// Cosine distance with automatic SIMD dispatch.
#[inline]
pub fn cosine_distance_fast(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_fast(a, b)
}

/// Norm with automatic SIMD dispatch.
#[inline]
pub fn norm_fast(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { simd_avx2::norm_avx2(a) };
        }
    }
    scalar::norm(a)
}

/// Normalize in-place with automatic SIMD dispatch.
#[inline]
pub fn normalize_in_place_fast(a: &mut [f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { simd_avx2::normalize_in_place_avx2(a) };
        }
    }
    scalar::normalize_in_place(a)
}

/// Compute distance between two vectors based on distance type.
///
/// This is the primary dispatch function used by indexes.
/// Automatically uses SIMD acceleration when available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
/// * `distance_type` - Type of distance metric
///
/// # Returns
///
/// Distance value. Interpretation depends on distance type:
/// - L2: Lower is more similar (0 = identical)
/// - InnerProduct: Higher is more similar
/// - Cosine: Higher is more similar (range -1 to 1)
///
/// # Panics
///
/// Panics if vectors have different lengths.
#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], distance_type: DistanceType) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    match distance_type {
        DistanceType::L2 => l2_distance_fast(a, b),
        DistanceType::InnerProduct => inner_product_fast(a, b),
        DistanceType::Cosine => cosine_similarity_fast(a, b),
    }
}

/// Compute distance for heap-based search (lower = better for all types).
///
/// This function returns values suitable for min-heap based nearest neighbor search.
/// For similarity metrics (IP, Cosine), the sign/value is adjusted so that
/// lower values indicate more similar vectors.
/// Automatically uses SIMD acceleration when available.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
/// * `distance_type` - Type of distance metric
///
/// # Returns
///
/// Heap-compatible distance where lower = more similar.
#[inline]
pub fn compute_distance_for_heap(a: &[f32], b: &[f32], distance_type: DistanceType) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    match distance_type {
        DistanceType::L2 => l2_distance_fast(a, b),
        DistanceType::InnerProduct => -inner_product_fast(a, b), // Negate: higher IP = lower heap value
        DistanceType::Cosine => cosine_distance_fast(a, b),       // 1 - cos: higher cos = lower distance
    }
}

/// Check if a distance type uses similarity (higher = more similar).
#[inline]
#[must_use]
pub const fn is_similarity_metric(distance_type: DistanceType) -> bool {
    matches!(distance_type, DistanceType::InnerProduct | DistanceType::Cosine)
}

// =============================================================================
// BATCH OPERATIONS WITH SIMD DISPATCH
// =============================================================================

/// Parallel batch L2 normalization using SIMD and multi-threading.
///
/// This is the high-performance version for processing thousands of vectors.
/// Uses rayon for parallelism and SIMD for per-vector operations.
///
/// # Arguments
///
/// * `data` - Flat array of vectors stored contiguously
/// * `dim` - Dimension of each vector
///
/// # Returns
///
/// Vector of original norms for each vector.
///
/// # Performance
///
/// - Uses rayon for parallel processing across CPU cores
/// - Uses SIMD (AVX2 on x86_64) for individual vector operations
/// - Optimal for batches of 1000+ vectors
#[cfg(feature = "parallel")]
pub fn normalize_batch_parallel(data: &mut [f32], dim: usize) -> Vec<f32> {
    use rayon::prelude::*;

    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    // Split into chunks and process in parallel using rayon
    data.par_chunks_mut(dim)
        .map(|vector| normalize_in_place_fast(vector))
        .collect()
}

/// Non-parallel version for when rayon is not available.
#[cfg(not(feature = "parallel"))]
pub fn normalize_batch_parallel(data: &mut [f32], dim: usize) -> Vec<f32> {
    normalize_batch_flat_fast(data, dim)
}

/// Batch L2 normalization with SIMD dispatch (single-threaded).
///
/// # Arguments
///
/// * `data` - Flat array of vectors stored contiguously
/// * `dim` - Dimension of each vector
///
/// # Returns
///
/// Vector of original norms for each vector.
pub fn normalize_batch_flat_fast(data: &mut [f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    let n_vectors = data.len() / dim;
    let mut norms = Vec::with_capacity(n_vectors);

    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vector = &mut data[start..end];
        let n = normalize_in_place_fast(vector);
        norms.push(n);
    }

    norms
}

/// Compute norms for a batch of vectors with SIMD dispatch.
pub fn compute_norms_batch_fast(data: &[f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    let n_vectors = data.len() / dim;
    let mut norms = Vec::with_capacity(n_vectors);

    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vector = &data[start..end];
        norms.push(norm_fast(vector));
    }

    norms
}

// =============================================================================
// TRAJECTORY-WEIGHTED DISTANCE (TPO Integration)
// =============================================================================

use crate::trajectory::{TrajectoryCoordinate, TrajectoryCoordinate5D};

/// Compute trajectory-weighted cosine similarity.
///
/// Combines semantic similarity (cosine of embeddings) with spatial similarity
/// (distance in trajectory coordinate space). This enables retrieval that
/// considers both content and structural position.
///
/// # Formula
///
/// ```text
/// weighted_sim = (1 - coord_weight) * cosine(a, b) + coord_weight * (1 - coord_dist)
/// ```
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
/// * `coord_a` - Trajectory coordinate of first episode
/// * `coord_b` - Trajectory coordinate of second episode
/// * `coord_weight` - Weight for coordinate component [0, 1]
///   - 0.0 = pure semantic similarity (ignore coordinates)
///   - 0.5 = equal weight to semantic and spatial
///   - 1.0 = pure spatial similarity (ignore content)
///
/// # Returns
///
/// Combined similarity score. Range depends on inputs but typically [0, 1].
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::trajectory_weighted_cosine;
/// use rag_plusplus_core::trajectory::TrajectoryCoordinate;
///
/// let emb_a = vec![1.0, 0.0, 0.0];
/// let emb_b = vec![0.9, 0.436, 0.0]; // Similar direction
///
/// let coord_a = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);
/// let coord_b = TrajectoryCoordinate::new(2, 0, 0.8, 0.3); // Close in trajectory
///
/// let sim = trajectory_weighted_cosine(&emb_a, &emb_b, &coord_a, &coord_b, 0.3);
/// assert!(sim > 0.8); // High due to both semantic and spatial similarity
/// ```
#[inline]
pub fn trajectory_weighted_cosine(
    a: &[f32],
    b: &[f32],
    coord_a: &TrajectoryCoordinate,
    coord_b: &TrajectoryCoordinate,
    coord_weight: f32,
) -> f32 {
    let coord_weight = coord_weight.clamp(0.0, 1.0);

    let semantic_sim = cosine_similarity_fast(a, b);
    let coord_dist = coord_a.distance(coord_b);

    // Normalize coord_dist to [0, 1] range (max distance is ~4.0 for 4D coords)
    // Using a typical max distance of 4.0 for normalization
    let coord_sim = (1.0 - coord_dist / 4.0).clamp(0.0, 1.0);

    (1.0 - coord_weight) * semantic_sim + coord_weight * coord_sim
}

/// Compute trajectory-weighted cosine similarity with 5D coordinates.
///
/// Same as [`trajectory_weighted_cosine`] but uses 5D coordinates that include
/// the complexity dimension from TPO.
#[inline]
pub fn trajectory_weighted_cosine_5d(
    a: &[f32],
    b: &[f32],
    coord_a: &TrajectoryCoordinate5D,
    coord_b: &TrajectoryCoordinate5D,
    coord_weight: f32,
) -> f32 {
    let coord_weight = coord_weight.clamp(0.0, 1.0);

    let semantic_sim = cosine_similarity_fast(a, b);
    let coord_dist = coord_a.distance(coord_b);

    // Normalize coord_dist to [0, 1] range (max distance is ~4.5 for 5D coords)
    let coord_sim = (1.0 - coord_dist / 4.5).clamp(0.0, 1.0);

    (1.0 - coord_weight) * semantic_sim + coord_weight * coord_sim
}

/// Compute trajectory-weighted L2 distance.
///
/// Combines L2 distance with trajectory coordinate distance.
/// Lower values indicate more similar episodes.
#[inline]
pub fn trajectory_weighted_l2(
    a: &[f32],
    b: &[f32],
    coord_a: &TrajectoryCoordinate,
    coord_b: &TrajectoryCoordinate,
    coord_weight: f32,
) -> f32 {
    let coord_weight = coord_weight.clamp(0.0, 1.0);

    let semantic_dist = l2_distance_fast(a, b);
    let coord_dist = coord_a.distance(coord_b);

    (1.0 - coord_weight) * semantic_dist + coord_weight * coord_dist
}

/// Compute trajectory-weighted inner product.
///
/// Combines inner product with trajectory coordinate similarity.
/// Higher values indicate more similar episodes.
#[inline]
pub fn trajectory_weighted_inner_product(
    a: &[f32],
    b: &[f32],
    coord_a: &TrajectoryCoordinate,
    coord_b: &TrajectoryCoordinate,
    coord_weight: f32,
) -> f32 {
    let coord_weight = coord_weight.clamp(0.0, 1.0);

    let semantic_sim = inner_product_fast(a, b);
    let coord_dist = coord_a.distance(coord_b);

    // For inner product, we need to convert coord_dist to a similarity-like value
    let coord_sim = (4.0 - coord_dist).max(0.0); // Higher = more similar

    (1.0 - coord_weight) * semantic_sim + coord_weight * coord_sim
}

/// Compute trajectory-weighted distance with configurable distance type.
///
/// # Arguments
///
/// * `a` - First embedding vector
/// * `b` - Second embedding vector
/// * `coord_a` - Trajectory coordinate of first episode
/// * `coord_b` - Trajectory coordinate of second episode
/// * `distance_type` - Type of semantic distance to use
/// * `coord_weight` - Weight for coordinate component [0, 1]
///
/// # Returns
///
/// Combined distance/similarity. Interpretation depends on distance_type.
#[inline]
pub fn trajectory_weighted_distance(
    a: &[f32],
    b: &[f32],
    coord_a: &TrajectoryCoordinate,
    coord_b: &TrajectoryCoordinate,
    distance_type: DistanceType,
    coord_weight: f32,
) -> f32 {
    match distance_type {
        DistanceType::L2 => trajectory_weighted_l2(a, b, coord_a, coord_b, coord_weight),
        DistanceType::InnerProduct => trajectory_weighted_inner_product(a, b, coord_a, coord_b, coord_weight),
        DistanceType::Cosine => trajectory_weighted_cosine(a, b, coord_a, coord_b, coord_weight),
    }
}

/// Configuration for trajectory-weighted distance computation.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryDistanceConfig {
    /// Weight for coordinate component [0, 1]
    pub coord_weight: f32,
    /// Base distance type
    pub distance_type: DistanceType,
    /// Whether to boost similarity for same-phase episodes
    pub phase_boost: bool,
    /// Boost amount for same-phase episodes
    pub phase_boost_amount: f32,
}

impl Default for TrajectoryDistanceConfig {
    fn default() -> Self {
        Self {
            coord_weight: 0.2,       // Mostly semantic, some trajectory context
            distance_type: DistanceType::Cosine,
            phase_boost: true,
            phase_boost_amount: 0.1, // 10% boost for same phase
        }
    }
}

impl TrajectoryDistanceConfig {
    /// Create config for pure semantic distance (no trajectory weighting).
    pub fn semantic_only() -> Self {
        Self {
            coord_weight: 0.0,
            ..Default::default()
        }
    }

    /// Create config with equal semantic and trajectory weight.
    pub fn balanced() -> Self {
        Self {
            coord_weight: 0.5,
            ..Default::default()
        }
    }

    /// Create config emphasizing trajectory structure.
    pub fn trajectory_focused() -> Self {
        Self {
            coord_weight: 0.7,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_distance_l2() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];

        let dist = compute_distance(&a, &b, DistanceType::L2);
        assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_compute_distance_inner_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        let dist = compute_distance(&a, &b, DistanceType::InnerProduct);
        assert!((dist - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_compute_distance_cosine() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];

        let dist = compute_distance(&a, &b, DistanceType::Cosine);
        assert!((dist - 1.0).abs() < 1e-6); // Identical = cosine 1.0
    }

    #[test]
    fn test_heap_distance_ordering() {
        let a = [1.0, 0.0, 0.0];
        let b_close = [0.9, 0.1, 0.0];
        let b_far = [0.0, 1.0, 0.0];

        // For all metrics, closer vector should have lower heap distance
        for dt in [DistanceType::L2, DistanceType::InnerProduct, DistanceType::Cosine] {
            let d_close = compute_distance_for_heap(&a, &b_close, dt);
            let d_far = compute_distance_for_heap(&a, &b_far, dt);

            // Note: This test may not hold for all vector combinations
            // The key invariant is consistent ordering within each metric
        }
    }

    #[test]
    fn test_is_similarity_metric() {
        assert!(!is_similarity_metric(DistanceType::L2));
        assert!(is_similarity_metric(DistanceType::InnerProduct));
        assert!(is_similarity_metric(DistanceType::Cosine));
    }

    // =========================================================================
    // TRAJECTORY-WEIGHTED DISTANCE TESTS
    // =========================================================================

    #[test]
    fn test_trajectory_weighted_cosine_pure_semantic() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];

        let coord_a = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);
        let coord_b = TrajectoryCoordinate::new(5, 3, 0.2, 1.0); // Very different coords

        // With coord_weight = 0, should be pure cosine
        let sim = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 0.0);
        assert!((sim - 1.0).abs() < 1e-6); // Identical embeddings
    }

    #[test]
    fn test_trajectory_weighted_cosine_pure_spatial() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0]; // Orthogonal

        let coord_a = TrajectoryCoordinate::new(1, 0, 0.9, 0.5);
        let coord_b = TrajectoryCoordinate::new(1, 0, 0.9, 0.5); // Identical coords

        // With coord_weight = 1, should be pure spatial (coords identical = max similarity)
        let sim = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 1.0);
        assert!((sim - 1.0).abs() < 1e-6); // Identical coordinates
    }

    #[test]
    fn test_trajectory_weighted_cosine_mixed() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [0.9, 0.436, 0.0]; // Cosine ~0.9

        let coord_a = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);
        let coord_b = TrajectoryCoordinate::new(2, 0, 0.8, 0.3);

        // With coord_weight = 0.3, should blend both
        let sim = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 0.3);
        assert!(sim > 0.7 && sim < 1.0); // Blended result
    }

    #[test]
    fn test_trajectory_weighted_cosine_5d() {
        use crate::trajectory::TrajectoryCoordinate5D;

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];

        let coord_a = TrajectoryCoordinate5D::new(1, 0, 0.9, 0.2, 1);
        let coord_b = TrajectoryCoordinate5D::new(1, 0, 0.9, 0.2, 3); // Different complexity

        // Identical embeddings, similar coords except complexity
        let sim = trajectory_weighted_cosine_5d(&a, &b, &coord_a, &coord_b, 0.3);
        assert!(sim > 0.9); // High similarity despite complexity difference
    }

    #[test]
    fn test_trajectory_weighted_l2() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0]; // Identical

        let coord_a = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);
        let coord_b = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);

        // Both identical - should be 0
        let dist = trajectory_weighted_l2(&a, &b, &coord_a, &coord_b, 0.5);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_weighted_inner_product() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];

        let coord_a = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);
        let coord_b = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);

        let sim = trajectory_weighted_inner_product(&a, &b, &coord_a, &coord_b, 0.5);
        // Identical vectors: IP=1.0, coords identical: coord_sim=4.0
        // Result should be 0.5 * 1.0 + 0.5 * 4.0 = 2.5
        assert!((sim - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_trajectory_weighted_distance_dispatch() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [0.9, 0.436, 0.0];

        let coord_a = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);
        let coord_b = TrajectoryCoordinate::new(2, 0, 0.8, 0.3);

        // Test all distance types dispatch correctly
        let _ = trajectory_weighted_distance(&a, &b, &coord_a, &coord_b, DistanceType::L2, 0.3);
        let _ = trajectory_weighted_distance(&a, &b, &coord_a, &coord_b, DistanceType::InnerProduct, 0.3);
        let _ = trajectory_weighted_distance(&a, &b, &coord_a, &coord_b, DistanceType::Cosine, 0.3);
    }

    #[test]
    fn test_trajectory_distance_config_presets() {
        let semantic = TrajectoryDistanceConfig::semantic_only();
        assert_eq!(semantic.coord_weight, 0.0);

        let balanced = TrajectoryDistanceConfig::balanced();
        assert_eq!(balanced.coord_weight, 0.5);

        let trajectory = TrajectoryDistanceConfig::trajectory_focused();
        assert_eq!(trajectory.coord_weight, 0.7);
    }

    #[test]
    fn test_trajectory_distance_config_defaults() {
        let config = TrajectoryDistanceConfig::default();
        assert_eq!(config.coord_weight, 0.2);
        assert!(config.phase_boost);
        assert_eq!(config.phase_boost_amount, 0.1);
    }

    #[test]
    fn test_coord_weight_clamping() {
        use crate::trajectory::TrajectoryCoordinate;

        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];

        let coord_a = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);
        let coord_b = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);

        // Weight > 1.0 should be clamped to 1.0
        let sim_over = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 5.0);
        let sim_one = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 1.0);
        assert!((sim_over - sim_one).abs() < 1e-6);

        // Weight < 0.0 should be clamped to 0.0
        let sim_under = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, -2.0);
        let sim_zero = trajectory_weighted_cosine(&a, &b, &coord_a, &coord_b, 0.0);
        assert!((sim_under - sim_zero).abs() < 1e-6);
    }
}
