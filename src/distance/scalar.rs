//! Scalar (Pure Rust) Distance Implementations
//!
//! These are the baseline implementations without SIMD.
//! They serve as:
//! - Fallback when SIMD is not available
//! - Reference implementations for correctness testing
//! - Baseline for benchmark comparisons

/// Compute L2 (Euclidean) distance between two vectors.
///
/// `L2 = sqrt(sum((a[i] - b[i])^2))`
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The Euclidean distance. Lower values indicate more similar vectors.
/// Returns 0.0 for identical vectors.
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::l2_distance;
///
/// let a = [1.0, 0.0, 0.0];
/// let b = [0.0, 1.0, 0.0];
/// let dist = l2_distance(&a, &b);
/// assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
/// ```
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// Compute squared L2 distance (avoids sqrt for comparison-only use).
///
/// When only comparing distances (not using the actual value), using
/// squared distance avoids the expensive sqrt operation.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The squared Euclidean distance.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute inner product (dot product) of two vectors.
///
/// `IP = sum(a[i] * b[i])`
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The inner product. Higher values indicate more similar vectors
/// (assuming vectors point in similar directions).
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::inner_product;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let ip = inner_product(&a, &b);
/// assert!((ip - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Compute cosine similarity between two vectors.
///
/// `cos(a, b) = dot(a, b) / (||a|| * ||b||)`
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Cosine similarity in range [-1, 1]. Higher values indicate more similar vectors.
/// Returns 1.0 for identical directions, 0.0 for orthogonal, -1.0 for opposite.
/// Returns 0.0 if either vector has zero norm.
///
/// # Performance Note
///
/// For pre-normalized vectors (norm = 1), use `inner_product` instead -
/// it's equivalent but faster (skips norm computation).
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::cosine_similarity;
///
/// let a = [1.0, 0.0];
/// let b = [1.0, 0.0];
/// let cos = cosine_similarity(&a, &b);
/// assert!((cos - 1.0).abs() < 1e-6); // Identical = 1.0
/// ```
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let dot = inner_product(a, b);
    let norm_a = norm(a);
    let norm_b = norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute cosine distance (1 - cosine_similarity).
///
/// This converts cosine similarity to a distance where lower = more similar,
/// suitable for min-heap based nearest neighbor search.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Cosine distance in range [0, 2]. Lower values indicate more similar vectors.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute the L2 norm (magnitude) of a vector.
///
/// `||a|| = sqrt(sum(a[i]^2))`
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// The L2 norm of the vector. Returns 0.0 for zero vector.
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute squared L2 norm (avoids sqrt).
#[inline]
pub fn norm_squared(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// Create a normalized copy of a vector (unit vector).
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// A new vector with the same direction but norm = 1.
/// Returns zero vector if input has zero norm.
#[inline]
#[must_use]
pub fn normalize(a: &[f32]) -> Vec<f32> {
    let n = norm(a);
    if n == 0.0 {
        vec![0.0; a.len()]
    } else {
        a.iter().map(|x| x / n).collect()
    }
}

/// Normalize a vector in-place.
///
/// # Arguments
///
/// * `a` - Vector to normalize (modified in-place)
///
/// # Returns
///
/// The original norm of the vector.
#[inline]
pub fn normalize_in_place(a: &mut [f32]) -> f32 {
    let n = norm(a);
    if n > 0.0 {
        for x in a.iter_mut() {
            *x /= n;
        }
    }
    n
}

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

/// Batch L2 normalization of multiple vectors.
///
/// Normalizes each vector in-place. Optimized for processing large batches
/// of embeddings (thousands of vectors).
///
/// # Arguments
///
/// * `vectors` - Mutable slice of vectors, each represented as a mutable slice
///
/// # Returns
///
/// Vector of original norms for each input vector.
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::normalize_batch;
///
/// let mut vecs = vec![
///     vec![3.0f32, 4.0],
///     vec![5.0, 12.0],
///     vec![8.0, 15.0],
/// ];
///
/// // Convert to mutable slices
/// let mut slices: Vec<&mut [f32]> = vecs.iter_mut().map(|v| v.as_mut_slice()).collect();
/// let norms = normalize_batch(&mut slices);
///
/// assert!((norms[0] - 5.0).abs() < 1e-6);   // 3-4-5
/// assert!((norms[1] - 13.0).abs() < 1e-6);  // 5-12-13
/// assert!((norms[2] - 17.0).abs() < 1e-6);  // 8-15-17
/// ```
pub fn normalize_batch(vectors: &mut [&mut [f32]]) -> Vec<f32> {
    vectors.iter_mut()
        .map(|v| normalize_in_place(v))
        .collect()
}

/// Batch L2 normalization of flat vector storage.
///
/// More memory-efficient version that operates on a flat array where
/// vectors are stored contiguously. Each vector has the same dimension.
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
/// # Panics
///
/// Panics if `data.len() % dim != 0`.
///
/// # Example
///
/// ```
/// use rag_plusplus_core::distance::normalize_batch_flat;
///
/// let mut data = vec![
///     3.0f32, 4.0,    // Vector 0: norm = 5
///     5.0, 12.0,      // Vector 1: norm = 13
/// ];
///
/// let norms = normalize_batch_flat(&mut data, 2);
///
/// assert!((norms[0] - 5.0).abs() < 1e-6);
/// assert!((norms[1] - 13.0).abs() < 1e-6);
/// assert!((data[0] - 0.6).abs() < 1e-6);  // 3/5
/// assert!((data[1] - 0.8).abs() < 1e-6);  // 4/5
/// ```
pub fn normalize_batch_flat(data: &mut [f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    let n_vectors = data.len() / dim;
    let mut norms = Vec::with_capacity(n_vectors);

    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vector = &mut data[start..end];
        let n = normalize_in_place(vector);
        norms.push(n);
    }

    norms
}

/// Compute norms for a batch of vectors (without normalization).
///
/// Useful for computing statistics or filtering before normalization.
///
/// # Arguments
///
/// * `data` - Flat array of vectors stored contiguously
/// * `dim` - Dimension of each vector
///
/// # Returns
///
/// Vector of L2 norms for each vector.
pub fn compute_norms_batch(data: &[f32], dim: usize) -> Vec<f32> {
    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    let n_vectors = data.len() / dim;
    let mut norms = Vec::with_capacity(n_vectors);

    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vector = &data[start..end];
        norms.push(norm(vector));
    }

    norms
}

/// Validate that all vectors are normalized (norm ≈ 1.0).
///
/// # Arguments
///
/// * `data` - Flat array of vectors stored contiguously
/// * `dim` - Dimension of each vector
/// * `tolerance` - Maximum allowed deviation from 1.0
///
/// # Returns
///
/// Vector of indices of vectors that are NOT normalized within tolerance.
/// Empty vector if all are normalized.
pub fn find_unnormalized(data: &[f32], dim: usize, tolerance: f32) -> Vec<usize> {
    assert!(dim > 0, "Dimension must be > 0");
    assert!(data.len() % dim == 0, "Data length must be multiple of dimension");

    let n_vectors = data.len() / dim;
    let mut unnormalized = Vec::new();

    for i in 0..n_vectors {
        let start = i * dim;
        let end = start + dim;
        let vector = &data[start..end];
        let n = norm(vector);
        if (n - 1.0).abs() > tolerance {
            unnormalized.push(i);
        }
    }

    unnormalized
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() < EPSILON, "Expected {} ≈ {}", a, b);
    }

    #[test]
    fn test_l2_distance_identical() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_approx_eq(l2_distance(&a, &a), 0.0);
    }

    #[test]
    fn test_l2_distance_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert_approx_eq(l2_distance(&a, &b), std::f32::consts::SQRT_2);
    }

    #[test]
    fn test_l2_distance_known_value() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert_approx_eq(l2_distance(&a, &b), 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert_approx_eq(inner_product(&a, &b), 0.0);
    }

    #[test]
    fn test_inner_product_parallel() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0];
        assert_approx_eq(inner_product(&a, &b), 28.0); // 2 + 8 + 18
    }

    #[test]
    fn test_inner_product_known_value() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        assert_approx_eq(inner_product(&a, &b), 20.0); // 4 + 6 + 6 + 4
    }

    #[test]
    fn test_cosine_identical() {
        let a = [1.0, 2.0, 3.0];
        assert_approx_eq(cosine_similarity(&a, &a), 1.0);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_approx_eq(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert_approx_eq(cosine_similarity(&a, &b), -1.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        assert_approx_eq(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert_approx_eq(cosine_distance(&a, &b), 0.0);

        let c = [1.0, 0.0];
        let d = [-1.0, 0.0];
        assert_approx_eq(cosine_distance(&c, &d), 2.0);
    }

    #[test]
    fn test_norm() {
        let a = [3.0, 4.0];
        assert_approx_eq(norm(&a), 5.0);

        let b = [0.0, 0.0, 0.0];
        assert_approx_eq(norm(&b), 0.0);
    }

    #[test]
    fn test_normalize() {
        let a = [3.0, 4.0];
        let n = normalize(&a);
        assert_approx_eq(n[0], 0.6);
        assert_approx_eq(n[1], 0.8);
        assert_approx_eq(norm(&n), 1.0);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let a = [0.0, 0.0];
        let n = normalize(&a);
        assert_approx_eq(n[0], 0.0);
        assert_approx_eq(n[1], 0.0);
    }

    #[test]
    fn test_normalize_in_place() {
        let mut a = [3.0, 4.0];
        let original_norm = normalize_in_place(&mut a);
        assert_approx_eq(original_norm, 5.0);
        assert_approx_eq(a[0], 0.6);
        assert_approx_eq(a[1], 0.8);
    }

    #[test]
    fn test_symmetry() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];

        assert_approx_eq(l2_distance(&a, &b), l2_distance(&b, &a));
        assert_approx_eq(inner_product(&a, &b), inner_product(&b, &a));
        assert_approx_eq(cosine_similarity(&a, &b), cosine_similarity(&b, &a));
    }

    #[test]
    fn test_high_dimension() {
        let dim = 512;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.01).collect();

        // Just verify no panics and reasonable values
        let l2 = l2_distance(&a, &b);
        let ip = inner_product(&a, &b);
        let cos = cosine_similarity(&a, &b);

        assert!(l2.is_finite());
        assert!(ip.is_finite());
        assert!(cos.is_finite());
        assert!(cos >= -1.0 && cos <= 1.0);
    }
}
