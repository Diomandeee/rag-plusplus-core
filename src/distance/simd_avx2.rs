//! AVX2-accelerated Distance Implementations (x86_64)
//!
//! SIMD-optimized distance functions using AVX2 intrinsics.
//! These provide 4-8x speedup over scalar implementations for large vectors.
//!
//! # Safety
//!
//! All functions in this module use `unsafe` for SIMD intrinsics.
//! They are marked with `#[target_feature(enable = "avx2", enable = "fma")]`
//! which means they can only be called on CPUs that support these features.
//! The runtime dispatch in `mod.rs` ensures correct feature detection.
//!
//! # Performance Notes
//!
//! - AVX2 processes 8 f32 values per instruction (256-bit registers)
//! - FMA (fused multiply-add) reduces latency for dot products
//! - Vectors should be 32-byte aligned for optimal performance (not enforced)

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute L2 (Euclidean) distance using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
/// Use `is_x86_feature_detected!("avx2")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared_avx2(a, b).sqrt()
}

/// Compute squared L2 distance using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    // Process 8 elements at a time
    let chunks = len / 8;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));

        // diff = a - b
        let diff = _mm256_sub_ps(va, vb);

        // sum += diff * diff (using FMA: sum = diff * diff + sum)
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum of the 8 lanes
    let mut result = horizontal_sum_avx2(sum);

    // Handle remaining elements (scalar)
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// Compute inner product (dot product) using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));

        // sum += a * b (FMA: sum = a * b + sum)
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result += a[i] * b[i];
    }

    result
}

/// Compute cosine similarity using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    let chunks = len / 8;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));

        // dot_sum += a * b
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);

        // norm_a_sum += a * a
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);

        // norm_b_sum += b * b
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    let mut dot = horizontal_sum_avx2(dot_sum);
    let mut norm_a_sq = horizontal_sum_avx2(norm_a_sum);
    let mut norm_b_sq = horizontal_sum_avx2(norm_b_sum);

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute cosine distance (1 - cosine_similarity) using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity_avx2(a, b)
}

/// Compute L2 norm using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn norm_avx2(a: &[f32]) -> f32 {
    norm_squared_avx2(a).sqrt()
}

/// Compute squared L2 norm using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn norm_squared_avx2(a: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    let a_ptr = a.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, va, sum);
    }

    let mut result = horizontal_sum_avx2(sum);

    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        result += a[i] * a[i];
    }

    result
}

/// Normalize a vector in-place using AVX2.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 and FMA instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn normalize_in_place_avx2(a: &mut [f32]) -> f32 {
    let n = norm_avx2(a);
    if n > 0.0 {
        let inv_n = 1.0 / n;
        let inv_n_vec = _mm256_set1_ps(inv_n);

        let len = a.len();
        let chunks = len / 8;
        let a_ptr = a.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let normalized = _mm256_mul_ps(va, inv_n_vec);
            _mm256_storeu_ps(a_ptr.add(offset), normalized);
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..len {
            a[i] *= inv_n;
        }
    }
    n
}

/// Horizontal sum of 8 f32 values in an AVX2 register.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2 instructions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit halves
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);

    // Add the two halves
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128-bit register
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf); // [0+1,1+1,2+3,3+3]
    let shuf2 = _mm_movehl_ps(sums, sums); // [2+3,3+3,2+3,3+3]
    let final_sum = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(final_sum)
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_approx_eq(a: f32, b: f32) {
        assert!(
            (a - b).abs() < EPSILON,
            "Expected {} ≈ {}, diff = {}",
            a,
            b,
            (a - b).abs()
        );
    }

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[test]
    fn test_l2_distance_avx2() {
        if !has_avx2() {
            println!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let dist = unsafe { l2_distance_avx2(&a, &b) };
        assert_approx_eq(dist, std::f32::consts::SQRT_2);
    }

    #[test]
    fn test_l2_distance_avx2_identical() {
        if !has_avx2() {
            return;
        }

        let a: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let dist = unsafe { l2_distance_avx2(&a, &a) };
        assert_approx_eq(dist, 0.0);
    }

    #[test]
    fn test_inner_product_avx2() {
        if !has_avx2() {
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let ip = unsafe { inner_product_avx2(&a, &b) };
        // 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1 = 8+14+18+20+20+18+14+8 = 120
        assert_approx_eq(ip, 120.0);
    }

    #[test]
    fn test_inner_product_avx2_orthogonal() {
        if !has_avx2() {
            return;
        }

        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let ip = unsafe { inner_product_avx2(&a, &b) };
        assert_approx_eq(ip, 0.0);
    }

    #[test]
    fn test_cosine_similarity_avx2_identical() {
        if !has_avx2() {
            return;
        }

        let a: Vec<f32> = (0..512).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let cos = unsafe { cosine_similarity_avx2(&a, &a) };
        assert_approx_eq(cos, 1.0);
    }

    #[test]
    fn test_cosine_similarity_avx2_orthogonal() {
        if !has_avx2() {
            return;
        }

        let mut a = vec![0.0; 16];
        let mut b = vec![0.0; 16];
        a[0] = 1.0;
        b[1] = 1.0;

        let cos = unsafe { cosine_similarity_avx2(&a, &b) };
        assert_approx_eq(cos, 0.0);
    }

    #[test]
    fn test_norm_avx2() {
        if !has_avx2() {
            return;
        }

        let a = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let n = unsafe { norm_avx2(&a) };
        assert_approx_eq(n, 5.0);
    }

    #[test]
    fn test_normalize_in_place_avx2() {
        if !has_avx2() {
            return;
        }

        let mut a = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let original_norm = unsafe { normalize_in_place_avx2(&mut a) };

        assert_approx_eq(original_norm, 5.0);
        assert_approx_eq(a[0], 0.6);
        assert_approx_eq(a[1], 0.8);

        // Verify normalized
        let new_norm = unsafe { norm_avx2(&a) };
        assert_approx_eq(new_norm, 1.0);
    }

    #[test]
    fn test_consistency_with_scalar() {
        if !has_avx2() {
            return;
        }

        use crate::distance::scalar;

        // Test with various sizes including non-multiples of 8
        for size in [7, 8, 15, 16, 31, 127, 256, 511, 512, 513, 1024] {
            let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.2).cos()).collect();

            let l2_scalar = scalar::l2_distance(&a, &b);
            let l2_avx2 = unsafe { l2_distance_avx2(&a, &b) };
            assert!(
                (l2_scalar - l2_avx2).abs() < 1e-4,
                "L2 mismatch at size {}: scalar={}, avx2={}",
                size,
                l2_scalar,
                l2_avx2
            );

            let ip_scalar = scalar::inner_product(&a, &b);
            let ip_avx2 = unsafe { inner_product_avx2(&a, &b) };
            assert!(
                (ip_scalar - ip_avx2).abs() < 1e-4,
                "IP mismatch at size {}: scalar={}, avx2={}",
                size,
                ip_scalar,
                ip_avx2
            );

            let cos_scalar = scalar::cosine_similarity(&a, &b);
            let cos_avx2 = unsafe { cosine_similarity_avx2(&a, &b) };
            assert!(
                (cos_scalar - cos_avx2).abs() < 1e-4,
                "Cosine mismatch at size {}: scalar={}, avx2={}",
                size,
                cos_scalar,
                cos_avx2
            );
        }
    }

    #[test]
    fn test_high_dimension() {
        if !has_avx2() {
            return;
        }

        let dim = 1536; // Common embedding dimension
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();

        let l2 = unsafe { l2_distance_avx2(&a, &b) };
        let ip = unsafe { inner_product_avx2(&a, &b) };
        let cos = unsafe { cosine_similarity_avx2(&a, &b) };

        assert!(l2.is_finite());
        assert!(ip.is_finite());
        assert!(cos.is_finite());
        assert!(cos >= -1.0 && cos <= 1.0);
    }
}
