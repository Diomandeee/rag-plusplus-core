//! Scalar Quantization (SQ8) - 8-bit Quantization
//!
//! Quantizes each vector dimension independently to an 8-bit signed integer.
//! Provides 4x memory reduction with typically 1-3% recall loss.
//!
//! # Algorithm
//!
//! For each dimension d:
//! 1. Compute min and max values across training vectors
//! 2. Scale = (max - min) / 255
//! 3. Encode: q[d] = round((v[d] - min[d]) / scale[d])
//! 4. Decode: v[d] = q[d] * scale[d] + min[d]
//!
//! # Memory Layout
//!
//! - Quantization parameters: 2 * dim * 4 bytes (min + scale per dimension)
//! - Encoded vector: dim bytes (1 byte per dimension)
//!
//! # Performance
//!
//! - Encoding: O(dim)
//! - Decoding: O(dim)
//! - Asymmetric distance: O(dim) - same as full precision, but with cache benefits

use super::{AsymmetricDistance, Quantizer};

/// SQ8 Quantizer - Scalar Quantization to 8-bit integers.
#[derive(Debug, Clone)]
pub struct SQ8Quantizer {
    /// Vector dimension
    dimension: usize,
    /// Minimum value per dimension
    mins: Vec<f32>,
    /// Scale factor per dimension: (max - min) / 255
    scales: Vec<f32>,
    /// Inverse scales for decoding: 1 / scale (precomputed for speed)
    inv_scales: Vec<f32>,
}

/// Encoded vector in SQ8 format.
#[derive(Debug, Clone)]
pub struct SQ8EncodedVector {
    /// Quantized values (0-255 stored as u8)
    pub codes: Vec<u8>,
}

impl SQ8Quantizer {
    /// Create a new SQ8 quantizer with given parameters.
    pub fn new(mins: Vec<f32>, scales: Vec<f32>) -> Self {
        let dimension = mins.len();
        debug_assert_eq!(scales.len(), dimension);

        let inv_scales: Vec<f32> = scales
            .iter()
            .map(|&s| if s > 0.0 { 1.0 / s } else { 0.0 })
            .collect();

        Self {
            dimension,
            mins,
            scales,
            inv_scales,
        }
    }

    /// Create a quantizer from min/max bounds.
    pub fn from_bounds(mins: Vec<f32>, maxs: Vec<f32>) -> Self {
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| {
                let range = max - min;
                if range > 0.0 {
                    range / 255.0
                } else {
                    1.0 // Prevent division by zero for constant dimensions
                }
            })
            .collect();

        Self::new(mins, scales)
    }

    /// Get the minimum values per dimension.
    #[must_use]
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Get the scale factors per dimension.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Encode a single dimension value.
    #[inline]
    fn encode_dim(&self, value: f32, dim: usize) -> u8 {
        let normalized = (value - self.mins[dim]) * self.inv_scales[dim];
        // Clamp to [0, 255] and round
        normalized.clamp(0.0, 255.0).round() as u8
    }

    /// Decode a single dimension value.
    #[inline]
    fn decode_dim(&self, code: u8, dim: usize) -> f32 {
        f32::from(code) * self.scales[dim] + self.mins[dim]
    }

    /// Compute squared L2 distance between query and encoded vector.
    ///
    /// This is the core asymmetric distance computation.
    #[inline]
    pub fn asymmetric_l2_squared(&self, query: &[f32], encoded: &SQ8EncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);
        debug_assert_eq!(encoded.codes.len(), self.dimension);

        let mut sum = 0.0f32;
        for (i, (&q, &code)) in query.iter().zip(encoded.codes.iter()).enumerate() {
            let decoded = self.decode_dim(code, i);
            let diff = q - decoded;
            sum += diff * diff;
        }
        sum
    }
}

impl Quantizer for SQ8Quantizer {
    type Encoded = SQ8EncodedVector;

    fn train(vectors: &[Vec<f32>]) -> Self {
        if vectors.is_empty() {
            return Self::new(vec![], vec![]);
        }

        let dimension = vectors[0].len();

        // Compute min and max for each dimension
        let mut mins = vec![f32::INFINITY; dimension];
        let mut maxs = vec![f32::NEG_INFINITY; dimension];

        for vector in vectors {
            debug_assert_eq!(vector.len(), dimension, "All vectors must have same dimension");

            for (i, &v) in vector.iter().enumerate() {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }

        Self::from_bounds(mins, maxs)
    }

    fn encode(&self, vector: &[f32]) -> SQ8EncodedVector {
        debug_assert_eq!(vector.len(), self.dimension);

        let codes: Vec<u8> = vector
            .iter()
            .enumerate()
            .map(|(i, &v)| self.encode_dim(v, i))
            .collect();

        SQ8EncodedVector { codes }
    }

    fn decode(&self, encoded: &SQ8EncodedVector) -> Vec<f32> {
        debug_assert_eq!(encoded.codes.len(), self.dimension);

        encoded
            .codes
            .iter()
            .enumerate()
            .map(|(i, &code)| self.decode_dim(code, i))
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.dimension // 1 byte per dimension
    }
}

impl AsymmetricDistance<SQ8EncodedVector> for SQ8Quantizer {
    fn asymmetric_l2(&self, query: &[f32], encoded: &SQ8EncodedVector) -> f32 {
        self.asymmetric_l2_squared(query, encoded).sqrt()
    }

    fn asymmetric_inner_product(&self, query: &[f32], encoded: &SQ8EncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);
        debug_assert_eq!(encoded.codes.len(), self.dimension);

        let mut sum = 0.0f32;
        for (i, (&q, &code)) in query.iter().zip(encoded.codes.iter()).enumerate() {
            let decoded = self.decode_dim(code, i);
            sum += q * decoded;
        }
        sum
    }

    fn asymmetric_cosine(&self, query: &[f32], encoded: &SQ8EncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);

        let mut dot = 0.0f32;
        let mut norm_q_sq = 0.0f32;
        let mut norm_e_sq = 0.0f32;

        for (i, (&q, &code)) in query.iter().zip(encoded.codes.iter()).enumerate() {
            let decoded = self.decode_dim(code, i);
            dot += q * decoded;
            norm_q_sq += q * q;
            norm_e_sq += decoded * decoded;
        }

        let norm_q = norm_q_sq.sqrt();
        let norm_e = norm_e_sq.sqrt();

        if norm_q == 0.0 || norm_e == 0.0 {
            0.0
        } else {
            dot / (norm_q * norm_e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.05; // Quantization introduces some error

    fn assert_approx_eq(a: f32, b: f32, msg: &str) {
        assert!(
            (a - b).abs() < EPSILON,
            "{}: Expected {} ≈ {}, diff = {}",
            msg,
            a,
            b,
            (a - b).abs()
        );
    }

    fn create_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![0.0, 0.5, 1.0],
            vec![0.1, 0.6, 0.9],
            vec![0.2, 0.4, 0.8],
            vec![-0.1, 0.3, 1.1],
        ]
    }

    #[test]
    fn test_train_and_encode() {
        let vectors = create_test_vectors();
        let quantizer = SQ8Quantizer::train(&vectors);

        assert_eq!(quantizer.dimension(), 3);

        // Mins should be approximately [-0.1, 0.3, 0.8]
        assert!(quantizer.mins()[0] <= 0.0);
        assert!(quantizer.mins()[1] <= 0.4);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let vectors = create_test_vectors();
        let quantizer = SQ8Quantizer::train(&vectors);

        for vector in &vectors {
            let encoded = quantizer.encode(vector);
            let decoded = quantizer.decode(&encoded);

            for (i, (&original, &reconstructed)) in vector.iter().zip(decoded.iter()).enumerate() {
                assert_approx_eq(
                    original,
                    reconstructed,
                    &format!("Roundtrip dim {}", i),
                );
            }
        }
    }

    #[test]
    fn test_asymmetric_l2() {
        let vectors = create_test_vectors();
        let quantizer = SQ8Quantizer::train(&vectors);

        let query = vec![0.0, 0.5, 1.0];
        let encoded = quantizer.encode(&query);

        // Distance to self should be approximately 0
        let dist = quantizer.asymmetric_l2(&query, &encoded);
        assert!(dist < 0.1, "Self-distance should be near 0, got {}", dist);
    }

    #[test]
    fn test_asymmetric_inner_product() {
        // Use vectors that cover the test range
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let quantizer = SQ8Quantizer::train(&vectors);

        let query = vec![1.0, 0.0, 0.0];
        let vector = vec![1.0, 0.0, 0.0];
        let encoded = quantizer.encode(&vector);

        // Parallel vectors should have high inner product
        let ip = quantizer.asymmetric_inner_product(&query, &encoded);
        assert!(ip > 0.9, "Parallel vectors should have IP near 1.0, got {}", ip);
    }

    #[test]
    fn test_asymmetric_cosine() {
        // Use vectors that cover the test range
        let vectors = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let quantizer = SQ8Quantizer::train(&vectors);

        let query = vec![1.0, 0.0, 0.0];
        let encoded_parallel = quantizer.encode(&vec![1.0, 0.0, 0.0]);
        let encoded_ortho = quantizer.encode(&vec![0.0, 1.0, 0.0]);

        let cos_parallel = quantizer.asymmetric_cosine(&query, &encoded_parallel);
        let cos_ortho = quantizer.asymmetric_cosine(&query, &encoded_ortho);

        assert!(cos_parallel > 0.9, "Parallel should have cosine near 1.0");
        assert!(cos_ortho.abs() < 0.1, "Orthogonal should have cosine near 0.0");
    }

    #[test]
    fn test_compression_ratio() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..512).map(|j| (i * j) as f32 * 0.001).collect())
            .collect();

        let quantizer = SQ8Quantizer::train(&vectors);

        // 512 * 4 bytes = 2048 bytes original
        // 512 * 1 byte = 512 bytes encoded
        assert_eq!(quantizer.encoded_size(), 512);
        assert!((quantizer.compression_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_vectors() {
        let vectors: Vec<Vec<f32>> = vec![];
        let quantizer = SQ8Quantizer::train(&vectors);
        assert_eq!(quantizer.dimension(), 0);
    }

    #[test]
    fn test_constant_dimension() {
        // All vectors have same value in dimension 0
        let vectors = vec![
            vec![5.0, 1.0, 2.0],
            vec![5.0, 2.0, 3.0],
            vec![5.0, 3.0, 4.0],
        ];

        let quantizer = SQ8Quantizer::train(&vectors);

        // Should handle constant dimension gracefully
        let encoded = quantizer.encode(&vec![5.0, 2.0, 3.0]);
        let decoded = quantizer.decode(&encoded);

        assert!((decoded[0] - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_high_dimension() {
        let dim = 1536; // Common embedding dimension
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..dim).map(|j| ((i * j) as f32 * 0.0001).sin()).collect())
            .collect();

        let quantizer = SQ8Quantizer::train(&vectors);

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).cos()).collect();
        let encoded = quantizer.encode(&query);

        let l2 = quantizer.asymmetric_l2(&query, &encoded);
        let ip = quantizer.asymmetric_inner_product(&query, &encoded);
        let cos = quantizer.asymmetric_cosine(&query, &encoded);

        assert!(l2.is_finite());
        assert!(ip.is_finite());
        assert!(cos.is_finite() && cos >= -1.0 && cos <= 1.0);
    }
}
