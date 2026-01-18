//! Vector Quantization Module
//!
//! Provides quantization methods for memory-efficient vector storage.
//!
//! # Quantization Methods
//!
//! - **SQ8 (Scalar Quantization)**: Quantizes each dimension to 8-bit integers.
//!   Provides 4x memory reduction with ~1-3% recall loss.
//!
//! - **PQ (Product Quantization)**: Divides vectors into subspaces and quantizes
//!   each subspace separately. Provides 32x-128x memory reduction.
//!
//! # Usage
//!
//! ```ignore
//! use rag_plusplus_core::quantization::{SQ8Quantizer, Quantizer};
//!
//! // Train quantizer on sample vectors
//! let vectors: Vec<Vec<f32>> = /* training data */;
//! let quantizer = SQ8Quantizer::train(&vectors);
//!
//! // Encode vectors
//! let encoded = quantizer.encode(&vector);
//!
//! // Compute distance (asymmetric: full precision query vs quantized corpus)
//! let distance = quantizer.asymmetric_l2(&query, &encoded);
//! ```
//!
//! # Performance Notes
//!
//! Asymmetric distance computation (ADC) keeps queries in full precision
//! while comparing against quantized corpus vectors. This provides better
//! accuracy than symmetric (both quantized) at minimal cost.

pub mod pq;
pub mod sq8;

pub use pq::{PQConfig, PQDistanceTable, PQEncodedVector, PQQuantizer};
pub use sq8::{SQ8EncodedVector, SQ8Quantizer};

/// Trait for vector quantizers.
pub trait Quantizer: Send + Sync {
    /// The encoded vector type.
    type Encoded;

    /// Train the quantizer on a set of vectors.
    fn train(vectors: &[Vec<f32>]) -> Self;

    /// Encode a single vector.
    fn encode(&self, vector: &[f32]) -> Self::Encoded;

    /// Encode multiple vectors.
    fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Self::Encoded> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Decode an encoded vector back to f32.
    fn decode(&self, encoded: &Self::Encoded) -> Vec<f32>;

    /// Vector dimension.
    fn dimension(&self) -> usize;

    /// Memory usage per encoded vector in bytes.
    fn encoded_size(&self) -> usize;

    /// Compression ratio (original_size / encoded_size).
    fn compression_ratio(&self) -> f32 {
        (self.dimension() * 4) as f32 / self.encoded_size() as f32
    }
}

/// Trait for asymmetric distance computation.
///
/// Asymmetric distance computes distance between a full-precision query
/// and a quantized corpus vector, providing better accuracy than symmetric.
pub trait AsymmetricDistance<E> {
    /// Compute L2 distance between full-precision query and encoded vector.
    fn asymmetric_l2(&self, query: &[f32], encoded: &E) -> f32;

    /// Compute inner product between full-precision query and encoded vector.
    fn asymmetric_inner_product(&self, query: &[f32], encoded: &E) -> f32;

    /// Compute cosine similarity between full-precision query and encoded vector.
    fn asymmetric_cosine(&self, query: &[f32], encoded: &E) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_ratio() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..128).map(|j| (i * j) as f32 * 0.01).collect())
            .collect();

        let quantizer = SQ8Quantizer::train(&vectors);

        // SQ8: 128 dimensions * 4 bytes = 512 bytes original
        // SQ8: 128 dimensions * 1 byte = 128 bytes encoded
        // Ratio: 512 / 128 = 4.0
        assert!((quantizer.compression_ratio() - 4.0).abs() < 0.01);
    }
}
