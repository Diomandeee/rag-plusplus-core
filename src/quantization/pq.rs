//! Product Quantization (PQ) - High Compression Quantization
//!
//! Divides vectors into M subspaces and quantizes each subspace using K centroids.
//! Provides 32x-128x memory reduction with ~5-10% recall loss.
//!
//! # Algorithm
//!
//! 1. **Training**: For each subspace m (m = 0..M):
//!    - Extract subvectors from training data
//!    - Run k-means to find K centroids
//!    - Store codebook[m] = {centroid_0, ..., centroid_{K-1}}
//!
//! 2. **Encoding**: For vector v:
//!    - Split v into M subvectors: v = [v_0, v_1, ..., v_{M-1}]
//!    - For each subvector, find nearest centroid: code[m] = argmin_k dist(v_m, codebook[m][k])
//!    - Return codes: [code_0, code_1, ..., code_{M-1}]
//!
//! 3. **Asymmetric Distance Computation (ADC)**:
//!    - Precompute distance table: table[m][k] = dist(query_m, codebook[m][k])
//!    - Distance = sum_m table[m][code[m]]
//!
//! # Memory Layout
//!
//! - Codebook: M * K * (dim/M) * 4 bytes
//! - Encoded vector: M bytes (1 byte per subspace, assuming K <= 256)
//!
//! # Performance Notes
//!
//! - ADC with precomputed tables is O(M) per distance, vs O(dim) for brute force
//! - For dim=512, M=16, K=256: 32x compression, O(16) per distance

use rand::prelude::*;

use super::{AsymmetricDistance, Quantizer};

/// Number of centroids per subspace (must be <= 256 for u8 codes).
pub const DEFAULT_K: usize = 256;

/// Product Quantizer.
#[derive(Debug, Clone)]
pub struct PQQuantizer {
    /// Vector dimension.
    dimension: usize,
    /// Number of subspaces.
    m: usize,
    /// Dimension of each subspace.
    sub_dim: usize,
    /// Number of centroids per subspace.
    k: usize,
    /// Codebooks: M × K × sub_dim centroids, stored flat.
    /// Layout: [subspace_0: k centroids, subspace_1: k centroids, ...]
    codebooks: Vec<f32>,
}

/// Encoded vector in PQ format.
#[derive(Debug, Clone)]
pub struct PQEncodedVector {
    /// Quantization codes, one per subspace.
    pub codes: Vec<u8>,
}

/// Precomputed distance table for efficient ADC.
///
/// Contains distances from query subvectors to all centroids.
/// table[m * k + c] = distance(query[m], centroid[m][c])
#[derive(Debug)]
pub struct PQDistanceTable {
    /// M × K distance values.
    distances: Vec<f32>,
    /// Number of subspaces.
    m: usize,
    /// Number of centroids per subspace.
    k: usize,
}

/// Configuration for PQ training.
#[derive(Debug, Clone)]
pub struct PQConfig {
    /// Number of subspaces.
    pub m: usize,
    /// Number of centroids per subspace.
    pub k: usize,
    /// Number of k-means iterations.
    pub kmeans_iters: usize,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            m: 16,
            k: 256,
            kmeans_iters: 25,
            seed: None,
        }
    }
}

impl PQQuantizer {
    /// Create a new PQ quantizer with pre-trained codebooks.
    pub fn new(dimension: usize, m: usize, k: usize, codebooks: Vec<f32>) -> Self {
        assert!(dimension % m == 0, "Dimension must be divisible by M");
        assert!(k <= 256, "K must be <= 256 for u8 codes");

        let sub_dim = dimension / m;
        assert_eq!(
            codebooks.len(),
            m * k * sub_dim,
            "Codebook size mismatch"
        );

        Self {
            dimension,
            m,
            sub_dim,
            k,
            codebooks,
        }
    }

    /// Train PQ quantizer on vectors with custom configuration.
    pub fn train_with_config(vectors: &[Vec<f32>], config: &PQConfig) -> Self {
        if vectors.is_empty() {
            return Self::new(0, config.m, config.k, vec![]);
        }

        let dimension = vectors[0].len();
        assert!(
            dimension % config.m == 0,
            "Dimension {} must be divisible by M {}",
            dimension,
            config.m
        );

        let sub_dim = dimension / config.m;
        let mut codebooks = Vec::with_capacity(config.m * config.k * sub_dim);

        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Train each subspace independently
        for subspace in 0..config.m {
            let start = subspace * sub_dim;
            let end = start + sub_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Run k-means to find centroids
            let centroids = kmeans(&subvectors, config.k, config.kmeans_iters, &mut rng);

            // Add centroids to codebook
            for centroid in centroids {
                codebooks.extend_from_slice(&centroid);
            }
        }

        Self::new(dimension, config.m, config.k, codebooks)
    }

    /// Get centroid for a subspace and code.
    #[inline]
    fn get_centroid(&self, subspace: usize, code: u8) -> &[f32] {
        let start = (subspace * self.k + code as usize) * self.sub_dim;
        &self.codebooks[start..start + self.sub_dim]
    }

    /// Find nearest centroid for a subvector.
    fn find_nearest_centroid(&self, subspace: usize, subvector: &[f32]) -> u8 {
        let mut best_code = 0u8;
        let mut best_dist = f32::INFINITY;

        for code in 0..self.k {
            let centroid = self.get_centroid(subspace, code as u8);
            let dist = l2_squared(subvector, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_code = code as u8;
            }
        }

        best_code
    }

    /// Compute precomputed distance table for a query.
    ///
    /// This table allows O(M) distance computation per encoded vector
    /// instead of O(dim).
    pub fn compute_distance_table(&self, query: &[f32]) -> PQDistanceTable {
        debug_assert_eq!(query.len(), self.dimension);

        let mut distances = Vec::with_capacity(self.m * self.k);

        for subspace in 0..self.m {
            let query_sub = &query[subspace * self.sub_dim..(subspace + 1) * self.sub_dim];

            for code in 0..self.k {
                let centroid = self.get_centroid(subspace, code as u8);
                let dist = l2_squared(query_sub, centroid);
                distances.push(dist);
            }
        }

        PQDistanceTable {
            distances,
            m: self.m,
            k: self.k,
        }
    }

    /// Compute asymmetric L2 distance using precomputed table.
    ///
    /// This is O(M) instead of O(dim).
    #[inline]
    pub fn asymmetric_l2_with_table(&self, table: &PQDistanceTable, encoded: &PQEncodedVector) -> f32 {
        debug_assert_eq!(encoded.codes.len(), self.m);

        let mut sum = 0.0f32;
        for (subspace, &code) in encoded.codes.iter().enumerate() {
            sum += table.get(subspace, code);
        }
        sum.sqrt()
    }

    /// Compute asymmetric L2 squared distance using precomputed table.
    #[inline]
    pub fn asymmetric_l2_squared_with_table(&self, table: &PQDistanceTable, encoded: &PQEncodedVector) -> f32 {
        debug_assert_eq!(encoded.codes.len(), self.m);

        let mut sum = 0.0f32;
        for (subspace, &code) in encoded.codes.iter().enumerate() {
            sum += table.get(subspace, code);
        }
        sum
    }

    /// Number of subspaces.
    #[must_use]
    pub fn num_subspaces(&self) -> usize {
        self.m
    }

    /// Number of centroids per subspace.
    #[must_use]
    pub fn num_centroids(&self) -> usize {
        self.k
    }

    /// Subspace dimension.
    #[must_use]
    pub fn sub_dimension(&self) -> usize {
        self.sub_dim
    }

    /// Codebook memory size in bytes.
    #[must_use]
    pub fn codebook_size(&self) -> usize {
        self.codebooks.len() * 4
    }
}

impl PQDistanceTable {
    /// Get precomputed distance for subspace and code.
    #[inline]
    pub fn get(&self, subspace: usize, code: u8) -> f32 {
        self.distances[subspace * self.k + code as usize]
    }
}

impl Quantizer for PQQuantizer {
    type Encoded = PQEncodedVector;

    fn train(vectors: &[Vec<f32>]) -> Self {
        Self::train_with_config(vectors, &PQConfig::default())
    }

    fn encode(&self, vector: &[f32]) -> PQEncodedVector {
        debug_assert_eq!(vector.len(), self.dimension);

        let codes: Vec<u8> = (0..self.m)
            .map(|subspace| {
                let subvector = &vector[subspace * self.sub_dim..(subspace + 1) * self.sub_dim];
                self.find_nearest_centroid(subspace, subvector)
            })
            .collect();

        PQEncodedVector { codes }
    }

    fn decode(&self, encoded: &PQEncodedVector) -> Vec<f32> {
        debug_assert_eq!(encoded.codes.len(), self.m);

        let mut result = Vec::with_capacity(self.dimension);
        for (subspace, &code) in encoded.codes.iter().enumerate() {
            let centroid = self.get_centroid(subspace, code);
            result.extend_from_slice(centroid);
        }
        result
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn encoded_size(&self) -> usize {
        self.m // 1 byte per subspace
    }
}

impl AsymmetricDistance<PQEncodedVector> for PQQuantizer {
    fn asymmetric_l2(&self, query: &[f32], encoded: &PQEncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);
        debug_assert_eq!(encoded.codes.len(), self.m);

        let mut sum = 0.0f32;
        for (subspace, &code) in encoded.codes.iter().enumerate() {
            let query_sub = &query[subspace * self.sub_dim..(subspace + 1) * self.sub_dim];
            let centroid = self.get_centroid(subspace, code);
            sum += l2_squared(query_sub, centroid);
        }
        sum.sqrt()
    }

    fn asymmetric_inner_product(&self, query: &[f32], encoded: &PQEncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);
        debug_assert_eq!(encoded.codes.len(), self.m);

        let mut sum = 0.0f32;
        for (subspace, &code) in encoded.codes.iter().enumerate() {
            let query_sub = &query[subspace * self.sub_dim..(subspace + 1) * self.sub_dim];
            let centroid = self.get_centroid(subspace, code);
            sum += inner_product(query_sub, centroid);
        }
        sum
    }

    fn asymmetric_cosine(&self, query: &[f32], encoded: &PQEncodedVector) -> f32 {
        debug_assert_eq!(query.len(), self.dimension);
        debug_assert_eq!(encoded.codes.len(), self.m);

        // Decode and compute cosine
        let decoded = self.decode(encoded);

        let mut dot = 0.0f32;
        let mut norm_q_sq = 0.0f32;
        let mut norm_d_sq = 0.0f32;

        for (&q, &d) in query.iter().zip(decoded.iter()) {
            dot += q * d;
            norm_q_sq += q * q;
            norm_d_sq += d * d;
        }

        let norm_q = norm_q_sq.sqrt();
        let norm_d = norm_d_sq.sqrt();

        if norm_q == 0.0 || norm_d == 0.0 {
            0.0
        } else {
            dot / (norm_q * norm_d)
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute squared L2 distance.
#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute inner product.
#[inline]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// K-means clustering for codebook training.
fn kmeans(vectors: &[Vec<f32>], k: usize, max_iters: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return vec![];
    }

    let n = vectors.len();
    let dim = vectors[0].len();

    // Handle case where we have fewer vectors than centroids
    let actual_k = k.min(n);

    // Initialize centroids using k-means++ style initialization
    let mut centroids = kmeans_plusplus_init(vectors, actual_k, rng);

    // Pad with random vectors if needed
    while centroids.len() < k {
        let random_idx = rng.gen_range(0..n);
        centroids.push(vectors[random_idx].clone());
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iters {
        // Assignment step
        let mut changed = false;
        for (i, vector) in vectors.iter().enumerate() {
            let mut best_centroid = 0;
            let mut best_dist = f32::INFINITY;

            for (c, centroid) in centroids.iter().enumerate() {
                let dist = l2_squared(vector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = c;
                }
            }

            if assignments[i] != best_centroid {
                assignments[i] = best_centroid;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, vector) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for (j, &v) in vector.iter().enumerate() {
                new_centroids[c][j] += v;
            }
        }

        for (c, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[c] > 0 {
                for v in centroid.iter_mut() {
                    *v /= counts[c] as f32;
                }
            } else {
                // Empty cluster: reinitialize with random vector
                let random_idx = rng.gen_range(0..n);
                *centroid = vectors[random_idx].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// K-means++ style initialization.
fn kmeans_plusplus_init(vectors: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let n = vectors.len();
    if n == 0 || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random
    let first_idx = rng.gen_range(0..n);
    centroids.push(vectors[first_idx].clone());

    // Remaining centroids: probability proportional to squared distance
    let mut distances = vec![f32::INFINITY; n];

    while centroids.len() < k && centroids.len() < n {
        // Update distances to nearest centroid
        let last_centroid = centroids.last().unwrap();
        for (i, vector) in vectors.iter().enumerate() {
            let dist = l2_squared(vector, last_centroid);
            distances[i] = distances[i].min(dist);
        }

        // Sample proportional to squared distance
        let total: f32 = distances.iter().sum();
        if total == 0.0 {
            break;
        }

        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0f32;
        let mut chosen = 0;

        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }

        centroids.push(vectors[chosen].clone());
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.1; // PQ has higher error than SQ8

    fn random_test_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect()
    }

    #[test]
    fn test_pq_train() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 16,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        assert_eq!(quantizer.dimension(), 64);
        assert_eq!(quantizer.num_subspaces(), 8);
        assert_eq!(quantizer.num_centroids(), 16);
        assert_eq!(quantizer.sub_dimension(), 8);
    }

    #[test]
    fn test_pq_encode_decode() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 32,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        // Test roundtrip
        for vector in vectors.iter().take(10) {
            let encoded = quantizer.encode(vector);
            assert_eq!(encoded.codes.len(), 8);

            let decoded = quantizer.decode(&encoded);
            assert_eq!(decoded.len(), 64);

            // PQ is lossy - with small K and random data, reconstruction error is higher
            let dist = l2_squared(vector, &decoded).sqrt();
            // Normalized vectors have norm ~1, so error < 2 is reasonable for small K
            assert!(
                dist < 2.0,
                "Reconstruction error too high: {}",
                dist
            );
        }
    }

    #[test]
    fn test_pq_compression_ratio() {
        let vectors = random_test_vectors(1000, 512, 42);
        let config = PQConfig {
            m: 16,
            k: 256,
            kmeans_iters: 5,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        // 512 * 4 = 2048 bytes original
        // 16 bytes encoded
        // Compression: 128x
        assert_eq!(quantizer.encoded_size(), 16);
        let ratio = quantizer.compression_ratio();
        assert!(
            (ratio - 128.0).abs() < 0.01,
            "Expected 128x compression, got {}x",
            ratio
        );
    }

    #[test]
    fn test_pq_asymmetric_l2() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 32,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        let query = &vectors[0];
        let encoded = quantizer.encode(query);

        // Self-distance should be reasonable (PQ is more lossy than SQ8)
        let dist = quantizer.asymmetric_l2(query, &encoded);
        assert!(
            dist < 2.0,
            "Self-distance should be reasonable, got {}",
            dist
        );
    }

    #[test]
    fn test_pq_distance_table() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 32,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        let query = &vectors[0];
        let encoded = quantizer.encode(&vectors[1]);

        // Compare table-based vs direct computation
        let direct = quantizer.asymmetric_l2(query, &encoded);
        let table = quantizer.compute_distance_table(query);
        let table_based = quantizer.asymmetric_l2_with_table(&table, &encoded);

        assert!(
            (direct - table_based).abs() < 1e-5,
            "Table-based distance should match direct: {} vs {}",
            direct,
            table_based
        );
    }

    #[test]
    fn test_pq_asymmetric_inner_product() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 32,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        let query = &vectors[0];
        let encoded = quantizer.encode(query);

        let ip = quantizer.asymmetric_inner_product(query, &encoded);
        // For similar vectors, IP should be positive
        assert!(ip > 0.0, "Self IP should be positive, got {}", ip);
    }

    #[test]
    fn test_pq_asymmetric_cosine() {
        let vectors = random_test_vectors(100, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 32,
            kmeans_iters: 10,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        let query = &vectors[0];
        let encoded = quantizer.encode(query);

        let cos = quantizer.asymmetric_cosine(query, &encoded);
        // Self-cosine should be near 1.0
        assert!(
            cos > 0.8,
            "Self-cosine should be high, got {}",
            cos
        );
    }

    #[test]
    fn test_pq_empty_vectors() {
        let vectors: Vec<Vec<f32>> = vec![];
        let quantizer = PQQuantizer::train(&vectors);
        assert_eq!(quantizer.dimension(), 0);
    }

    #[test]
    fn test_pq_few_vectors() {
        // Test with fewer vectors than centroids
        let vectors = random_test_vectors(5, 64, 42);
        let config = PQConfig {
            m: 8,
            k: 256, // More centroids than vectors
            kmeans_iters: 5,
            seed: Some(42),
        };

        let quantizer = PQQuantizer::train_with_config(&vectors, &config);

        // Should still work
        let encoded = quantizer.encode(&vectors[0]);
        let decoded = quantizer.decode(&encoded);
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn test_kmeans() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create 2 clear clusters
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                if i < 25 {
                    vec![0.0, 0.0]
                } else {
                    vec![10.0, 10.0]
                }
            })
            .collect();

        let centroids = kmeans(&vectors, 2, 10, &mut rng);

        assert_eq!(centroids.len(), 2);

        // Centroids should be near [0, 0] and [10, 10]
        let mut found_origin = false;
        let mut found_corner = false;

        for centroid in &centroids {
            if centroid[0] < 1.0 && centroid[1] < 1.0 {
                found_origin = true;
            }
            if centroid[0] > 9.0 && centroid[1] > 9.0 {
                found_corner = true;
            }
        }

        assert!(found_origin, "Should find centroid near origin");
        assert!(found_corner, "Should find centroid near (10, 10)");
    }
}
