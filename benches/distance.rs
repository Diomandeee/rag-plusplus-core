//! Distance Computation Benchmarks for RAG++
//!
//! Benchmarks for vector distance/similarity functions:
//! - L2 (Euclidean) distance
//! - Inner product
//! - Cosine similarity
//! - Normalization operations
//!
//! Compares:
//! - Scalar (pure Rust) implementations
//! - SIMD-dispatched (AVX2 when available) implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

use rag_plusplus_core::{
    // Scalar functions
    l2_distance, l2_distance_squared, inner_product,
    cosine_similarity, cosine_distance, norm, normalize, normalize_in_place,
    // SIMD-dispatched functions
    l2_distance_fast, inner_product_fast, cosine_similarity_fast,
    norm_fast, normalize_in_place_fast,
    // Dispatch functions
    compute_distance, compute_distance_for_heap, DistanceType,
};

/// Generate a random normalized vector of given dimension.
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.into_iter().map(|x| x / n).collect()
    } else {
        v
    }
}

/// Generate a pair of random vectors for distance computation.
fn random_vector_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
    (random_vector(dim), random_vector(dim))
}

// =============================================================================
// L2 DISTANCE BENCHMARKS
// =============================================================================

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/l2");

    for dim in [64, 128, 256, 512, 768, 1024, 1536].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| l2_distance(black_box(&a), black_box(&b)))
        });

        // SIMD-dispatched (AVX2 on x86_64, scalar fallback on ARM)
        group.bench_with_input(BenchmarkId::new("simd", dim), dim, |bench, _| {
            bench.iter(|| l2_distance_fast(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_l2_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/l2_squared");

    for dim in [128, 512, 1024].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| l2_distance_squared(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// INNER PRODUCT BENCHMARKS
// =============================================================================

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/inner_product");

    for dim in [64, 128, 256, 512, 768, 1024, 1536].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| inner_product(black_box(&a), black_box(&b)))
        });

        // SIMD-dispatched
        group.bench_with_input(BenchmarkId::new("simd", dim), dim, |bench, _| {
            bench.iter(|| inner_product_fast(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// COSINE SIMILARITY BENCHMARKS
// =============================================================================

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/cosine");

    for dim in [64, 128, 256, 512, 768, 1024, 1536].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| cosine_similarity(black_box(&a), black_box(&b)))
        });

        // SIMD-dispatched
        group.bench_with_input(BenchmarkId::new("simd", dim), dim, |bench, _| {
            bench.iter(|| cosine_similarity_fast(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/cosine_dist");

    for dim in [128, 512, 1024].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

// =============================================================================
// NORMALIZATION BENCHMARKS
// =============================================================================

fn bench_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/norm");

    for dim in [128, 512, 1024].iter() {
        let a = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| norm(black_box(&a)))
        });

        // SIMD-dispatched
        group.bench_with_input(BenchmarkId::new("simd", dim), dim, |bench, _| {
            bench.iter(|| norm_fast(black_box(&a)))
        });
    }

    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/normalize");

    for dim in [128, 512, 1024].iter() {
        let a = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("scalar_copy", dim), dim, |bench, _| {
            bench.iter(|| normalize(black_box(&a)))
        });
    }

    group.finish();
}

fn bench_normalize_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/normalize_inplace");

    for dim in [128, 512, 1024].iter() {
        let template = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        // Scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter_batched(
                || template.clone(),
                |mut v| {
                    normalize_in_place(black_box(&mut v));
                    v
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // SIMD-dispatched
        group.bench_with_input(BenchmarkId::new("simd", dim), dim, |bench, _| {
            bench.iter_batched(
                || template.clone(),
                |mut v| {
                    normalize_in_place_fast(black_box(&mut v));
                    v
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// =============================================================================
// DISPATCH FUNCTION BENCHMARKS
// =============================================================================

fn bench_compute_distance_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/dispatch");

    let dim = 512;
    let (a, b) = random_vector_pair(dim);

    for distance_type in [DistanceType::L2, DistanceType::InnerProduct, DistanceType::Cosine] {
        let name = match distance_type {
            DistanceType::L2 => "L2",
            DistanceType::InnerProduct => "IP",
            DistanceType::Cosine => "Cosine",
        };

        group.bench_function(name, |bench| {
            bench.iter(|| compute_distance(black_box(&a), black_box(&b), distance_type))
        });
    }

    group.finish();
}

fn bench_compute_distance_for_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/heap_dispatch");

    let dim = 512;
    let (a, b) = random_vector_pair(dim);

    for distance_type in [DistanceType::L2, DistanceType::InnerProduct, DistanceType::Cosine] {
        let name = match distance_type {
            DistanceType::L2 => "L2",
            DistanceType::InnerProduct => "IP",
            DistanceType::Cosine => "Cosine",
        };

        group.bench_function(name, |bench| {
            bench.iter(|| compute_distance_for_heap(black_box(&a), black_box(&b), distance_type))
        });
    }

    group.finish();
}

// =============================================================================
// BATCH DISTANCE BENCHMARKS
// =============================================================================

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/batch");

    let dim = 512;
    let query = random_vector(dim);

    for batch_size in [10, 100, 1000].iter() {
        let vectors: Vec<Vec<f32>> = (0..*batch_size).map(|_| random_vector(dim)).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        // L2 batch
        group.bench_with_input(
            BenchmarkId::new("l2", batch_size),
            batch_size,
            |bench, _| {
                bench.iter(|| {
                    vectors.iter().map(|v| l2_distance(black_box(&query), black_box(v))).collect::<Vec<_>>()
                })
            },
        );

        // Inner product batch
        group.bench_with_input(
            BenchmarkId::new("ip", batch_size),
            batch_size,
            |bench, _| {
                bench.iter(|| {
                    vectors.iter().map(|v| inner_product(black_box(&query), black_box(v))).collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// DIMENSION SCALING BENCHMARK
// =============================================================================

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance/scaling");
    group.sample_size(50);

    // Test how performance scales with dimension for each metric
    for dim in [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096].iter() {
        let (a, b) = random_vector_pair(*dim);

        group.throughput(Throughput::Elements(*dim as u64));

        group.bench_with_input(BenchmarkId::new("l2", dim), dim, |bench, _| {
            bench.iter(|| l2_distance(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("ip", dim), dim, |bench, _| {
            bench.iter(|| inner_product(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_l2_squared,
    bench_inner_product,
    bench_cosine_similarity,
    bench_cosine_distance,
    bench_norm,
    bench_normalize,
    bench_normalize_in_place,
    bench_compute_distance_dispatch,
    bench_compute_distance_for_heap,
    bench_batch_distances,
    bench_dimension_scaling,
);

criterion_main!(benches);
