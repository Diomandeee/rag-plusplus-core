//! Index Benchmarks for RAG++
//!
//! Benchmarks for vector index operations:
//! - Add single/batch vectors
//! - Search k-nearest neighbors
//! - Flat vs HNSW comparison
//! - Scaling behavior

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rag_plusplus_core::{FlatIndex, HNSWConfig, HNSWIndex, DistanceType, IndexConfig, VectorIndex};

/// Generate random vectors for benchmarking
fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Generate normalized random vectors (for cosine similarity)
fn random_normalized_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    random_vectors(count, dim)
        .into_iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

// =============================================================================
// FLAT INDEX BENCHMARKS
// =============================================================================

fn bench_flat_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_index/add");

    for dim in [128, 256, 512, 768].iter() {
        let vectors = random_vectors(1000, *dim);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("single", dim), dim, |b, &dim| {
            b.iter_batched(
                || {
                    let config = IndexConfig::new(dim).with_distance(DistanceType::L2);
                    let index = FlatIndex::new(config);
                    let vec = &vectors[0];
                    (index, vec)
                },
                |(mut index, vec)| {
                    index.add("test".to_string(), black_box(vec)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_flat_add_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_index/add_batch");

    for batch_size in [10, 100, 1000].iter() {
        let dim = 512;
        let vectors = random_vectors(*batch_size, dim);

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, &size| {
            b.iter_batched(
                || {
                    let config = IndexConfig::new(dim).with_distance(DistanceType::L2);
                    let index = FlatIndex::new(config);
                    let batch: Vec<_> = (0..size)
                        .map(|i| (format!("id-{i}"), vectors[i].clone()))
                        .collect();
                    (index, batch)
                },
                |(mut index, batch)| {
                    for (id, vec) in batch {
                        index.add(id, black_box(&vec)).unwrap();
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_flat_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_index/search");

    // Vary corpus size
    for corpus_size in [1000, 10000, 50000].iter() {
        let dim = 512;
        let vectors = random_vectors(*corpus_size, dim);
        let query = random_vectors(1, dim).pop().unwrap();

        // Build index
        let config = IndexConfig::new(dim).with_distance(DistanceType::L2);
        let mut index = FlatIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("id-{i}"), vec).unwrap();
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("k=10", corpus_size),
            corpus_size,
            |b, _| {
                b.iter(|| {
                    index.search(black_box(&query), 10).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_flat_search_vary_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_index/search_k");

    let dim = 512;
    let corpus_size = 10000;
    let vectors = random_vectors(corpus_size, dim);
    let query = random_vectors(1, dim).pop().unwrap();

    // Build index
    let config = IndexConfig::new(dim).with_distance(DistanceType::L2);
    let mut index = FlatIndex::new(config);
    for (i, vec) in vectors.iter().enumerate() {
        index.add(format!("id-{i}"), vec).unwrap();
    }

    for k in [1, 5, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("n=10k", k), k, |b, &k| {
            b.iter(|| {
                index.search(black_box(&query), k).unwrap()
            });
        });
    }

    group.finish();
}

// =============================================================================
// HNSW INDEX BENCHMARKS
// =============================================================================

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_index/build");
    group.sample_size(10); // HNSW build is slow

    for corpus_size in [1000, 5000].iter() {
        let dim = 512;
        let vectors = random_normalized_vectors(*corpus_size, dim);

        group.throughput(Throughput::Elements(*corpus_size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors", corpus_size),
            corpus_size,
            |b, _| {
                b.iter(|| {
                    let config = HNSWConfig::new(dim)
                        .with_m(16)
                        .with_ef_construction(100)
                        .with_ef_search(50)
                        .with_distance(DistanceType::Cosine);
                    let mut index = HNSWIndex::new(config);
                    for (i, vec) in vectors.iter().enumerate() {
                        index.add(format!("id-{i}"), black_box(vec)).unwrap();
                    }
                    index
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_index/search");

    let dim = 512;
    let corpus_size = 10000;
    let vectors = random_normalized_vectors(corpus_size, dim);
    let query = random_normalized_vectors(1, dim).pop().unwrap();

    // Build index
    let config = HNSWConfig::new(dim)
        .with_m(16)
        .with_ef_construction(200)
        .with_ef_search(50)
        .with_distance(DistanceType::Cosine);
    let mut index = HNSWIndex::new(config);
    for (i, vec) in vectors.iter().enumerate() {
        index.add(format!("id-{i}"), vec).unwrap();
    }

    for k in [1, 10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("k", k), k, |b, &k| {
            b.iter(|| {
                index.search(black_box(&query), k).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_hnsw_ef_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_index/ef_search");

    let dim = 512;
    let corpus_size = 10000;
    let vectors = random_normalized_vectors(corpus_size, dim);
    let queries = random_normalized_vectors(10, dim);

    for ef in [10, 50, 100, 200].iter() {
        let config = HNSWConfig::new(dim)
            .with_m(16)
            .with_ef_construction(200)
            .with_ef_search(*ef)
            .with_distance(DistanceType::Cosine);
        let mut index = HNSWIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("id-{i}"), vec).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("ef", ef), ef, |b, _| {
            b.iter(|| {
                for query in &queries {
                    index.search(black_box(query), 10).unwrap();
                }
            });
        });
    }

    group.finish();
}

// =============================================================================
// COMPARISON: FLAT vs HNSW
// =============================================================================

fn bench_compare_flat_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_comparison");

    let dim = 512;
    let corpus_size = 10000;
    let vectors = random_normalized_vectors(corpus_size, dim);
    let query = random_normalized_vectors(1, dim).pop().unwrap();

    // Build Flat index
    let config = IndexConfig::new(dim).with_distance(DistanceType::Cosine);
    let mut flat_index = FlatIndex::new(config);
    for (i, vec) in vectors.iter().enumerate() {
        flat_index.add(format!("id-{i}"), vec).unwrap();
    }

    // Build HNSW index
    let config = HNSWConfig::new(dim)
        .with_m(16)
        .with_ef_construction(200)
        .with_ef_search(50)
        .with_distance(DistanceType::Cosine);
    let mut hnsw_index = HNSWIndex::new(config);
    for (i, vec) in vectors.iter().enumerate() {
        hnsw_index.add(format!("id-{i}"), vec).unwrap();
    }

    group.bench_function("flat/k=10", |b| {
        b.iter(|| flat_index.search(black_box(&query), 10).unwrap());
    });

    group.bench_function("hnsw/k=10", |b| {
        b.iter(|| hnsw_index.search(black_box(&query), 10).unwrap());
    });

    group.finish();
}

// =============================================================================
// DISTANCE TYPE COMPARISON
// =============================================================================

fn bench_distance_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_types");

    let dim = 512;
    let corpus_size = 5000;
    let vectors = random_normalized_vectors(corpus_size, dim);
    let query = random_normalized_vectors(1, dim).pop().unwrap();

    for distance_type in [DistanceType::L2, DistanceType::Cosine, DistanceType::InnerProduct].iter() {
        let config = IndexConfig::new(dim).with_distance(*distance_type);
        let mut index = FlatIndex::new(config);
        for (i, vec) in vectors.iter().enumerate() {
            index.add(format!("id-{i}"), vec).unwrap();
        }

        let name = match distance_type {
            DistanceType::L2 => "l2",
            DistanceType::Cosine => "cosine",
            DistanceType::InnerProduct => "inner_product",
        };

        group.bench_function(name, |b| {
            b.iter(|| index.search(black_box(&query), 10).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_flat_add,
    bench_flat_add_batch,
    bench_flat_search,
    bench_flat_search_vary_k,
    bench_hnsw_build,
    bench_hnsw_search,
    bench_hnsw_ef_search,
    bench_compare_flat_hnsw,
    bench_distance_types,
);

criterion_main!(benches);
