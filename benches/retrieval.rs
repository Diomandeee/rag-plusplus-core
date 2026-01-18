//! Retrieval Pipeline Benchmarks for RAG++
//!
//! End-to-end benchmarks for the query engine:
//! - Full retrieval pipeline
//! - Reranking strategies
//! - Prior computation
//! - Cache performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::collections::HashMap;

use std::time::Duration;

use rag_plusplus_core::{
    FlatIndex, DistanceType, IndexConfig, IndexRegistry,
    InMemoryStore, MemoryRecord, RecordStore,
    QueryEngine, QueryEngineConfig, QueryRequest,
    RerankerConfig, RerankerType,
    CacheConfig, CacheKey, QueryCache,
};
use rag_plusplus_core::retrieval::QueryResponse;
use rag_plusplus_core::types::MetadataValue;

/// Generate random embedding
fn random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

/// Build a test corpus with index
fn build_test_corpus(size: usize, dim: usize) -> (IndexRegistry, InMemoryStore) {
    let mut registry = IndexRegistry::new();
    let config = IndexConfig::new(dim).with_distance(DistanceType::Cosine);
    let index = FlatIndex::new(config);
    registry.register("primary", index).unwrap();

    let mut store = InMemoryStore::new();
    let mut rng = rand::thread_rng();

    for i in 0..size {
        let embedding = random_embedding(dim);
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), MetadataValue::String(format!("cat-{}", i % 10)));
        metadata.insert("score".to_string(), MetadataValue::Float(rng.gen::<f64>()));

        let mut record = MemoryRecord::new(
            format!("record-{i}"),
            embedding.clone(),
            format!("Test context for record {i}"),
            rng.gen::<f64>(),
        );
        record.metadata = metadata;

        store.insert(record.clone()).unwrap();
        registry.add("primary", record.id.clone(), &embedding).unwrap();
    }

    (registry, store)
}

// =============================================================================
// QUERY ENGINE BENCHMARKS
// =============================================================================

fn bench_query_engine_basic(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_engine/basic");

    let dim = 512;
    let corpus_size = 10000;
    let (registry, store) = build_test_corpus(corpus_size, dim);

    let config = QueryEngineConfig::default();
    let engine = QueryEngine::new(config, &registry, &store);

    let query_embedding = random_embedding(dim);

    for k in [5, 10, 50, 100].iter() {
        let k_val = *k;
        let emb = query_embedding.clone();
        group.bench_with_input(BenchmarkId::new("k", k), k, |b, _| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(k_val);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_query_engine_with_priors(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_engine/priors");

    let dim = 512;
    let corpus_size = 10000;
    let (registry, store) = build_test_corpus(corpus_size, dim);
    let query_embedding = random_embedding(dim);

    // Without priors
    {
        let config = QueryEngineConfig {
            build_priors: false,
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("no_priors", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    // With priors
    {
        let config = QueryEngineConfig {
            build_priors: true,
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("with_priors", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_corpus_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_engine/corpus_scaling");
    group.sample_size(20);

    let dim = 512;
    let query_embedding = random_embedding(dim);

    for corpus_size in [1000, 5000, 10000, 25000].iter() {
        let (registry, store) = build_test_corpus(*corpus_size, dim);
        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("corpus", corpus_size),
            corpus_size,
            |b, _| {
                b.iter(|| {
                    let request = QueryRequest::new(emb.clone()).with_k(10);
                    engine.query(black_box(request)).unwrap()
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// RERANKING BENCHMARKS
// =============================================================================

fn bench_reranking_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("reranking/strategies");

    let dim = 512;
    let corpus_size = 10000;
    let (registry, store) = build_test_corpus(corpus_size, dim);
    let query_embedding = random_embedding(dim);

    // No reranking
    {
        let config = QueryEngineConfig {
            reranker: None,
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("none", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    // Outcome-based reranking
    {
        let config = QueryEngineConfig {
            reranker: Some(RerankerConfig {
                strategy: RerankerType::OutcomeWeighted,
                outcome_weight: 0.5,
                ..Default::default()
            }),
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("outcome", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    // Recency reranking
    {
        let config = QueryEngineConfig {
            reranker: Some(RerankerConfig {
                strategy: RerankerType::Recency,
                recency_weight: 0.5,
                recency_half_life: 86400.0, // 1 day
                ..Default::default()
            }),
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("recency", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    // MMR diversity
    {
        let config = QueryEngineConfig {
            reranker: Some(RerankerConfig {
                strategy: RerankerType::MMR,
                mmr_lambda: 0.5,
                ..Default::default()
            }),
            ..Default::default()
        };
        let engine = QueryEngine::new(config, &registry, &store);
        let emb = query_embedding.clone();

        group.bench_function("mmr", |b| {
            b.iter(|| {
                let request = QueryRequest::new(emb.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    group.finish();
}

// =============================================================================
// CACHE BENCHMARKS
// =============================================================================

/// Create a mock QueryResponse for cache testing
fn mock_query_response() -> QueryResponse {
    QueryResponse {
        results: vec![],
        priors: None,
        latency: Duration::from_millis(1),
        indexes_searched: 1,
        candidates_considered: 10,
    }
}

fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache/performance");

    let dim = 512;

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..100).map(|_| random_embedding(dim)).collect();

    // Pre-populate cache
    let config = CacheConfig {
        max_entries: 1000,
        ttl: Duration::from_secs(300),
        cache_filtered: true,
    };
    let cache = QueryCache::new(config);

    // Insert first 50 queries
    for query in queries.iter().take(50) {
        let key = CacheKey::new(query, 10, None, None);
        let response = mock_query_response();
        cache.put(key, response);
    }

    let hit_query = queries[0].clone();
    let miss_query = queries[75].clone();

    group.bench_function("cache_hit", |b| {
        b.iter(|| {
            let key = CacheKey::new(&hit_query, 10, None, None);
            cache.get(black_box(&key))
        });
    });

    group.bench_function("cache_miss", |b| {
        b.iter(|| {
            let key = CacheKey::new(&miss_query, 10, None, None);
            cache.get(black_box(&key))
        });
    });

    group.bench_function("cache_put", |b| {
        let mut i = 0;
        b.iter(|| {
            let query = &queries[i % queries.len()];
            let key = CacheKey::new(query, 10, None, None);
            cache.put(key, black_box(mock_query_response()));
            i += 1;
        });
    });

    group.finish();
}

fn bench_cache_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache/sizes");

    let dim = 512;
    let queries: Vec<Vec<f32>> = (0..1000).map(|_| random_embedding(dim)).collect();

    for cache_size in [100, 1000, 10000].iter() {
        let config = CacheConfig {
            max_entries: *cache_size,
            ttl: Duration::from_secs(300),
            cache_filtered: true,
        };
        let cache = QueryCache::new(config);

        // Fill to capacity
        for query in queries.iter().take(*cache_size) {
            let key = CacheKey::new(query, 10, None, None);
            cache.put(key, mock_query_response());
        }

        let lookup_query = queries[0].clone();
        group.bench_with_input(
            BenchmarkId::new("lookup", cache_size),
            cache_size,
            |b, _| {
                b.iter(|| {
                    let key = CacheKey::new(&lookup_query, 10, None, None);
                    cache.get(black_box(&key))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// BATCH QUERY BENCHMARKS
// =============================================================================

fn bench_batch_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_engine/batch");

    let dim = 512;
    let corpus_size = 10000;
    let (registry, store) = build_test_corpus(corpus_size, dim);

    let config = QueryEngineConfig::default();
    let engine = QueryEngine::new(config, &registry, &store);

    for batch_size in [1, 10, 50, 100].iter() {
        let queries: Vec<Vec<f32>> = (0..*batch_size).map(|_| random_embedding(dim)).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("queries", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    for query in &queries {
                        let request = QueryRequest::new(query.clone()).with_k(10);
                        engine.query(black_box(request)).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// DIMENSION SCALING
// =============================================================================

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_engine/dimension");

    let corpus_size = 5000;

    for dim in [128, 256, 512, 768, 1024].iter() {
        let (registry, store) = build_test_corpus(corpus_size, *dim);
        let query_embedding = random_embedding(*dim);

        let config = QueryEngineConfig::default();
        let engine = QueryEngine::new(config, &registry, &store);

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, _| {
            b.iter(|| {
                let request = QueryRequest::new(query_embedding.clone()).with_k(10);
                engine.query(black_box(request)).unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_query_engine_basic,
    bench_query_engine_with_priors,
    bench_corpus_scaling,
    bench_reranking_strategies,
    bench_cache_performance,
    bench_cache_sizes,
    bench_batch_queries,
    bench_dimension_scaling,
);

criterion_main!(benches);
