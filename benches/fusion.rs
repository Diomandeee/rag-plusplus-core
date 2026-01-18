//! Score Fusion Benchmarks for RAG++
//!
//! Benchmarks for multi-index result fusion:
//! - RRF (Reciprocal Rank Fusion)
//! - CombSUM, CombMNZ, CombMIN, CombMAX
//! - Varying number of indexes and results

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rag_plusplus_core::{
    FusionConfig, FusionStrategy, ScoreFusion, SearchResult, DistanceType,
    MultiIndexResults, rrf_fuse, rrf_fuse_top_k,
};

/// Generate mock search results
fn mock_results(count: usize, overlap_ratio: f32) -> Vec<SearchResult> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|i| {
            // Some IDs overlap across indexes
            let id = if rng.gen::<f32>() < overlap_ratio {
                format!("shared-{}", i % (count / 2).max(1))
            } else {
                format!("unique-{i}")
            };
            SearchResult::new(id, rng.gen::<f32>(), DistanceType::Cosine)
        })
        .collect()
}

/// Generate MultiIndexResults from multiple indexes
fn mock_multi_index_results(
    num_indexes: usize,
    results_per_index: usize,
    overlap_ratio: f32,
) -> MultiIndexResults {
    let mut results = MultiIndexResults::new();
    for i in 0..num_indexes {
        let index_name = format!("index-{i}");
        let index_results = mock_results(results_per_index, overlap_ratio);
        results.add(index_name, index_results);
    }
    results
}

// =============================================================================
// RRF FUSION BENCHMARKS
// =============================================================================

fn bench_rrf_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/rrf");

    // Vary number of indexes
    for num_indexes in [2, 3, 5, 10].iter() {
        let results_per_index = 100;
        let multi_results = mock_multi_index_results(*num_indexes, results_per_index, 0.3);

        group.throughput(Throughput::Elements(*num_indexes as u64));
        group.bench_with_input(
            BenchmarkId::new("indexes", num_indexes),
            num_indexes,
            |b, _| {
                b.iter(|| {
                    rrf_fuse(black_box(&multi_results))
                });
            },
        );
    }

    group.finish();
}

fn bench_rrf_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/rrf_top_k");

    let num_indexes = 5;
    let results_per_index = 100;
    let multi_results = mock_multi_index_results(num_indexes, results_per_index, 0.3);

    for k in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("k", k), k, |b, &k| {
            b.iter(|| {
                rrf_fuse_top_k(black_box(&multi_results), k)
            });
        });
    }

    group.finish();
}

fn bench_rrf_vary_results(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/rrf_results");

    let num_indexes = 3;

    for results_per_index in [50, 100, 500, 1000].iter() {
        let multi_results = mock_multi_index_results(num_indexes, *results_per_index, 0.3);

        group.throughput(Throughput::Elements(*results_per_index as u64));
        group.bench_with_input(
            BenchmarkId::new("per_index", results_per_index),
            results_per_index,
            |b, _| {
                b.iter(|| {
                    rrf_fuse_top_k(black_box(&multi_results), 10)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SCORE FUSION STRATEGIES
// =============================================================================

fn bench_fusion_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/strategies");

    let num_indexes = 5;
    let results_per_index = 100;
    let multi_results = mock_multi_index_results(num_indexes, results_per_index, 0.3);

    for strategy in [
        FusionStrategy::RRF,
        FusionStrategy::CombSUM,
        FusionStrategy::CombMNZ,
        FusionStrategy::CombMIN,
        FusionStrategy::CombMAX,
    ].iter() {
        let config = FusionConfig {
            strategy: strategy.clone(),
            rrf_k: 60,
            weights: None,
            normalize_scores: true,
        };
        let fuser = ScoreFusion::with_config(config);

        let name = match strategy {
            FusionStrategy::RRF => "rrf",
            FusionStrategy::CombSUM => "comb_sum",
            FusionStrategy::CombMNZ => "comb_mnz",
            FusionStrategy::CombMIN => "comb_min",
            FusionStrategy::CombMAX => "comb_max",
        };

        group.bench_function(name, |b| {
            b.iter(|| {
                fuser.fuse(black_box(&multi_results))
            });
        });
    }

    group.finish();
}

// =============================================================================
// OVERLAP RATIO IMPACT
// =============================================================================

fn bench_overlap_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/overlap");

    let num_indexes = 5;
    let results_per_index = 100;

    for overlap in [0.0, 0.25, 0.5, 0.75, 1.0].iter() {
        let multi_results = mock_multi_index_results(num_indexes, results_per_index, *overlap);

        group.bench_with_input(
            BenchmarkId::new("ratio", format!("{:.0}%", overlap * 100.0)),
            overlap,
            |b, _| {
                b.iter(|| {
                    rrf_fuse_top_k(black_box(&multi_results), 10)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// LARGE SCALE BENCHMARK
// =============================================================================

fn bench_large_scale_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion/large_scale");
    group.sample_size(20);

    let num_indexes = 10;
    let results_per_index = 1000;
    let multi_results = mock_multi_index_results(num_indexes, results_per_index, 0.3);

    group.bench_function("10_indexes_1k_results", |b| {
        b.iter(|| {
            rrf_fuse_top_k(black_box(&multi_results), 100)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rrf_fusion,
    bench_rrf_top_k,
    bench_rrf_vary_results,
    bench_fusion_strategies,
    bench_overlap_impact,
    bench_large_scale_fusion,
);

criterion_main!(benches);
