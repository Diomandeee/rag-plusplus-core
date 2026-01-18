//! Benchmarks for Welford's algorithm implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rag_plusplus_core::OutcomeStats;

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("welford_update");
    
    for dim in [8, 32, 128, 512].iter() {
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let mut stats = OutcomeStats::new(dim);
            let outcome: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
            
            b.iter(|| {
                stats.update(black_box(&outcome));
            });
        });
    }
    
    group.finish();
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("welford_merge");
    
    for dim in [8, 32, 128, 512].iter() {
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let mut stats1 = OutcomeStats::new(dim);
            let mut stats2 = OutcomeStats::new(dim);
            
            // Pre-populate with some data
            for i in 0..100 {
                let outcome: Vec<f32> = (0..dim).map(|j| (i * j) as f32 * 0.001).collect();
                stats1.update(&outcome);
                stats2.update(&outcome);
            }
            
            b.iter(|| {
                black_box(stats1.merge(&stats2));
            });
        });
    }
    
    group.finish();
}

fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("welford_variance");
    
    for dim in [8, 32, 128, 512].iter() {
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let mut stats = OutcomeStats::new(dim);
            
            // Pre-populate
            for i in 0..1000 {
                let outcome: Vec<f32> = (0..dim).map(|j| (i * j) as f32 * 0.001).collect();
                stats.update(&outcome);
            }
            
            b.iter(|| {
                black_box(stats.variance());
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_update, bench_merge, bench_variance);
criterion_main!(benches);
