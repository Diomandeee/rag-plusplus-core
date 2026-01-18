//! Benchmarks for rkyv serialization.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rag_plusplus_core::{MemoryRecord, OutcomeStats};
use rkyv::{to_bytes, from_bytes};
use rkyv::ser::serializers::AllocSerializer;

fn create_test_record(dim: usize) -> MemoryRecord {
    let mut stats = OutcomeStats::new(16);
    for i in 0..10 {
        let outcome: Vec<f32> = (0..16).map(|j| (i * j) as f32 * 0.01).collect();
        stats.update(&outcome);
    }
    
    let embedding: Vec<f32> = (0..dim).map(|i| i as f32 * 0.001).collect();
    MemoryRecord::new(format!("record_{}", dim), embedding, stats)
}

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("rkyv_serialize");
    
    for dim in [128, 256, 512, 1024].iter() {
        let record = create_test_record(*dim);
        
        group.bench_with_input(BenchmarkId::new("dim", dim), &record, |b, record| {
            b.iter(|| {
                let bytes = to_bytes::<rkyv::rancor::Error>(black_box(record)).unwrap();
                black_box(bytes);
            });
        });
    }
    
    group.finish();
}

fn bench_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("rkyv_deserialize");
    
    for dim in [128, 256, 512, 1024].iter() {
        let record = create_test_record(*dim);
        let bytes = to_bytes::<rkyv::rancor::Error>(&record).unwrap();
        
        group.bench_with_input(BenchmarkId::new("dim", dim), &bytes, |b, bytes| {
            b.iter(|| {
                let archived = unsafe { rkyv::access_unchecked::<MemoryRecord>(black_box(bytes)) };
                black_box(archived);
            });
        });
    }
    
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("rkyv_roundtrip");
    
    for dim in [128, 256, 512, 1024].iter() {
        let record = create_test_record(*dim);
        
        group.bench_with_input(BenchmarkId::new("dim", dim), &record, |b, record| {
            b.iter(|| {
                let bytes = to_bytes::<rkyv::rancor::Error>(black_box(record)).unwrap();
                let archived = unsafe { rkyv::access_unchecked::<MemoryRecord>(&bytes) };
                black_box(archived);
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_serialize, bench_deserialize, bench_roundtrip);
criterion_main!(benches);
