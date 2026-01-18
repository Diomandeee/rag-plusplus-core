//! WAL (Write-Ahead Log) Benchmarks for RAG++
//!
//! Benchmarks for durability layer:
//! - Write throughput
//! - Recovery/replay performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use tempfile::TempDir;

use rag_plusplus_core::wal::{WalConfig, WalWriter, WalReader};
use rag_plusplus_core::MemoryRecord;
use rag_plusplus_core::types::MetadataValue;

/// Generate a random MemoryRecord for benchmarking
fn random_record(id: usize, dim: usize) -> MemoryRecord {
    let mut rng = rand::thread_rng();

    let embedding: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
    let mut record = MemoryRecord::new(
        format!("record-{id}"),
        embedding,
        format!("Context for record {id}"),
        rng.gen(),
    );
    record.metadata.insert("key".to_string(), MetadataValue::String("value".to_string()));
    record
}

// =============================================================================
// WAL WRITE BENCHMARKS
// =============================================================================

fn bench_wal_write_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/write_single");

    for dim in [128, 512, 1024].iter() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(false);
        let mut writer = WalWriter::new(config).unwrap();

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let mut seq = 0usize;
            b.iter(|| {
                let record = random_record(seq, dim);
                writer.log_insert(black_box(&record)).unwrap();
                seq += 1;
            });
        });
    }

    group.finish();
}

fn bench_wal_write_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/write_batch");

    let dim = 512;

    for batch_size in [10, 100, 1000].iter() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(false);
        let mut writer = WalWriter::new(config).unwrap();

        let records: Vec<MemoryRecord> = (0..*batch_size)
            .map(|i| random_record(i, dim))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                for record in &records {
                    writer.log_insert(black_box(record)).unwrap();
                }
            });
        });
    }

    group.finish();
}

fn bench_wal_sync_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/sync_modes");
    group.sample_size(50);

    let dim = 512;

    // No sync
    {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(false);
        let mut writer = WalWriter::new(config).unwrap();

        group.bench_function("no_sync", |b| {
            let mut seq = 0;
            b.iter(|| {
                let record = random_record(seq, dim);
                writer.log_insert(black_box(&record)).unwrap();
                seq += 1;
            });
        });
    }

    // With sync (much slower due to fsync)
    {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(true);
        let mut writer = WalWriter::new(config).unwrap();

        group.bench_function("with_sync", |b| {
            let mut seq = 0;
            b.iter(|| {
                let record = random_record(seq, dim);
                writer.log_insert(black_box(&record)).unwrap();
                seq += 1;
            });
        });
    }

    group.finish();
}

// =============================================================================
// WAL READ/REPLAY BENCHMARKS
// =============================================================================

fn bench_wal_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/read");

    let dim = 512;

    for num_entries in [100, 1000, 10000].iter() {
        // Write entries first
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(false);
        let mut writer = WalWriter::new(config).unwrap();

        for i in 0..*num_entries {
            let record = random_record(i, dim);
            writer.log_insert(&record).unwrap();
        }
        drop(writer);

        // Benchmark reading
        group.throughput(Throughput::Elements(*num_entries as u64));
        group.bench_with_input(
            BenchmarkId::new("entries", num_entries),
            num_entries,
            |b, _| {
                b.iter(|| {
                    let reader = WalReader::open(temp_dir.path()).unwrap();
                    let count = reader.count();
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

fn bench_wal_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/replay");
    group.sample_size(20);

    let dim = 512;
    let num_entries = 1000;

    // Write entries
    let temp_dir = TempDir::new().unwrap();
    let config = WalConfig::new(temp_dir.path())
        .with_max_file_size(100 * 1024 * 1024)
        .with_sync_on_write(false);
    let mut writer = WalWriter::new(config).unwrap();

    for i in 0..num_entries {
        let record = random_record(i, dim);
        writer.log_insert(&record).unwrap();
    }
    drop(writer);

    group.bench_function("full_replay", |b| {
        b.iter(|| {
            let reader = WalReader::open(temp_dir.path()).unwrap();
            let entries: Vec<_> = reader.collect();
            black_box(entries.len())
        });
    });

    group.finish();
}

// =============================================================================
// EMBEDDING SIZE SCALING
// =============================================================================

fn bench_wal_embedding_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/embedding_size");

    for dim in [128, 256, 512, 768, 1024, 2048].iter() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(100 * 1024 * 1024)
            .with_sync_on_write(false);
        let mut writer = WalWriter::new(config).unwrap();

        // Approximate bytes per entry
        let bytes_per_entry = (*dim * 4) + 100; // 4 bytes per f32 + overhead
        group.throughput(Throughput::Bytes(bytes_per_entry as u64));

        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |b, &dim| {
            let mut seq = 0;
            b.iter(|| {
                let record = random_record(seq, dim);
                writer.log_insert(black_box(&record)).unwrap();
                seq += 1;
            });
        });
    }

    group.finish();
}

// =============================================================================
// UPDATE STATS BENCHMARK
// =============================================================================

fn bench_wal_update_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal/update_stats");

    let temp_dir = TempDir::new().unwrap();
    let config = WalConfig::new(temp_dir.path())
        .with_max_file_size(100 * 1024 * 1024)
        .with_sync_on_write(false);
    let mut writer = WalWriter::new(config).unwrap();

    let mut rng = rand::thread_rng();

    group.bench_function("update", |b| {
        let mut seq = 0;
        b.iter(|| {
            let record_id = format!("record-{}", seq % 1000);
            let outcome: f64 = rng.gen();
            writer.log_update_stats(black_box(&record_id), black_box(outcome)).unwrap();
            seq += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_wal_write_single,
    bench_wal_write_batch,
    bench_wal_sync_modes,
    bench_wal_read,
    bench_wal_replay,
    bench_wal_embedding_sizes,
    bench_wal_update_stats,
);

criterion_main!(benches);
