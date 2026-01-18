//! Quantization Benchmarks for RAG++
//!
//! Benchmarks for vector quantization operations:
//! - SQ8 encoding/decoding
//! - Asymmetric distance computation
//! - Memory compression ratios

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

use rag_plusplus_core::{
    SQ8Quantizer, PQQuantizer, PQConfig, Quantizer, AsymmetricDistance,
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

/// Generate training vectors for quantizer.
fn training_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| random_vector(dim)).collect()
}

// =============================================================================
// SQ8 ENCODING BENCHMARKS
// =============================================================================

fn bench_sq8_train(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_train");

    for (n_vectors, dim) in [(100, 128), (1000, 512), (10000, 768)].iter() {
        let vectors = training_vectors(*n_vectors, *dim);
        let name = format!("{}x{}", n_vectors, dim);

        group.throughput(Throughput::Elements(*n_vectors as u64));
        group.bench_function(&name, |bench| {
            bench.iter(|| SQ8Quantizer::train(black_box(&vectors)))
        });
    }

    group.finish();
}

fn bench_sq8_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_encode");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let vectors = training_vectors(1000, *dim);
        let quantizer = SQ8Quantizer::train(&vectors);
        let vector = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("single", dim), dim, |bench, _| {
            bench.iter(|| quantizer.encode(black_box(&vector)))
        });
    }

    group.finish();
}

fn bench_sq8_encode_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_encode_batch");

    let dim = 512;
    let vectors = training_vectors(1000, dim);
    let quantizer = SQ8Quantizer::train(&vectors);

    for batch_size in [10, 100, 1000].iter() {
        let batch: Vec<Vec<f32>> = (0..*batch_size).map(|_| random_vector(dim)).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |bench, _| {
            bench.iter(|| quantizer.encode_batch(black_box(&batch)))
        });
    }

    group.finish();
}

fn bench_sq8_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_decode");

    for dim in [128, 512, 1024].iter() {
        let vectors = training_vectors(1000, *dim);
        let quantizer = SQ8Quantizer::train(&vectors);
        let vector = random_vector(*dim);
        let encoded = quantizer.encode(&vector);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("single", dim), dim, |bench, _| {
            bench.iter(|| quantizer.decode(black_box(&encoded)))
        });
    }

    group.finish();
}

// =============================================================================
// ASYMMETRIC DISTANCE BENCHMARKS
// =============================================================================

fn bench_sq8_asymmetric_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_asym_l2");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let vectors = training_vectors(1000, *dim);
        let quantizer = SQ8Quantizer::train(&vectors);
        let query = random_vector(*dim);
        let vector = random_vector(*dim);
        let encoded = quantizer.encode(&vector);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| quantizer.asymmetric_l2(black_box(&query), black_box(&encoded)))
        });
    }

    group.finish();
}

fn bench_sq8_asymmetric_ip(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_asym_ip");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let vectors = training_vectors(1000, *dim);
        let quantizer = SQ8Quantizer::train(&vectors);
        let query = random_vector(*dim);
        let vector = random_vector(*dim);
        let encoded = quantizer.encode(&vector);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| quantizer.asymmetric_inner_product(black_box(&query), black_box(&encoded)))
        });
    }

    group.finish();
}

fn bench_sq8_asymmetric_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_asym_cosine");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let vectors = training_vectors(1000, *dim);
        let quantizer = SQ8Quantizer::train(&vectors);
        let query = random_vector(*dim);
        let vector = random_vector(*dim);
        let encoded = quantizer.encode(&vector);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::new("dim", dim), dim, |bench, _| {
            bench.iter(|| quantizer.asymmetric_cosine(black_box(&query), black_box(&encoded)))
        });
    }

    group.finish();
}

// =============================================================================
// BATCH SEARCH SIMULATION
// =============================================================================

fn bench_sq8_batch_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/sq8_batch_search");
    group.sample_size(50);

    let dim = 512;
    let vectors = training_vectors(1000, dim);
    let quantizer = SQ8Quantizer::train(&vectors);
    let query = random_vector(dim);

    for corpus_size in [100, 1000, 10000].iter() {
        let corpus: Vec<_> = (0..*corpus_size)
            .map(|_| quantizer.encode(&random_vector(dim)))
            .collect();

        group.throughput(Throughput::Elements(*corpus_size as u64));
        group.bench_with_input(
            BenchmarkId::new("linear_scan", corpus_size),
            corpus_size,
            |bench, _| {
                bench.iter(|| {
                    corpus
                        .iter()
                        .map(|enc| quantizer.asymmetric_l2(black_box(&query), enc))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// MEMORY COMPARISON
// =============================================================================

fn bench_sq8_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/memory_comparison");

    let dim = 512;
    let n_vectors = 10000;

    // Create training data
    let vectors = training_vectors(n_vectors, dim);
    let quantizer = SQ8Quantizer::train(&vectors);

    // Encode all vectors
    let encoded: Vec<_> = vectors.iter().map(|v| quantizer.encode(v)).collect();
    let query = random_vector(dim);

    // Memory sizes
    let original_bytes = n_vectors * dim * 4; // f32
    let quantized_bytes = n_vectors * dim; // u8
    let compression = original_bytes as f64 / quantized_bytes as f64;

    println!("\nMemory comparison for {} vectors x {} dim:", n_vectors, dim);
    println!("  Original: {} MB", original_bytes / 1024 / 1024);
    println!("  Quantized: {} MB", quantized_bytes / 1024 / 1024);
    println!("  Compression ratio: {:.1}x", compression);

    // Benchmark full precision vs quantized search
    group.throughput(Throughput::Elements(n_vectors as u64));

    group.bench_function("full_precision_l2", |bench| {
        bench.iter(|| {
            vectors
                .iter()
                .map(|v| rag_plusplus_core::l2_distance(black_box(&query), v))
                .collect::<Vec<_>>()
        })
    });

    group.bench_function("quantized_asym_l2", |bench| {
        bench.iter(|| {
            encoded
                .iter()
                .map(|enc| quantizer.asymmetric_l2(black_box(&query), enc))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

// =============================================================================
// PQ BENCHMARKS
// =============================================================================

fn bench_pq_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/pq_encode");
    group.sample_size(30);

    for (dim, m) in [(512, 16), (512, 32), (768, 24)] {
        let vectors = training_vectors(1000, dim);
        let config = PQConfig {
            m,
            k: 256,
            kmeans_iters: 10,
            seed: Some(42),
        };
        let quantizer = PQQuantizer::train_with_config(&vectors, &config);
        let vector = random_vector(dim);

        let name = format!("{}d_m{}", dim, m);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_function(&name, |bench| {
            bench.iter(|| quantizer.encode(black_box(&vector)))
        });
    }

    group.finish();
}

fn bench_pq_asymmetric_with_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/pq_adc_table");

    let dim = 512;
    let m = 16;
    let vectors = training_vectors(1000, dim);
    let config = PQConfig {
        m,
        k: 256,
        kmeans_iters: 10,
        seed: Some(42),
    };
    let quantizer = PQQuantizer::train_with_config(&vectors, &config);
    let query = random_vector(dim);
    let table = quantizer.compute_distance_table(&query);

    for corpus_size in [100, 1000, 10000] {
        let corpus: Vec<_> = (0..corpus_size)
            .map(|_| quantizer.encode(&random_vector(dim)))
            .collect();

        group.throughput(Throughput::Elements(corpus_size as u64));
        group.bench_with_input(
            BenchmarkId::new("with_table", corpus_size),
            &corpus_size,
            |bench, _| {
                bench.iter(|| {
                    corpus
                        .iter()
                        .map(|enc| quantizer.asymmetric_l2_squared_with_table(black_box(&table), enc))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

fn bench_pq_compression_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization/compression_compare");
    group.sample_size(30);

    let dim = 512;
    let n_vectors = 1000;
    let vectors = training_vectors(n_vectors, dim);
    let query = random_vector(dim);

    // SQ8: 4x compression
    let sq8 = SQ8Quantizer::train(&vectors);
    let sq8_encoded: Vec<_> = vectors.iter().map(|v| sq8.encode(v)).collect();

    // PQ M=16: 128x compression
    let pq_config = PQConfig {
        m: 16,
        k: 256,
        kmeans_iters: 10,
        seed: Some(42),
    };
    let pq16 = PQQuantizer::train_with_config(&vectors, &pq_config);
    let pq16_encoded: Vec<_> = vectors.iter().map(|v| pq16.encode(v)).collect();
    let pq16_table = pq16.compute_distance_table(&query);

    println!("\nCompression comparison for {} vectors x {} dim:", n_vectors, dim);
    println!("  Original: {} KB", n_vectors * dim * 4 / 1024);
    println!("  SQ8 (4x): {} KB", n_vectors * dim / 1024);
    println!("  PQ-16 (128x): {} KB", n_vectors * 16 / 1024);

    group.throughput(Throughput::Elements(n_vectors as u64));

    group.bench_function("sq8_search", |bench| {
        bench.iter(|| {
            sq8_encoded
                .iter()
                .map(|enc| sq8.asymmetric_l2(black_box(&query), enc))
                .collect::<Vec<_>>()
        })
    });

    group.bench_function("pq16_search_with_table", |bench| {
        bench.iter(|| {
            pq16_encoded
                .iter()
                .map(|enc| pq16.asymmetric_l2_squared_with_table(black_box(&pq16_table), enc))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sq8_train,
    bench_sq8_encode,
    bench_sq8_encode_batch,
    bench_sq8_decode,
    bench_sq8_asymmetric_l2,
    bench_sq8_asymmetric_ip,
    bench_sq8_asymmetric_cosine,
    bench_sq8_batch_search,
    bench_sq8_memory_comparison,
    bench_pq_encode,
    bench_pq_asymmetric_with_table,
    bench_pq_compression_comparison,
);

criterion_main!(benches);
