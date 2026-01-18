//! End-to-End Benchmarks for RAG++ SOTA Improvements
//!
//! Compares:
//! - Dense-only vs Hybrid (Dense + Sparse BM25) retrieval
//! - Memory usage: Full f32 vs SQ8 (4x) vs PQ (32-128x)
//! - Index construction: Sequential vs Parallel HNSW
//!
//! Run with: cargo bench --bench e2e_benchmark

use std::collections::HashSet;
use std::time::{Duration, Instant};

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::prelude::*;

use rag_plusplus_core::{
    // Indexes
    FlatIndex, HNSWConfig, HNSWIndex, IndexConfig, VectorIndex,
    ParallelHNSWBuilder, ParallelHNSWIndex,
    // Sparse / Hybrid
    BM25Index, HybridSearcher, HybridFusionStrategy, HybridSearchConfig,
    // Quantization
    SQ8Quantizer, PQQuantizer, PQConfig, Quantizer, AsymmetricDistance,
    // Evaluation
    Evaluator, EvaluationSummary,
};

// =============================================================================
// DATASET GENERATORS
// =============================================================================

/// Generate random normalized vectors.
fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect()
}

/// Generate text documents for BM25.
fn generate_documents(n: usize, seed: u64) -> Vec<String> {
    let vocabulary = [
        "machine", "learning", "deep", "neural", "network", "algorithm",
        "training", "model", "data", "feature", "optimization", "gradient",
        "loss", "accuracy", "precision", "recall", "transformer", "attention",
        "embedding", "vector", "retrieval", "search", "index", "query",
        "reinforcement", "policy", "agent", "environment", "reward", "state",
        "action", "value", "function", "approximation", "convergence", "batch",
        "epoch", "layer", "activation", "dropout", "regularization", "inference",
    ];

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let num_words = rng.gen_range(5..20);
            (0..num_words)
                .map(|_| vocabulary[rng.gen_range(0..vocabulary.len())])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Generate ground truth: for each query, the k-nearest neighbors.
fn compute_ground_truth(
    corpus: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<HashSet<String>> {
    queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, doc)| {
                    let dist: f32 = query
                        .iter()
                        .zip(doc.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances
                .into_iter()
                .take(k)
                .map(|(i, _)| format!("doc-{i}"))
                .collect()
        })
        .collect()
}

// =============================================================================
// BENCHMARK: DENSE VS HYBRID RETRIEVAL
// =============================================================================

fn bench_dense_vs_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieval/dense_vs_hybrid");
    group.sample_size(30);

    let n_docs = 10000;
    let n_queries = 100;
    let dim = 128;
    let k = 10;

    // Generate data
    let corpus = generate_vectors(n_docs, dim, 42);
    let documents = generate_documents(n_docs, 42);
    let queries = generate_vectors(n_queries, dim, 123);
    let query_texts: Vec<String> = (0..n_queries)
        .map(|i| {
            // Create query text that matches some documents
            let words = ["machine learning", "neural network", "deep learning", "retrieval search"];
            words[i % words.len()].to_string()
        })
        .collect();

    // Build dense index
    let mut dense_index = FlatIndex::new(IndexConfig::new(dim));
    for (i, vec) in corpus.iter().enumerate() {
        dense_index.add(format!("doc-{i}"), vec).unwrap();
    }

    // Build sparse index
    let mut sparse_index = BM25Index::new();
    for (i, doc) in documents.iter().enumerate() {
        sparse_index.add(format!("doc-{i}"), doc);
    }

    // Compute ground truth for recall evaluation
    let ground_truth = compute_ground_truth(&corpus, &queries, k);

    // === Dense-only benchmark ===
    group.throughput(Throughput::Elements(n_queries as u64));
    group.bench_function("dense_only", |bench| {
        bench.iter(|| {
            for query in &queries {
                let _ = dense_index.search(black_box(query), k);
            }
        })
    });

    // === Hybrid (RRF) benchmark ===
    group.bench_function("hybrid_rrf", |bench| {
        let config = HybridSearchConfig {
            strategy: HybridFusionStrategy::RRF { k: 60.0 },
            candidates_per_index: 50,
            min_sparse_score: 0.0,
        };
        let searcher = HybridSearcher::new(&dense_index, &sparse_index).with_config(config);

        bench.iter(|| {
            for (query, text) in queries.iter().zip(query_texts.iter()) {
                let _ = searcher.search(black_box(query), black_box(text), k);
            }
        })
    });

    // === Hybrid (Linear) benchmark ===
    group.bench_function("hybrid_linear", |bench| {
        let config = HybridSearchConfig {
            strategy: HybridFusionStrategy::Linear { alpha: 0.7 },
            candidates_per_index: 50,
            min_sparse_score: 0.0,
        };
        let searcher = HybridSearcher::new(&dense_index, &sparse_index).with_config(config);

        bench.iter(|| {
            for (query, text) in queries.iter().zip(query_texts.iter()) {
                let _ = searcher.search(black_box(query), black_box(text), k);
            }
        })
    });

    // Print recall comparison
    println!("\n=== Recall@{k} Comparison ===");

    // Dense recall
    let evaluator = Evaluator::new(k);
    let mut dense_evals = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = dense_index.search(query, k).unwrap();
        let retrieved: Vec<String> = results.into_iter().map(|r| r.id).collect();
        let eval = evaluator.evaluate_query(
            format!("q-{i}"),
            &retrieved,
            &ground_truth[i],
            start.elapsed(),
        );
        dense_evals.push(eval);
    }
    let dense_summary = EvaluationSummary::from_evaluations(&dense_evals);
    println!("Dense-only:   Recall@{k} = {:.4}, MRR = {:.4}", dense_summary.mean_recall, dense_summary.mrr);

    // Hybrid recall (using dense ground truth for fair comparison)
    let config = HybridSearchConfig {
        strategy: HybridFusionStrategy::RRF { k: 60.0 },
        candidates_per_index: 50,
        min_sparse_score: 0.0,
    };
    let searcher = HybridSearcher::new(&dense_index, &sparse_index).with_config(config);
    let mut hybrid_evals = Vec::new();
    for (i, (query, text)) in queries.iter().zip(query_texts.iter()).enumerate() {
        let start = Instant::now();
        let results = searcher.search(query, text, k).unwrap();
        let retrieved: Vec<String> = results.into_iter().map(|r| r.id).collect();
        let eval = evaluator.evaluate_query(
            format!("q-{i}"),
            &retrieved,
            &ground_truth[i],
            start.elapsed(),
        );
        hybrid_evals.push(eval);
    }
    let hybrid_summary = EvaluationSummary::from_evaluations(&hybrid_evals);
    println!("Hybrid (RRF): Recall@{k} = {:.4}, MRR = {:.4}", hybrid_summary.mean_recall, hybrid_summary.mrr);

    group.finish();
}

// =============================================================================
// BENCHMARK: MEMORY COMPRESSION (FULL vs SQ8 vs PQ)
// =============================================================================

fn bench_memory_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/compression");
    group.sample_size(20);

    let n_docs = 50000;
    let dim = 512;
    let n_queries = 100;
    let k = 10;

    println!("\n=== Memory Compression Benchmark ===");
    println!("Corpus: {} vectors x {} dimensions", n_docs, dim);

    // Generate data
    let corpus = generate_vectors(n_docs, dim, 42);
    let queries = generate_vectors(n_queries, dim, 123);

    // Memory calculation
    let full_memory = n_docs * dim * 4; // f32 = 4 bytes
    println!("\nMemory Usage:");
    println!("  Full f32:  {:>8.2} MB", full_memory as f64 / 1_000_000.0);
    println!("  SQ8 (4x):  {:>8.2} MB", (full_memory / 4) as f64 / 1_000_000.0);
    println!("  PQ-16 (32x): {:>6.2} MB", (n_docs * 16) as f64 / 1_000_000.0);
    println!("  PQ-8 (64x):  {:>6.2} MB", (n_docs * 8) as f64 / 1_000_000.0);

    // Train quantizers
    let training_sample: Vec<Vec<f32>> = corpus.iter().take(5000).cloned().collect();

    println!("\nTraining quantizers...");
    let sq8 = SQ8Quantizer::train(&training_sample);

    let pq16_config = PQConfig {
        m: 16,
        k: 256,
        kmeans_iters: 10,
        seed: Some(42),
    };
    let pq16 = PQQuantizer::train_with_config(&training_sample, &pq16_config);

    let pq8_config = PQConfig {
        m: 8,
        k: 256,
        kmeans_iters: 10,
        seed: Some(42),
    };
    let pq8 = PQQuantizer::train_with_config(&training_sample, &pq8_config);

    // Encode corpus
    println!("Encoding corpus...");
    let sq8_corpus: Vec<_> = corpus.iter().map(|v| sq8.encode(v)).collect();
    let pq16_corpus: Vec<_> = corpus.iter().map(|v| pq16.encode(v)).collect();
    let pq8_corpus: Vec<_> = corpus.iter().map(|v| pq8.encode(v)).collect();

    // === Full precision search ===
    group.throughput(Throughput::Elements(n_queries as u64));
    group.bench_function("full_f32_search", |bench| {
        bench.iter(|| {
            for query in &queries {
                // Brute force search over full corpus
                let mut scores: Vec<(usize, f32)> = corpus
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| {
                        let ip: f32 = query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                        (i, ip)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let _top_k: Vec<_> = scores.into_iter().take(k).collect();
            }
        })
    });

    // === SQ8 search ===
    group.bench_function("sq8_search", |bench| {
        bench.iter(|| {
            for query in &queries {
                let mut scores: Vec<(usize, f32)> = sq8_corpus
                    .iter()
                    .enumerate()
                    .map(|(i, enc)| {
                        let ip = sq8.asymmetric_inner_product(query, enc);
                        (i, ip)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let _top_k: Vec<_> = scores.into_iter().take(k).collect();
            }
        })
    });

    // === PQ-16 search with precomputed tables ===
    group.bench_function("pq16_table_search", |bench| {
        bench.iter(|| {
            for query in &queries {
                let table = pq16.compute_distance_table(query);
                let mut scores: Vec<(usize, f32)> = pq16_corpus
                    .iter()
                    .enumerate()
                    .map(|(i, enc)| {
                        let dist = pq16.asymmetric_l2_squared_with_table(&table, enc);
                        (i, -dist) // Negative for sorting (lower dist = better)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let _top_k: Vec<_> = scores.into_iter().take(k).collect();
            }
        })
    });

    // === PQ-8 search with precomputed tables ===
    group.bench_function("pq8_table_search", |bench| {
        bench.iter(|| {
            for query in &queries {
                let table = pq8.compute_distance_table(query);
                let mut scores: Vec<(usize, f32)> = pq8_corpus
                    .iter()
                    .enumerate()
                    .map(|(i, enc)| {
                        let dist = pq8.asymmetric_l2_squared_with_table(&table, enc);
                        (i, -dist)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let _top_k: Vec<_> = scores.into_iter().take(k).collect();
            }
        })
    });

    // Print recall comparison (sample)
    println!("\n=== Recall@{k} vs Full Precision ===");

    let ground_truth = compute_ground_truth(&corpus, &queries, k);
    let evaluator = Evaluator::new(k);

    // SQ8 recall
    let mut sq8_evals = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let mut scores: Vec<(usize, f32)> = sq8_corpus
            .iter()
            .enumerate()
            .map(|(j, enc)| (j, sq8.asymmetric_inner_product(query, enc)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let retrieved: Vec<String> = scores.into_iter().take(k).map(|(j, _)| format!("doc-{j}")).collect();
        let eval = evaluator.evaluate_query(format!("q-{i}"), &retrieved, &ground_truth[i], start.elapsed());
        sq8_evals.push(eval);
    }
    let sq8_summary = EvaluationSummary::from_evaluations(&sq8_evals);
    println!("SQ8 (4x):    Recall@{k} = {:.4}", sq8_summary.mean_recall);

    // PQ-16 recall
    let mut pq16_evals = Vec::new();
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let table = pq16.compute_distance_table(query);
        let mut scores: Vec<(usize, f32)> = pq16_corpus
            .iter()
            .enumerate()
            .map(|(j, enc)| (j, -pq16.asymmetric_l2_squared_with_table(&table, enc)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let retrieved: Vec<String> = scores.into_iter().take(k).map(|(j, _)| format!("doc-{j}")).collect();
        let eval = evaluator.evaluate_query(format!("q-{i}"), &retrieved, &ground_truth[i], start.elapsed());
        pq16_evals.push(eval);
    }
    let pq16_summary = EvaluationSummary::from_evaluations(&pq16_evals);
    println!("PQ-16 (32x): Recall@{k} = {:.4}", pq16_summary.mean_recall);

    group.finish();
}

// =============================================================================
// BENCHMARK: SEQUENTIAL VS PARALLEL HNSW CONSTRUCTION
// =============================================================================

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index/hnsw_construction");
    group.sample_size(10);

    println!("\n=== HNSW Construction Benchmark ===");

    for (n_docs, dim) in [(5000, 128), (10000, 128), (20000, 64)] {
        let vectors: Vec<(String, Vec<f32>)> = generate_vectors(n_docs, dim, 42)
            .into_iter()
            .enumerate()
            .map(|(i, v)| (format!("doc-{i}"), v))
            .collect();

        let config = HNSWConfig::new(dim)
            .with_m(16)
            .with_ef_construction(100);

        println!("\nDataset: {} vectors x {} dim", n_docs, dim);

        // === Sequential HNSW ===
        group.throughput(Throughput::Elements(n_docs as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{}x{}", n_docs, dim)),
            &vectors,
            |bench, vectors| {
                bench.iter(|| {
                    let mut index = HNSWIndex::new(config.clone());
                    for (id, vec) in vectors {
                        index.add(id.clone(), vec).unwrap();
                    }
                    black_box(index)
                })
            },
        );

        // === Parallel HNSW (default threads) ===
        group.bench_with_input(
            BenchmarkId::new("parallel_auto", format!("{}x{}", n_docs, dim)),
            &vectors,
            |bench, vectors| {
                bench.iter(|| {
                    let index = ParallelHNSWBuilder::new()
                        .with_config(config.clone())
                        .with_seed(42)
                        .build(vectors.clone())
                        .unwrap();
                    black_box(index)
                })
            },
        );

        // === Parallel HNSW (4 threads) ===
        group.bench_with_input(
            BenchmarkId::new("parallel_4t", format!("{}x{}", n_docs, dim)),
            &vectors,
            |bench, vectors| {
                bench.iter(|| {
                    let index = ParallelHNSWBuilder::new()
                        .with_config(config.clone())
                        .with_threads(4)
                        .with_seed(42)
                        .build(vectors.clone())
                        .unwrap();
                    black_box(index)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK: HNSW SEARCH PERFORMANCE
// =============================================================================

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search/hnsw");
    group.sample_size(30);

    let n_docs = 100000;
    let dim = 128;
    let n_queries = 100;

    println!("\n=== HNSW Search Benchmark ===");
    println!("Corpus: {} vectors x {} dimensions", n_docs, dim);

    let vectors: Vec<(String, Vec<f32>)> = generate_vectors(n_docs, dim, 42)
        .into_iter()
        .enumerate()
        .map(|(i, v)| (format!("doc-{i}"), v))
        .collect();

    let queries = generate_vectors(n_queries, dim, 123);

    // Build HNSW with parallel builder
    let config = HNSWConfig::new(dim)
        .with_m(16)
        .with_ef_construction(100);

    println!("Building HNSW index...");
    let start = Instant::now();
    let index = ParallelHNSWBuilder::new()
        .with_config(config)
        .with_seed(42)
        .build(vectors)
        .unwrap();
    println!("Built in {:?}", start.elapsed());

    // Search benchmarks with different ef values
    for ef in [10, 50, 100, 200] {
        let k = 10;
        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(
            BenchmarkId::new("search", format!("ef={}", ef)),
            &ef,
            |bench, &ef| {
                bench.iter(|| {
                    for query in &queries {
                        // Note: ef is used in search, but current API doesn't expose it
                        // This benchmark uses default ef
                        let _ = index.search(black_box(query), k);
                    }
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    benches,
    bench_dense_vs_hybrid,
    bench_memory_compression,
    bench_hnsw_construction,
    bench_hnsw_search,
);

criterion_main!(benches);
