//! Evaluation Module for RAG++
//!
//! Provides metrics and utilities for evaluating retrieval quality.
//!
//! # Metrics
//!
//! - **Recall@K**: Fraction of relevant items retrieved in top-K
//! - **Precision@K**: Fraction of retrieved items that are relevant
//! - **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant item
//! - **NDCG@K**: Normalized Discounted Cumulative Gain
//! - **Hit Rate@K**: Whether at least one relevant item is in top-K

use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Evaluation result for a single query.
#[derive(Debug, Clone)]
pub struct QueryEvaluation {
    /// Query identifier
    pub query_id: String,
    /// Recall at K
    pub recall: f64,
    /// Precision at K
    pub precision: f64,
    /// Reciprocal rank (1/rank of first relevant item, 0 if none)
    pub reciprocal_rank: f64,
    /// NDCG at K
    pub ndcg: f64,
    /// Whether any relevant item was in top-K
    pub hit: bool,
    /// Query latency
    pub latency: Duration,
    /// Number of results returned
    pub num_results: usize,
}

/// Aggregated evaluation metrics across multiple queries.
#[derive(Debug, Clone, Default)]
pub struct EvaluationSummary {
    /// Number of queries evaluated
    pub num_queries: usize,
    /// Mean Recall@K
    pub mean_recall: f64,
    /// Mean Precision@K
    pub mean_precision: f64,
    /// Mean Reciprocal Rank
    pub mrr: f64,
    /// Mean NDCG@K
    pub mean_ndcg: f64,
    /// Hit rate (fraction of queries with at least one hit)
    pub hit_rate: f64,
    /// Mean latency
    pub mean_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P95 latency
    pub p95_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
}

impl EvaluationSummary {
    /// Create summary from individual evaluations.
    pub fn from_evaluations(evals: &[QueryEvaluation]) -> Self {
        if evals.is_empty() {
            return Self::default();
        }

        let n = evals.len() as f64;

        let mean_recall = evals.iter().map(|e| e.recall).sum::<f64>() / n;
        let mean_precision = evals.iter().map(|e| e.precision).sum::<f64>() / n;
        let mrr = evals.iter().map(|e| e.reciprocal_rank).sum::<f64>() / n;
        let mean_ndcg = evals.iter().map(|e| e.ndcg).sum::<f64>() / n;
        let hit_rate = evals.iter().filter(|e| e.hit).count() as f64 / n;

        let mean_latency_nanos = evals.iter().map(|e| e.latency.as_nanos()).sum::<u128>() / evals.len() as u128;
        let mean_latency = Duration::from_nanos(mean_latency_nanos as u64);

        // Compute latency percentiles
        let mut latencies: Vec<Duration> = evals.iter().map(|e| e.latency).collect();
        latencies.sort();

        let p50_idx = (evals.len() as f64 * 0.50) as usize;
        let p95_idx = (evals.len() as f64 * 0.95) as usize;
        let p99_idx = (evals.len() as f64 * 0.99) as usize;

        let p50_latency = latencies.get(p50_idx.min(latencies.len() - 1)).copied().unwrap_or_default();
        let p95_latency = latencies.get(p95_idx.min(latencies.len() - 1)).copied().unwrap_or_default();
        let p99_latency = latencies.get(p99_idx.min(latencies.len() - 1)).copied().unwrap_or_default();

        Self {
            num_queries: evals.len(),
            mean_recall,
            mean_precision,
            mrr,
            mean_ndcg,
            hit_rate,
            mean_latency,
            p50_latency,
            p95_latency,
            p99_latency,
        }
    }

    /// Print a formatted summary report.
    pub fn report(&self) -> String {
        format!(
            r#"
=== RAG++ Evaluation Summary ===
Queries evaluated: {}

Retrieval Quality:
  Mean Recall@K:    {:.4}
  Mean Precision@K: {:.4}
  MRR:              {:.4}
  Mean NDCG@K:      {:.4}
  Hit Rate:         {:.2}%

Latency:
  Mean:  {:?}
  P50:   {:?}
  P95:   {:?}
  P99:   {:?}
================================
"#,
            self.num_queries,
            self.mean_recall,
            self.mean_precision,
            self.mrr,
            self.mean_ndcg,
            self.hit_rate * 100.0,
            self.mean_latency,
            self.p50_latency,
            self.p95_latency,
            self.p99_latency,
        )
    }
}

/// Evaluator for computing retrieval metrics.
pub struct Evaluator {
    /// K for recall@K, precision@K, etc.
    k: usize,
}

impl Evaluator {
    /// Create new evaluator with given K.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Evaluate a single query.
    ///
    /// # Arguments
    /// * `query_id` - Identifier for this query
    /// * `retrieved_ids` - IDs returned by the retrieval system (in ranked order)
    /// * `relevant_ids` - Ground truth relevant IDs
    /// * `latency` - Query execution time
    pub fn evaluate_query(
        &self,
        query_id: impl Into<String>,
        retrieved_ids: &[String],
        relevant_ids: &HashSet<String>,
        latency: Duration,
    ) -> QueryEvaluation {
        let k = self.k.min(retrieved_ids.len());
        let top_k: Vec<_> = retrieved_ids.iter().take(k).collect();

        // Recall@K: How many relevant items did we find?
        let relevant_found = top_k.iter().filter(|id| relevant_ids.contains(id.as_str())).count();
        let recall = if relevant_ids.is_empty() {
            1.0 // No relevant items means perfect recall
        } else {
            relevant_found as f64 / relevant_ids.len() as f64
        };

        // Precision@K: How many retrieved items are relevant?
        let precision = if k == 0 {
            0.0
        } else {
            relevant_found as f64 / k as f64
        };

        // Reciprocal Rank: 1/rank of first relevant item
        let reciprocal_rank = top_k
            .iter()
            .position(|id| relevant_ids.contains(id.as_str()))
            .map(|pos| 1.0 / (pos + 1) as f64)
            .unwrap_or(0.0);

        // Hit: Did we find at least one relevant item?
        let hit = relevant_found > 0;

        // NDCG@K
        let ndcg = self.compute_ndcg(&top_k, relevant_ids);

        QueryEvaluation {
            query_id: query_id.into(),
            recall,
            precision,
            reciprocal_rank,
            ndcg,
            hit,
            latency,
            num_results: retrieved_ids.len(),
        }
    }

    /// Compute NDCG (Normalized Discounted Cumulative Gain).
    fn compute_ndcg(&self, retrieved: &[&String], relevant: &HashSet<String>) -> f64 {
        if relevant.is_empty() {
            return 1.0;
        }

        // DCG: sum of relevance / log2(rank + 1)
        let dcg: f64 = retrieved
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let rel = if relevant.contains(id.as_str()) { 1.0 } else { 0.0 };
                rel / (i as f64 + 2.0).log2()
            })
            .sum();

        // Ideal DCG: all relevant items at top
        let ideal_k = self.k.min(relevant.len());
        let idcg: f64 = (0..ideal_k)
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum();

        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }
}

/// Benchmark runner for performance evaluation.
pub struct Benchmarker {
    /// Warm-up iterations
    warmup_iters: usize,
    /// Measurement iterations
    measure_iters: usize,
}

impl Benchmarker {
    /// Create new benchmarker.
    #[must_use]
    pub fn new(warmup_iters: usize, measure_iters: usize) -> Self {
        Self {
            warmup_iters,
            measure_iters,
        }
    }

    /// Run benchmark on a function.
    pub fn run<F>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warm-up
        for _ in 0..self.warmup_iters {
            f();
        }

        // Measure
        let mut durations = Vec::with_capacity(self.measure_iters);
        for _ in 0..self.measure_iters {
            let start = Instant::now();
            f();
            durations.push(start.elapsed());
        }

        // Compute stats
        durations.sort();
        let total: Duration = durations.iter().sum();
        let mean = total / self.measure_iters as u32;

        let p50 = durations[durations.len() / 2];
        let p95 = durations[(durations.len() as f64 * 0.95) as usize];
        let p99 = durations[(durations.len() as f64 * 0.99) as usize];
        let min = durations[0];
        let max = durations[durations.len() - 1];

        BenchmarkResult {
            iterations: self.measure_iters,
            mean,
            p50,
            p95,
            p99,
            min,
            max,
        }
    }
}

/// Result of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of measured iterations
    pub iterations: usize,
    /// Mean duration
    pub mean: Duration,
    /// P50 (median) duration
    pub p50: Duration,
    /// P95 duration
    pub p95: Duration,
    /// P99 duration
    pub p99: Duration,
    /// Minimum duration
    pub min: Duration,
    /// Maximum duration
    pub max: Duration,
}

impl BenchmarkResult {
    /// Throughput in operations per second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        1.0 / self.mean.as_secs_f64()
    }

    /// Print formatted benchmark report.
    pub fn report(&self, name: &str) -> String {
        format!(
            r#"
=== Benchmark: {} ===
Iterations: {}
Mean:       {:?}
P50:        {:?}
P95:        {:?}
P99:        {:?}
Min:        {:?}
Max:        {:?}
Throughput: {:.2} ops/sec
======================
"#,
            name,
            self.iterations,
            self.mean,
            self.p50,
            self.p95,
            self.p99,
            self.min,
            self.max,
            self.throughput(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_recall() {
        let evaluator = Evaluator::new(10);
        let retrieved: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let relevant: HashSet<String> = (0..5).map(|i| format!("doc-{i}")).collect();

        let eval = evaluator.evaluate_query("q1", &retrieved, &relevant, Duration::from_millis(10));

        assert_eq!(eval.recall, 1.0); // All 5 relevant items in top 10
        assert_eq!(eval.precision, 0.5); // 5 of 10 retrieved are relevant
        assert!(eval.hit);
    }

    #[test]
    fn test_no_relevant_items() {
        let evaluator = Evaluator::new(10);
        let retrieved: Vec<String> = (0..10).map(|i| format!("doc-{i}")).collect();
        let relevant: HashSet<String> = HashSet::new();

        let eval = evaluator.evaluate_query("q1", &retrieved, &relevant, Duration::from_millis(10));

        assert_eq!(eval.recall, 1.0); // No relevant = perfect recall
        assert_eq!(eval.precision, 0.0); // No relevant = 0 precision
        assert!(!eval.hit);
    }

    #[test]
    fn test_mrr() {
        let evaluator = Evaluator::new(10);

        // Relevant item at position 0
        let retrieved1 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let relevant1: HashSet<_> = ["a".to_string()].into();
        let eval1 = evaluator.evaluate_query("q1", &retrieved1, &relevant1, Duration::ZERO);
        assert!((eval1.reciprocal_rank - 1.0).abs() < 1e-6);

        // Relevant item at position 2
        let retrieved2 = vec!["x".to_string(), "y".to_string(), "a".to_string()];
        let eval2 = evaluator.evaluate_query("q2", &retrieved2, &relevant1, Duration::ZERO);
        assert!((eval2.reciprocal_rank - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluation_summary() {
        let evals = vec![
            QueryEvaluation {
                query_id: "q1".into(),
                recall: 1.0,
                precision: 0.5,
                reciprocal_rank: 1.0,
                ndcg: 0.8,
                hit: true,
                latency: Duration::from_millis(10),
                num_results: 10,
            },
            QueryEvaluation {
                query_id: "q2".into(),
                recall: 0.5,
                precision: 0.25,
                reciprocal_rank: 0.5,
                ndcg: 0.6,
                hit: true,
                latency: Duration::from_millis(20),
                num_results: 10,
            },
        ];

        let summary = EvaluationSummary::from_evaluations(&evals);

        assert_eq!(summary.num_queries, 2);
        assert!((summary.mean_recall - 0.75).abs() < 1e-6);
        assert!((summary.mrr - 0.75).abs() < 1e-6);
        assert_eq!(summary.hit_rate, 1.0);
    }

    #[test]
    fn test_benchmarker() {
        let benchmarker = Benchmarker::new(2, 10);
        let mut counter = 0;

        let result = benchmarker.run(|| {
            counter += 1;
            std::thread::sleep(Duration::from_micros(100));
        });

        assert_eq!(counter, 12); // 2 warmup + 10 measured
        assert!(result.mean >= Duration::from_micros(100));
        assert!(result.throughput() > 0.0);
    }
}
