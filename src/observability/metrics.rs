//! Metrics Module
//!
//! Prometheus-compatible metrics for RAG++ operations.

use metrics::{counter, gauge, histogram};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Metrics configuration.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Prefix for all metric names
    pub prefix: String,
    /// Enable detailed per-index metrics
    pub per_index_metrics: bool,
    /// Histogram buckets for latency (in seconds)
    pub latency_buckets: Vec<f64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prefix: "ragpp".into(),
            per_index_metrics: true,
            latency_buckets: vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
        }
    }
}

impl MetricsConfig {
    /// Create new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set prefix.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }
}

/// Central metrics collector for RAG++.
///
/// Provides counters, gauges, and histograms for monitoring.
///
/// # Metrics
///
/// - `ragpp_queries_total` - Total queries processed
/// - `ragpp_query_latency_seconds` - Query latency histogram
/// - `ragpp_index_size` - Number of vectors per index
/// - `ragpp_cache_hits_total` - Cache hit count
/// - `ragpp_cache_misses_total` - Cache miss count
/// - `ragpp_wal_writes_total` - WAL write count
/// - `ragpp_buffer_flushes_total` - Buffer flush count
#[derive(Debug)]
pub struct Metrics {
    config: MetricsConfig,
    // Counters (using atomics for thread-safety without locking)
    queries_total: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    wal_writes: AtomicU64,
    buffer_flushes: AtomicU64,
    errors_total: AtomicU64,
}

impl Metrics {
    /// Create new metrics collector.
    #[must_use]
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            queries_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            wal_writes: AtomicU64::new(0),
            buffer_flushes: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
        }
    }

    /// Create with default config.
    #[must_use]
    pub fn default_metrics() -> Self {
        Self::new(MetricsConfig::default())
    }

    /// Record a query.
    pub fn record_query(&self, latency: Duration, result_count: usize, index_name: Option<&str>) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);

        let latency_secs = latency.as_secs_f64();
        let prefix = &self.config.prefix;

        // Record to metrics crate
        if let Some(name) = index_name {
            histogram!(format!("{prefix}_query_latency_seconds"), "index" => name.to_string())
                .record(latency_secs);
            counter!(format!("{prefix}_queries_total"), "index" => name.to_string())
                .increment(1);
            gauge!(format!("{prefix}_query_results"), "index" => name.to_string())
                .set(result_count as f64);
        } else {
            histogram!(format!("{prefix}_query_latency_seconds"))
                .record(latency_secs);
            counter!(format!("{prefix}_queries_total"))
                .increment(1);
            gauge!(format!("{prefix}_query_results"))
                .set(result_count as f64);
        }
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        counter!(format!("{}_cache_hits_total", self.config.prefix)).increment(1);
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        counter!(format!("{}_cache_misses_total", self.config.prefix)).increment(1);
    }

    /// Record a WAL write.
    pub fn record_wal_write(&self) {
        self.wal_writes.fetch_add(1, Ordering::Relaxed);
        counter!(format!("{}_wal_writes_total", self.config.prefix)).increment(1);
    }

    /// Record a buffer flush.
    pub fn record_buffer_flush(&self, records_flushed: usize) {
        self.buffer_flushes.fetch_add(1, Ordering::Relaxed);
        counter!(format!("{}_buffer_flushes_total", self.config.prefix)).increment(1);
        counter!(format!("{}_records_flushed_total", self.config.prefix))
            .increment(records_flushed as u64);
    }

    /// Record an error.
    pub fn record_error(&self, error_type: &str) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
        counter!(
            format!("{}_errors_total", self.config.prefix),
            "type" => error_type.to_string()
        )
        .increment(1);
    }

    /// Update index size gauge.
    pub fn set_index_size(&self, index_name: &str, size: usize) {
        gauge!(
            format!("{}_index_size", self.config.prefix),
            "index" => index_name.to_string()
        )
        .set(size as f64);
    }

    /// Update store size gauge.
    pub fn set_store_size(&self, size: usize) {
        gauge!(format!("{}_store_size", self.config.prefix)).set(size as f64);
    }

    /// Update memory usage gauge.
    pub fn set_memory_bytes(&self, bytes: usize) {
        gauge!(format!("{}_memory_bytes", self.config.prefix)).set(bytes as f64);
    }

    /// Get snapshot of current metrics.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            queries_total: self.queries_total.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            wal_writes: self.wal_writes.load(Ordering::Relaxed),
            buffer_flushes: self.buffer_flushes.load(Ordering::Relaxed),
            errors_total: self.errors_total.load(Ordering::Relaxed),
        }
    }

    /// Calculate cache hit ratio.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::default_metrics()
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// Total queries
    pub queries_total: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// WAL writes
    pub wal_writes: u64,
    /// Buffer flushes
    pub buffer_flushes: u64,
    /// Errors
    pub errors_total: u64,
}

impl MetricsSnapshot {
    /// Calculate cache hit ratio.
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Timer for measuring operation duration.
#[allow(dead_code)]
pub struct Timer {
    start: Instant,
}

#[allow(dead_code)]
impl Timer {
    /// Start a new timer.
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed duration.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop timer and return duration.
    #[must_use]
    pub fn stop(self) -> Duration {
        self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::default_metrics();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries_total, 0);
    }

    #[test]
    fn test_record_query() {
        let metrics = Metrics::default_metrics();

        metrics.record_query(Duration::from_millis(50), 10, Some("test"));
        metrics.record_query(Duration::from_millis(100), 5, None);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries_total, 2);
    }

    #[test]
    fn test_cache_metrics() {
        let metrics = Metrics::default_metrics();

        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.cache_hits, 2);
        assert_eq!(snapshot.cache_misses, 1);
        assert!((metrics.cache_hit_ratio() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_config_builder() {
        let config = MetricsConfig::new().with_prefix("myapp");
        assert_eq!(config.prefix, "myapp");
    }
}
