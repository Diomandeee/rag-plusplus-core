//! Tracing Spans
//!
//! Structured tracing for query execution.

use std::time::{Duration, Instant};
use tracing::{info_span, Span};

/// Context for distributed tracing.
#[derive(Debug, Clone, Default)]
pub struct SpanContext {
    /// Trace ID (for distributed tracing)
    pub trace_id: Option<String>,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Additional baggage
    pub baggage: Vec<(String, String)>,
}

impl SpanContext {
    /// Create empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with trace ID.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Add baggage item.
    #[must_use]
    pub fn with_baggage(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.baggage.push((key.into(), value.into()));
        self
    }
}

/// Query execution span.
///
/// Tracks timing and attributes for a query.
pub struct QuerySpan {
    span: Span,
    start: Instant,
    #[allow(dead_code)]
    k: usize,
    index_count: usize,
}

impl QuerySpan {
    /// Start a new query span.
    #[must_use]
    pub fn start(k: usize, has_filter: bool, context: Option<&SpanContext>) -> Self {
        let span = info_span!(
            "ragpp.query",
            k = k,
            has_filter = has_filter,
            otel.kind = "client",
        );

        // Add trace context if provided
        if let Some(ctx) = context {
            if let Some(ref trace_id) = ctx.trace_id {
                span.record("trace_id", trace_id.as_str());
            }
        }

        Self {
            span,
            start: Instant::now(),
            k,
            index_count: 0,
        }
    }

    /// Enter the span.
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }

    /// Record index search.
    pub fn record_index_search(&mut self, index_name: &str, candidates: usize, latency: Duration) {
        self.index_count += 1;

        tracing::info!(
            parent: &self.span,
            index = index_name,
            candidates = candidates,
            latency_ms = latency.as_millis() as u64,
            "index search completed"
        );
    }

    /// Record reranking.
    pub fn record_rerank(&self, input_count: usize, output_count: usize, latency: Duration) {
        tracing::info!(
            parent: &self.span,
            input_count = input_count,
            output_count = output_count,
            latency_ms = latency.as_millis() as u64,
            "reranking completed"
        );
    }

    /// Record filter application.
    pub fn record_filter(&self, before: usize, after: usize) {
        tracing::info!(
            parent: &self.span,
            before = before,
            after = after,
            filtered = before - after,
            "filter applied"
        );
    }

    /// Finish the span with results.
    pub fn finish(self, result_count: usize, error: Option<&str>) {
        let latency = self.start.elapsed();

        if let Some(err) = error {
            tracing::error!(
                parent: &self.span,
                result_count = result_count,
                latency_ms = latency.as_millis() as u64,
                indexes_searched = self.index_count,
                error = err,
                "query failed"
            );
        } else {
            tracing::info!(
                parent: &self.span,
                result_count = result_count,
                latency_ms = latency.as_millis() as u64,
                indexes_searched = self.index_count,
                "query completed"
            );
        }
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Create a span for index operations.
#[allow(dead_code)]
pub fn index_span(operation: &str, index_name: &str) -> Span {
    info_span!(
        "ragpp.index",
        operation = operation,
        index = index_name,
    )
}

/// Create a span for store operations.
#[allow(dead_code)]
pub fn store_span(operation: &str) -> Span {
    info_span!(
        "ragpp.store",
        operation = operation,
    )
}

/// Create a span for WAL operations.
#[allow(dead_code)]
pub fn wal_span(operation: &str) -> Span {
    info_span!(
        "ragpp.wal",
        operation = operation,
    )
}

/// Log structured event.
#[macro_export]
macro_rules! ragpp_event {
    ($level:expr, $($field:tt)*) => {
        tracing::event!($level, $($field)*);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_context() {
        let ctx = SpanContext::new()
            .with_trace_id("abc123")
            .with_baggage("user_id", "42");

        assert_eq!(ctx.trace_id, Some("abc123".to_string()));
        assert_eq!(ctx.baggage.len(), 1);
    }

    #[test]
    fn test_query_span() {
        let mut span = QuerySpan::start(10, false, None);

        span.record_index_search("test", 100, Duration::from_millis(5));
        span.record_rerank(100, 10, Duration::from_millis(2));

        assert!(span.elapsed() >= Duration::from_millis(0));
        span.finish(10, None);
    }

    #[test]
    fn test_query_span_with_error() {
        let span = QuerySpan::start(10, false, None);
        span.finish(0, Some("timeout"));
    }
}
