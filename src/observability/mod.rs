//! Observability Module
//!
//! Provides metrics, tracing, and logging for production monitoring.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Observability                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Metrics:                                                    │
//! │    - Query latency histogram                                 │
//! │    - Query throughput counter                                │
//! │    - Index size gauge                                        │
//! │    - Cache hit/miss ratio                                    │
//! │                                                              │
//! │  Tracing:                                                    │
//! │    - Request spans with attributes                           │
//! │    - Component-level timing                                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod metrics;
mod spans;

pub use metrics::{Metrics, MetricsConfig};
pub use spans::{QuerySpan, SpanContext};
