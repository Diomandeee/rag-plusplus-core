//! Caching Module
//!
//! Provides query result caching for improved performance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    QueryCache                                │
//! ├─────────────────────────────────────────────────────────────┤
//! │  LRU eviction policy                                         │
//! │  TTL-based expiration                                        │
//! │  Thread-safe access                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Cache Key
//!
//! Cache keys are computed from:
//! - Query embedding (hashed)
//! - k value
//! - Filter expression (if any)
//! - Index names (if specified)

mod query_cache;

pub use query_cache::{CacheConfig, CacheEntry, CacheKey, CacheStats, QueryCache};
