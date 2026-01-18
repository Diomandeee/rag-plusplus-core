//! Retrieval Module
//!
//! End-to-end query processing and result retrieval.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     QueryEngine                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │  1. Validate query                                           │
//! │  2. Pre-filter candidates (optional)                         │
//! │  3. Vector search                                            │
//! │  4. Fetch records from store                                 │
//! │  5. Rerank results                                           │
//! │  6. Build priors                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Query Types
//!
//! - Single-index query
//! - Multi-index query with fusion
//! - Filtered query with metadata constraints

pub mod engine;
pub mod rerank;

pub use engine::{QueryEngine, QueryEngineConfig, QueryRequest, QueryResponse};
pub use rerank::{RerankerConfig, RerankerType, Reranker};
