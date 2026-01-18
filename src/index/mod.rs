//! Index Module
//!
//! Provides vector index implementations for approximate nearest neighbor search.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │              VectorIndex Trait              │
//! ├─────────────────────────────────────────────┤
//! │  + add(&mut self, id, vector)               │
//! │  + search(&self, query, k) -> Results       │
//! │  + remove(&mut self, id)                    │
//! │  + len(&self) -> usize                      │
//! └─────────────────────────────────────────────┘
//!            ▲              ▲              ▲
//!            │              │              │
//!     ┌──────┴──────┐ ┌────┴────┐ ┌───────┴───────┐
//!     │  FlatIndex  │ │HNSWIndex│ │  IVFIndex     │
//!     │ (exact)     │ │(approx) │ │  (approx)     │
//!     └─────────────┘ └─────────┘ └───────────────┘
//! ```
//!
//! # Invariants
//!
//! - INV-004: Index-record consistency (every ID in index has valid record)

mod flat;
mod fusion;
mod hnsw;
mod parallel;
mod registry;
mod traits;

pub use flat::FlatIndex;
pub use fusion::{FusedResult, FusionConfig, FusionStrategy, ScoreFusion, rrf_fuse, rrf_fuse_top_k};
pub use hnsw::{HNSWConfig, HNSWIndex};
pub use parallel::{ParallelSearchConfig, ParallelSearcher, ResultsAggregator, parallel_add_batch};
pub use registry::{IndexInfo, IndexRegistry, MultiIndexResult, MultiIndexResults, SharedRegistry, shared_registry};
pub use traits::{SearchResult, VectorIndex, IndexConfig, DistanceType};
