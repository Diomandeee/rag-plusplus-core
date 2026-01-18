//! RAG++ Core Library
//!
//! High-performance retrieval engine for memory-conditioned candidate selection.
//!
//! # Overview
//!
//! RAG++ retrieves outcome-annotated trajectories and provides statistical priors
//! for downstream conditioning. It does NOT train models - pure retrieval + statistics.
//!
//! # Invariants
//!
//! - INV-001: Record immutability after creation
//! - INV-002: Welford numerical stability
//! - INV-003: WAL-before-buffer for durability
//! - INV-004: Index-record consistency
//!
//! # Modules
//!
//! - `types`: Core data structures (MemoryRecord, QueryBundle, PriorBundle)
//! - `stats`: Outcome statistics with Welford's algorithm
//! - `error`: Error types and Result aliases
//! - `index`: Vector index implementations (Flat, HNSW)
//! - `filter`: Metadata filter expressions and evaluation
//! - `store`: Record storage (in-memory, persistent)
//! - `wal`: Write-ahead log for durability
//! - `buffer`: Write buffering for batched operations
//! - `retrieval`: Query execution and reranking
//! - `cache`: Query result caching
//! - `observability`: Metrics and tracing
//! - `eval`: Evaluation metrics and benchmarking
//! - `api`: Public API contracts and traits
//! - `trajectory`: Trajectory memory algorithms (DAG traversal, phase inference, salience)
//! - `distance`: Vector distance operations with SIMD and batch support

#![deny(unsafe_code)]  // deny instead of forbid to allow SIMD module override
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod api;
pub mod buffer;
pub mod cache;
pub mod distance;
pub mod error;
pub mod eval;
pub mod filter;
pub mod index;
pub mod observability;
pub mod retrieval;
pub mod stats;
pub mod store;
pub mod trajectory;
pub mod types;
pub mod wal;

// Re-exports for convenience
pub use buffer::{BufferStats, WriteBuffer, WriteBufferConfig};
pub use cache::{CacheConfig, CacheKey, CacheStats, QueryCache};
pub use error::{Error, Result};
pub use filter::{CompiledFilter, FilterEvaluator, FilterExpr, FilterOp, FilterValue};
pub use index::{
    DistanceType, FlatIndex, FusedResult, FusionConfig, FusionStrategy, HNSWConfig, HNSWIndex,
    IndexConfig, IndexInfo, IndexRegistry, MultiIndexResult, MultiIndexResults, ParallelSearchConfig,
    ParallelSearcher, ResultsAggregator, ScoreFusion, SearchResult, SharedRegistry, VectorIndex,
    parallel_add_batch, rrf_fuse, rrf_fuse_top_k, shared_registry,
};
pub use stats::OutcomeStats;
pub use store::{InMemoryStore, RecordStore, SharedStore};
pub use types::{MemoryRecord, PriorBundle, QueryBundle, RecordId, VectorRef};
pub use observability::{Metrics, MetricsConfig, QuerySpan, SpanContext};
pub use retrieval::{QueryEngine, QueryEngineConfig, QueryRequest, QueryResponse, Reranker, RerankerConfig, RerankerType};
pub use wal::{WalConfig, WalEntry, WalEntryType, WalReader, WalWriter};
pub use eval::{Evaluator, EvaluationSummary, QueryEvaluation, Benchmarker, BenchmarkResult};

// Trajectory memory algorithms
pub use trajectory::{
    TrajectoryGraph, Episode, Edge, EdgeType, NodeId as EpisodeId,
    PathSelectionPolicy, TraversalOrder, BranchInfo, PathResult,
    TrajectoryPhase, PhaseInferencer, PhaseConfig, TurnFeatures, PhaseTransition,
    SalienceScorer, SalienceConfig, SalienceFactors, TurnSalience, CorpusSalienceStats, Feedback,
    TrajectoryCoordinate, NormalizedCoordinate, compute_trajectory_coordinates,
    // 5D coordinates (TPO extension with complexity dimension)
    TrajectoryCoordinate5D, NormalizedCoordinate5D,
    Ring, RingNode, EpisodeRing, build_weighted_ring,
    // Dual ring for IRCP/RCP
    DualRing, DualRingNode, build_dual_ring, build_dual_ring_with_attention,
    ConservationMetrics, ConservationViolation, ConservationConfig, ConservationTracker,
    weighted_centroid, weighted_covariance,
    // Path quality (TPO integration)
    PathQuality, PathQualityWeights, PathQualityFactors,
};

// Backward compatibility - deprecated in favor of TrajectoryPhase
#[allow(deprecated)]
pub use trajectory::ConversationPhase;

// Distance operations with batch support
pub use distance::{
    l2_distance, l2_distance_squared, inner_product, cosine_similarity, cosine_distance,
    norm, norm_squared, normalize, normalize_in_place,
    normalize_batch, normalize_batch_flat, compute_norms_batch, find_unnormalized,
    l2_distance_fast, inner_product_fast, cosine_similarity_fast, cosine_distance_fast,
    norm_fast, normalize_in_place_fast, normalize_batch_flat_fast, compute_norms_batch_fast,
    normalize_batch_parallel, compute_distance, compute_distance_for_heap,
    // Trajectory-weighted distance (TPO integration)
    trajectory_weighted_cosine, trajectory_weighted_cosine_5d, trajectory_weighted_l2,
    trajectory_weighted_inner_product, trajectory_weighted_distance, TrajectoryDistanceConfig,
};

// Public API contracts
pub use api::{
    RetrievalRequest, RetrievalResponse, PriorBundle as ApiPriorBundle,
    RankedCandidate, FilterExpression, FilterValue as ApiFilterValue,
    IngestRecord, MetadataValue as ApiMetadataValue,
    RetrievalEngine, Corpus, VectorSearcher, SearchHit,
    RAGBuilder, IndexType,
};
