# rag-plusplus-core

[![crates.io](https://img.shields.io/crates/v/rag-plusplus-core.svg)](https://crates.io/crates/rag-plusplus-core)
[![docs.rs](https://docs.rs/rag-plusplus-core/badge.svg)](https://docs.rs/rag-plusplus-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**High-performance retrieval engine in Rust with SIMD-accelerated vector search and trajectory memory.**

rag-plusplus-core provides the foundational algorithms for trajectory-aware retrieval-augmented generation. It implements 5D coordinate positioning, dual-ring attention structures, and conservation-validated bounded forgetting, all optimized with AVX2/NEON SIMD intrinsics for production performance.

## Key Features

- **SIMD-Accelerated Distances** - AVX2 and NEON optimized L2, cosine, and inner product with 4-8x speedup over scalar
- **HNSW Index** - Hierarchical Navigable Small World graph for O(log n) approximate nearest neighbor search
- **5D Trajectory Coordinates** - Novel positioning system (depth, sibling_order, homogeneity, temporal, complexity)
- **Dual-Ring Topology** - RCP (temporal) and IRCP (influence) orderings for structure-aware retrieval
- **Conservation Metrics** - Magnitude, energy, and information conservation for bounded forgetting validation
- **Salience Scoring** - Multi-factor importance weighting with feedback, phase transitions, and novelty
- **Write-Ahead Log** - Durable operations with checkpoint/recovery for crash safety
- **Query Result Caching** - LRU cache with configurable TTL for repeated queries
- **Product Quantization** - Memory-efficient vector compression (SQ8, PQ) for large-scale deployment

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rag-plusplus-core = "0.1"
```

## Quick Start

```rust
use rag_plusplus_core::index::{HNSWIndex, HNSWConfig, VectorIndex, SearchResult};

// Create an HNSW index for 768-dimensional vectors
let config = HNSWConfig::new(768)
    .with_m(16)                    // Connections per layer
    .with_ef_construction(200)     // Build-time search depth
    .with_ef_search(128);          // Query-time search depth

let mut index = HNSWIndex::new(config);

// Add vectors with string IDs
index.add("doc-1".to_string(), &[0.1; 768])?;
index.add("doc-2".to_string(), &[0.2; 768])?;
index.add("doc-3".to_string(), &[0.15; 768])?;

// Search for k nearest neighbors
let query = [0.12; 768];
let results: Vec<SearchResult> = index.search(&query, 10)?;

for result in results {
    println!("Found: {} (distance: {:.4})", result.id, result.distance);
}
```

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  RAG++ Core (Rust)                       │
                    ├─────────────────────────────────────────────────────────┤
                    │  ┌───────────────────────────────────────────────────┐  │
                    │  │              Vector Index Layer                    │  │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │  │
                    │  │  │FlatIndex │  │HNSWIndex │  │ FusionIndex  │     │  │
                    │  │  │ (exact)  │  │ (approx) │  │  (hybrid)    │     │  │
                    │  │  └──────────┘  └──────────┘  └──────────────┘     │  │
                    │  └───────────────────────────────────────────────────┘  │
                    │  ┌───────────────────────────────────────────────────┐  │
                    │  │            Distance Layer (SIMD)                   │  │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │  │
                    │  │  │ L2 AVX2  │  │ Cosine   │  │ Inner Product│     │  │
                    │  │  └──────────┘  └──────────┘  └──────────────┘     │  │
                    │  └───────────────────────────────────────────────────┘  │
                    │  ┌───────────────────────────────────────────────────┐  │
                    │  │              Trajectory Memory                     │  │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │  │
                    │  │  │5D Coords │  │ DualRing │  │ Salience     │     │  │
                    │  │  │  (DLM)   │  │(RCP/IRCP)│  │ Scoring      │     │  │
                    │  │  └──────────┘  └──────────┘  └──────────────┘     │  │
                    │  └───────────────────────────────────────────────────┘  │
                    │  ┌───────────────────────────────────────────────────┐  │
                    │  │              Durability Layer                      │  │
                    │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │  │
                    │  │  │   WAL    │  │  Cache   │  │Conservation  │     │  │
                    │  │  └──────────┘  └──────────┘  └──────────────┘     │  │
                    │  └───────────────────────────────────────────────────┘  │
                    └─────────────────────────────────────────────────────────┘
```

## Core Concepts

### 5D Trajectory Coordinates

Each episode (memory turn) has a 5D position extending the TPO (Trajectory Position Operator) framework:

| Dimension | Type | Range | Meaning |
|-----------|------|-------|---------|
| `depth` | u32 | [0, max_depth] | Distance from conversation root (0 = root) |
| `sibling_order` | u32 | [0, n_siblings] | Position among siblings in the DAG |
| `homogeneity` | f32 | [0, 1] | Semantic similarity to parent turn |
| `temporal` | f32 | [0, 1] | Normalized timestamp within conversation |
| `complexity` | u32 | [1, ∞) | Message complexity (n_parts from DLM) |

```rust
use rag_plusplus_core::trajectory::{TrajectoryCoordinate5D, DLMWeights};

// Create a 5D coordinate
let coord = TrajectoryCoordinate5D {
    depth: 3,
    sibling_order: 0,
    homogeneity: 0.85,
    temporal: 0.45,
    complexity: 2,
};

// Compute weighted distance between coordinates
let weights = DLMWeights::default();
let distance = coord.weighted_distance(&other_coord, &weights);
```

### Dual-Ring Topology (IRCP/RCP)

The `DualRing` provides two simultaneous orderings over the same episodes:

| Ring | Direction | Ordering | Use Case |
|------|-----------|----------|----------|
| **Temporal** (RCP) | Forward | Time/causal sequence | "What context led here?" |
| **Influence** (IRCP) | Inverse | By influence weight | "What had most impact?" |

```
Temporal Ring (RCP):     [E₀] → [E₁] → [E₂] → [E₃] → [E₄] ─┐
                           ↑                                 │
                           └─────────────────────────────────┘

Influence Ring (IRCP):   [E₂] → [E₄] → [E₀] → [E₃] → [E₁] ─┐
                           ↑     (sorted by influence)      │
                           └────────────────────────────────┘
```

```rust
use rag_plusplus_core::trajectory::{
    DualRing, build_dual_ring_with_attention
};

// Build dual ring with attention weights
let turn_ids = vec![1, 2, 3, 4, 5];
let attention = vec![0.1, 0.3, 0.4, 0.1, 0.1];
let dual_ring = build_dual_ring_with_attention(&turn_ids, &attention);

// Navigate temporal order
let next_temporal = dual_ring.temporal_next(0);

// Navigate influence order
let next_influence = dual_ring.influence_next(0);

// Ring distance (shortest path on circular topology)
let distance = dual_ring.temporal_ring_distance(0, 3);
```

### Conservation Metrics

Per RCP (Ring Contextual Propagation), memory operations should preserve invariants:

| Law | Formula | Meaning |
|-----|---------|---------|
| **Magnitude** | Σ aᵢ‖eᵢ‖ = const | Context doesn't disappear |
| **Energy** | ½Σᵢⱼ aᵢaⱼ cos(eᵢ, eⱼ) | Attention capacity conserved |
| **Information** | -Σ aᵢ log(aᵢ) | Shannon entropy of attention |

```rust
use rag_plusplus_core::trajectory::{ConservationMetrics, ConservationConfig};

// Compute conservation metrics
let embeddings: Vec<&[f32]> = vec![&emb1, &emb2, &emb3];
let attention = vec![0.5, 0.3, 0.2];

let metrics_before = ConservationMetrics::compute(&embeddings, &attention);

// After some memory operation...
let metrics_after = ConservationMetrics::compute(&new_embeddings, &new_attention);

// Check if conservation is preserved
if metrics_before.is_conserved(&metrics_after, 0.01) {
    println!("Conservation laws satisfied");
} else {
    let violation = metrics_before.violation(&metrics_after);
    println!("Violation: magnitude_delta={:.4}", violation.magnitude_delta);
}
```

### Salience Scoring

Multi-factor importance weighting for bounded forgetting:

```rust
use rag_plusplus_core::trajectory::{
    SalienceScorer, SalienceFactors, Feedback
};

let scorer = SalienceScorer::new();

let factors = SalienceFactors {
    turn_id: 42,
    feedback: Some(Feedback::ThumbsUp),    // +0.35
    is_phase_transition: true,              // +0.10
    reference_count: 3,                     // +0.15 (capped)
    embedding: Some(vec![0.1; 768]),        // For novelty
    ..Default::default()
};

let salience = scorer.score_single(&factors, Some(&recent_embeddings));
// salience.score is bounded to [0, 1]
println!("Salience: {:.2} (feedback: {:.2}, phase: {:.2})",
    salience.score,
    salience.feedback_contribution,
    salience.phase_contribution
);
```

### Phase Inference

Automatically detect conversation phases from turn features:

```rust
use rag_plusplus_core::trajectory::{
    PhaseInferencer, TurnFeatures, TrajectoryPhase
};

let inferencer = PhaseInferencer::new();

let features = TurnFeatures::from_content(1, "user", "What is the bug?");
let (phase, confidence) = inferencer.infer_single(&features).unwrap();

match phase {
    TrajectoryPhase::Exploration => println!("Initial inquiry"),
    TrajectoryPhase::Consolidation => println!("Building understanding"),
    TrajectoryPhase::Synthesis => println!("Drawing conclusions"),
    TrajectoryPhase::Debugging => println!("Error resolution"),
    TrajectoryPhase::Planning => println!("Creating roadmap"),
}
```

## SIMD Distance Functions

AVX2-accelerated distance computations with automatic runtime detection:

```rust
use rag_plusplus_core::distance::{
    l2_distance_fast, cosine_similarity_fast, inner_product_fast
};

let a = vec![0.1; 1536];  // OpenAI embedding dimension
let b = vec![0.2; 1536];

// Automatically uses AVX2 on supported CPUs, falls back to scalar
let l2 = l2_distance_fast(&a, &b);
let cosine = cosine_similarity_fast(&a, &b);
let ip = inner_product_fast(&a, &b);
```

**Performance**: 4-8x speedup over scalar for vectors ≥256 dimensions.

## Module Reference

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `index` | Vector indices | `HNSWIndex`, `FlatIndex`, `VectorIndex` trait |
| `distance` | SIMD distances | `l2_distance_fast`, `cosine_similarity_fast` |
| `retrieval` | Query engine | `QueryEngine`, `SearchResult`, `Reranker` |
| `store` | Record storage | `RecordStore` trait, `InMemoryStore` |
| `trajectory` | 5D coordinates | `TrajectoryCoordinate5D`, `DualRing`, `SalienceScorer` |
| `trajectory::graph` | DAG traversal | `TrajectoryGraph`, `Edge`, `PathSelectionPolicy` |
| `trajectory::conservation` | Forgetting validation | `ConservationMetrics`, `ConservationTracker` |
| `trajectory::ircp` | Attention propagation | `IRCPPropagator`, `AttentionWeights` |
| `trajectory::chainlink` | Cross-conversation links | `ChainLinkEstimator`, `LinkType` |
| `quantization` | Vector compression | `ScalarQuantizer`, `ProductQuantizer` |
| `wal` | Write-ahead log | `WriteAheadLog`, `WALEntry` |
| `cache` | Query caching | `QueryCache`, `CacheConfig` |
| `filter` | Metadata filters | `FilterExpression`, `FilterOp` |
| `stats` | Running statistics | `WelfordStats`, `OnlineVariance` |
| `observability` | Metrics/tracing | Prometheus metrics, tracing spans |

## Configuration

### HNSW Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | 16 | Connections per layer (higher = better recall, more memory) |
| `m_max0` | 32 | Max connections at layer 0 |
| `ef_construction` | 200 | Build-time search depth (higher = better quality) |
| `ef_search` | 128 | Query-time search depth (higher = better recall, slower) |
| `ml` | 1/ln(m) | Level multiplier for probabilistic layer assignment |

### Salience Weights

| Factor | Default Weight | Description |
|--------|---------------|-------------|
| `base_score` | 0.5 | Neutral starting score |
| `feedback_up_boost` | +0.35 | ThumbsUp contribution |
| `feedback_down_penalty` | -0.35 | ThumbsDown penalty |
| `phase_transition_boost` | +0.10 | Phase boundary bonus |
| `reference_boost_per` | +0.05 | Per downstream reference |
| `reference_boost_cap` | +0.15 | Maximum reference contribution |
| `novelty_boost_max` | +0.10 | Maximum novelty contribution |

### Conservation Tolerances

| Metric | Default Tolerance | Strict Mode |
|--------|------------------|-------------|
| `magnitude_tolerance` | 0.01 | 0.001 |
| `energy_tolerance` | 0.01 | 0.001 |
| `information_tolerance` | 0.1 | 0.01 |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | Yes | Enable rayon-based parallel processing |
| `salience-debug` | No | Log salience distribution statistics for tuning |

Enable features in `Cargo.toml`:

```toml
[dependencies]
rag-plusplus-core = { version = "0.1", features = ["salience-debug"] }
```

## Performance Characteristics

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| HNSW Build | O(n log n) | ~1ms per vector |
| HNSW Search | O(log n) | <1ms for k=10 |
| Flat Search | O(n × d) | ~10ms per 10K vectors |
| SIMD L2 (1536d) | O(d) | ~500ns |
| Conservation Check | O(n²) | ~1ms for n=100 |
| Salience Score | O(1) | ~100ns |

## Benchmarks

Run comprehensive benchmarks:

```bash
# All benchmarks
cargo bench

# Specific benchmark
cargo bench --bench welford
cargo bench --bench serialization
```

Benchmark suites:
- **welford** - Running statistics performance
- **serialization** - rkyv zero-copy serialization throughput
- **distance** - SIMD vs scalar comparison
- **index** - HNSW build and search scaling

## Examples

### Trajectory-Aware Retrieval

```rust
use rag_plusplus_core::{
    index::{HNSWIndex, HNSWConfig, VectorIndex},
    trajectory::{
        TrajectoryCoordinate5D, DLMWeights,
        SalienceScorer, SalienceFactors,
    },
};

// Build index with trajectory awareness
let config = HNSWConfig::new(768);
let mut index = HNSWIndex::new(config);

// Add vectors with salience-weighted insertion
let scorer = SalienceScorer::new();
for turn in turns {
    let factors = SalienceFactors {
        turn_id: turn.id,
        feedback: turn.feedback,
        is_phase_transition: turn.is_boundary,
        reference_count: turn.ref_count,
        embedding: Some(turn.embedding.clone()),
        ..Default::default()
    };

    let salience = scorer.score_single(&factors, None);

    // Only index high-salience turns
    if salience.score > 0.4 {
        index.add(turn.id.to_string(), &turn.embedding)?;
    }
}
```

### Chain Management

```rust
use rag_plusplus_core::trajectory::chain::{
    ChainManager, ChainManagerConfig,
    detect_knowledge_transfer,
};

// Manage multiple conversation chains
let config = ChainManagerConfig::default();
let mut manager = ChainManager::new(config);

// Register chains
let chain_a = manager.create_chain("project-alpha")?;
let chain_b = manager.create_chain("project-beta")?;

// Detect knowledge transfer between chains
let links = detect_knowledge_transfer(
    &chain_a_embeddings,
    &chain_b_embeddings,
    0.8,  // similarity threshold
);

for link in links {
    println!("Knowledge transfer: {} -> {} (strength: {:.2})",
        link.source_turn, link.target_turn, link.strength);
}
```

## Integration with RAG++ Python

This crate provides the high-performance core for the `rag-plusplus` Python package via PyO3 bindings:

```python
import rag_plusplus

# Python API calls into Rust core
index = rag_plusplus.HNSWIndex(dimension=768)
index.add("doc-1", embedding)
results = index.search(query, k=10)
```

## Related Packages

| Package | Description |
|---------|-------------|
| [`rag-plusplus`](https://pypi.org/project/rag-plusplus/) | Python bindings and high-level API |
| [`admissibility-kernel`](https://crates.io/crates/admissibility-kernel) | Context slicing with cryptographic verification |
| [`cognitive-twin`](https://pypi.org/project/cognitive-twin/) | User pattern learning with trajectory-aware DPO |

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy -- -D warnings
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use rag-plusplus-core in academic work, please cite:

```bibtex
@software{rag_plusplus_core,
  author = {Diomande, Mohamed},
  title = {rag-plusplus-core: SIMD-Accelerated Vector Search with Trajectory Memory},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Diomandeee/rag-plusplus-core}
}
```
