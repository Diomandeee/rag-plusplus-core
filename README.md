# rag-plusplus-core

High-performance retrieval engine in Rust with SIMD-accelerated vector search and trajectory memory.

## Features

- **SIMD-accelerated vector distances** - AVX2 optimized L2, cosine, inner product
- **HNSW index** - Approximate nearest neighbor search with configurable parameters
- **Flat index** - Exact brute-force search for small datasets
- **Trajectory memory** - 5D coordinate system (depth, sibling, homogeneity, temporal, complexity)
- **Write-ahead log** - Durable operations with checkpoint/recovery
- **Query result caching** - LRU cache with configurable TTL
- **Product quantization** - Memory-efficient vector compression (SQ8, PQ)

## Quick Start

```rust
use rag_plusplus_core::{HNSWIndex, HNSWConfig, MemoryRecord, RecordId};

// Create an HNSW index for 768-dimensional vectors
let config = HNSWConfig::default();
let mut index = HNSWIndex::new(768, config);

// Add vectors
let record = MemoryRecord::new(
    RecordId::new("doc-1"),
    vec![0.1; 768],
    Default::default(),
);
index.add(&record)?;

// Search for nearest neighbors
let query = vec![0.15; 768];
let results = index.search(&query, 10)?;

for (id, distance) in results {
    println!("Found: {} at distance {}", id, distance);
}
```

## Modules

| Module | Description |
|--------|-------------|
| `index` | Vector indices (HNSW, Flat) with fusion strategies |
| `distance` | SIMD-optimized distance functions |
| `retrieval` | Query engine with reranking |
| `store` | Record storage trait and implementations |
| `trajectory` | 5D trajectory coordinates, salience scoring, ring structures |
| `quantization` | Vector compression (SQ8, PQ) |
| `wal` | Write-ahead log for durability |
| `cache` | Query result caching |
| `filter` | Metadata filter expressions |
| `observability` | Metrics and tracing |

## Feature Flags

- `parallel` (default) - Enable rayon-based parallel processing
- `salience-debug` - Enable debug logging for salience distribution

## Benchmarks

Run benchmarks with:

```bash
cargo bench
```

Available benchmarks:
- `distance` - Distance computation performance
- `index` - Index build/search performance
- `retrieval` - Query engine latency
- `serialization` - rkyv serialization throughput
- `wal` - Write-ahead log performance

## License

MIT
