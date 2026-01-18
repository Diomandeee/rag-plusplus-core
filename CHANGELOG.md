# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-17

### Added

- Initial release extracted from Comp-Core monorepo
- **Vector Indices**
  - HNSW index for approximate nearest neighbor search
  - Flat index for exact brute-force search
  - Index registry for multi-index queries
  - Reciprocal rank fusion (RRF) for result merging
- **Distance Operations**
  - L2 (Euclidean) distance
  - Cosine similarity/distance
  - Inner product
  - SIMD-accelerated variants (AVX2)
  - Trajectory-weighted distance functions
- **Trajectory Memory**
  - 5D coordinate system (depth, sibling, homogeneity, temporal, complexity)
  - Salience scoring with configurable factors
  - Phase inference (exploration, consolidation, synthesis)
  - Ring structures (Episode, Dual)
  - Conservation metrics
  - Path selection policies
- **Storage & Durability**
  - RecordStore trait with InMemory implementation
  - Write-ahead log (WAL) with checkpointing
  - Write buffer for batched operations
- **Query Engine**
  - Configurable retrieval pipeline
  - Reranking support
  - Metadata filtering
- **Quantization**
  - Scalar quantization (SQ8)
  - Product quantization (PQ)
- **Observability**
  - Metrics integration
  - Tracing spans
- **Evaluation**
  - Benchmarking utilities
  - Query evaluation metrics
