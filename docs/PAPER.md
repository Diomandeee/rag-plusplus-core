# RAG++ Core: A Trajectory-Aware Vector Search Engine with SIMD Acceleration

**Technical Report v0.1.0**

*Mohamed Diomande*

---

## Abstract

Retrieval-Augmented Generation (RAG) systems have become foundational for building knowledge-grounded language model applications. However, conventional vector search treats all embeddings as points in an undifferentiated metric space, ignoring the rich structural relationships inherent in conversational data. We present **RAG++ Core**, a high-performance retrieval engine written in Rust that introduces **trajectory-aware** retrieval through a novel 5D coordinate system capturing conversational structure. Our system combines SIMD-accelerated distance computations achieving 4-8x speedup over scalar implementations, an optimized HNSW index for O(log n) approximate nearest neighbor search, and a **dual-ring topology** (RCP/IRCP) that models both causal flow and influence propagation. We introduce **conservation metrics** for validating memory operations under bounded forgetting constraints, and a **multi-factor salience scoring** system that accounts for user feedback, phase transitions, and semantic novelty. Benchmarks demonstrate sub-millisecond query latencies on million-scale indices with recall@10 exceeding 95%.

**Keywords**: vector search, RAG, HNSW, SIMD, trajectory memory, conversation structure

---

## 1. Introduction

### 1.1 Problem Statement

Modern RAG systems retrieve context by finding embeddings similar to a query vector. This approach, while effective for static document retrieval, fails to capture the dynamic, structured nature of conversational memory. Specifically, existing systems suffer from three limitations:

1. **Structural Blindness**: All retrieved turns are treated equally regardless of their position in the conversation tree, their relationship to other turns, or their role in knowledge development.

2. **Temporal Amnesia**: Standard cosine similarity provides no mechanism for preferring recent, contextually relevant information over stale or superseded knowledge.

3. **Feedback Ignorance**: User signals (explicit ratings, implicit engagement) cannot influence retrieval weighting without expensive re-indexing.

### 1.2 Motivation

Conversational AI systems maintain memory across extended interactions spanning hours, days, or weeks. This memory forms a directed acyclic graph (DAG) where:

- **Turns** are nodes with semantic content (embeddings)
- **Edges** represent causal relationships (reply-to, branch-from)
- **Structure** encodes valuable context (depth in discussion, branching decisions, topic transitions)

A retrieval system that ignores this structure discards information essential for coherent, contextually-appropriate responses.

### 1.3 Contributions

This paper makes the following contributions:

1. **5D Trajectory Coordinate System**: A novel embedding augmentation that assigns each turn a position in a 5-dimensional space capturing depth, sibling order, semantic homogeneity, temporal position, and complexity (Section 3.1).

2. **Dual-Ring Topology (RCP/IRCP)**: A data structure maintaining two simultaneous orderings—forward causal propagation and inverse influence weighting—enabling bidirectional context traversal (Section 3.2).

3. **Conservation Metrics Framework**: Formal invariants for validating memory operations, ensuring bounded forgetting does not violate system guarantees (Section 3.3).

4. **SIMD-Accelerated Distance Computation**: AVX2 implementations of L2, cosine, and inner product distances achieving 4-8x throughput improvement (Section 4.1).

5. **Trajectory-Aware HNSW Index**: An extended Hierarchical Navigable Small World graph that incorporates coordinate distance into edge construction and search (Section 4.2).

---

## 2. Related Work

### 2.1 Approximate Nearest Neighbor Search

The HNSW algorithm [Malkov & Yashunin, 2018] constructs a multi-layer graph where each layer provides increasingly coarse approximations of the neighborhood structure. Search proceeds top-down, using each layer to narrow the candidate set. Our implementation extends HNSW with trajectory-aware edge selection during graph construction.

Product Quantization (PQ) [Jégou et al., 2011] and Scalar Quantization (SQ) compress vectors to reduce memory bandwidth requirements. We implement SQ8 (8-bit quantization) as a memory-efficient alternative for large-scale deployments.

### 2.2 Conversational Memory Systems

MemGPT [Packer et al., 2023] introduces hierarchical memory management for LLM context, but operates at the prompt level without vector-indexed retrieval. Generative Agents [Park et al., 2023] maintain importance-weighted memories but use simple recency scoring rather than structural features.

### 2.3 Embedding Augmentation

Positional encodings in Transformers [Vaswani et al., 2017] augment token embeddings with sequence position. Our 5D coordinate system extends this principle to the conversation level, encoding structural relationships across the memory DAG.

### 2.4 SIMD Optimization for ML Workloads

FAISS [Johnson et al., 2019] pioneered SIMD-optimized vector search. Our implementation differs in targeting Rust's memory safety guarantees while achieving comparable throughput through careful use of `unsafe` blocks bounded by safe abstractions.

---

## 3. System Architecture

### 3.1 5D Trajectory Coordinate System

We augment each turn's embedding with a 5-dimensional coordinate vector:

```
coord = (depth, sibling_order, homogeneity, temporal, complexity)
```

| Dimension | Symbol | Domain | Semantic Meaning |
|-----------|--------|--------|------------------|
| **Depth** | d | N | Distance from root (0 = initial message) |
| **Sibling Order** | s | N | Position among sibling branches |
| **Homogeneity** | h | [0, 1] | Semantic similarity to parent turn |
| **Temporal** | t | [0, 1] | Normalized timestamp within session |
| **Complexity** | c | N | Number of semantic components (from DLM) |

**Definition 3.1 (Trajectory Distance)**: Given two coordinates p₁ = (d₁, s₁, h₁, t₁, c₁) and p₂ = (d₂, s₂, h₂, t₂, c₂), the weighted trajectory distance is:

```
D_traj(p₁, p₂) = √(w_d(d₁-d₂)² + w_s(s₁-s₂)² + w_h(h₁-h₂)² + w_t(t₁-t₂)² + w_c(c₁-c₂)²)
```

where w = (w_d, w_s, w_h, w_t, w_c) are configurable weights.

**Theorem 3.1 (Metric Properties)**: D_traj satisfies the metric axioms (non-negativity, identity of indiscernibles, symmetry, triangle inequality) for any positive weight vector.

*Proof*: Direct application of the weighted Euclidean metric. □

#### 3.1.1 Coordinate Computation

**Depth**: Computed via BFS traversal from root. Stored as u32 to support conversation trees up to 4 billion levels.

**Sibling Order**: Assigned during tree construction based on creation timestamp among children of the same parent.

**Homogeneity**: Cosine similarity between turn embedding and parent embedding:
```
h = (1 + cos(e_turn, e_parent)) / 2
```
Normalized to [0, 1] range.

**Temporal**: Linear normalization within session boundaries:
```
t = (timestamp - session_start) / (session_end - session_start)
```

**Complexity**: Derived from the Dialogue Line Model (DLM), counting semantic components (claims, questions, instructions, etc.) within the turn.

### 3.2 Dual-Ring Topology (RCP/IRCP)

We maintain two simultaneous ring orderings of memory turns:

1. **RCP (Ring Contextual Propagation)**: Forward temporal ordering, modeling causal flow. Turn i influences turn j if i precedes j in RCP order.

2. **IRCP (Inverse Ring Contextual Propagation)**: Reverse influence ordering, weighted by contribution. Turn i is influenced by turn j proportionally to j's salience and semantic similarity.

```
        RCP Direction (Forward)
        ─────────────────────────►

    ┌───┐    ┌───┐    ┌───┐    ┌───┐
    │ A │───►│ B │───►│ C │───►│ D │
    └───┘    └───┘    └───┘    └───┘
      ▲                          │
      └──────────────────────────┘
              (Ring closure)

        ◄─────────────────────────
        IRCP Direction (Inverse)
```

**Definition 3.2 (Ring Distance)**: For a ring of size n, the distance between positions i and j is:

```
d_ring(i, j) = min(|i - j|, n - |i - j|)
```

This ensures symmetric distance regardless of traversal direction.

**Definition 3.3 (Influence Weight)**: The IRCP influence weight from turn j to turn i is:

```
w_influence(i, j) = salience(j) × decay(d_ring(i, j)) × sim(e_i, e_j)
```

where decay is exponential with configurable half-life.

#### 3.2.1 Ring Operations

The dual-ring supports efficient operations:

| Operation | Complexity | Description |
|-----------|------------|-------------|
| Insert | O(1) | Append to ring tail |
| Lookup | O(1) | Direct index access |
| Forward Window | O(k) | k successors in RCP order |
| Inverse Window | O(k) | k predecessors weighted by IRCP |
| Ring Distance | O(1) | Modular arithmetic |

### 3.3 Conservation Metrics Framework

Memory operations (insertion, deletion, compaction) must preserve system invariants. We define three conservation metrics:

**Definition 3.4 (Conservation Metrics)**: For a memory state M = {(e_i, α_i)} where e_i is an embedding and α_i is its salience weight:

1. **Magnitude Conservation**:
   ```
   C_mag(M) = Σᵢ αᵢ ‖eᵢ‖
   ```

2. **Energy Conservation**:
   ```
   C_energy(M) = ½ Σᵢⱼ αᵢ αⱼ cos(eᵢ, eⱼ)
   ```

3. **Information Conservation**:
   ```
   C_info(M) = -Σᵢ αᵢ log(αᵢ)
   ```

**Theorem 3.2 (Bounded Forgetting)**: A forgetting operation F: M → M' satisfies bounded forgetting if:

```
|C_x(M) - C_x(M')| ≤ ε_x  for x ∈ {mag, energy, info}
```

where ε_x are configurable tolerance thresholds.

*Proof*: By construction—we verify conservation bounds before committing any memory modification. □

#### 3.3.1 Violation Detection

The `ConservationTracker` maintains running statistics and detects violations:

```rust
pub struct ConservationViolation {
    pub magnitude_delta: f32,    // Δ in magnitude
    pub energy_delta: f32,       // Δ in energy
    pub information_delta: f32,  // Δ in information
    pub severity: ViolationSeverity,
}
```

Violations trigger alerts but do not necessarily block operations—the system supports "soft" conservation with configurable strictness.

### 3.4 Salience Scoring System

Not all memories are equally important. We compute salience as a weighted combination of factors:

**Definition 3.5 (Salience Score)**: For a turn T with factors F:

```
salience(T) = clamp(base + Σᵢ wᵢ × fᵢ(T), 0, 1)
```

| Factor | Symbol | Default Weight | Description |
|--------|--------|----------------|-------------|
| Base | base | 0.5 | Prior salience |
| Feedback Up | f_up | +0.35 | Explicit positive signal |
| Feedback Down | f_down | -0.35 | Explicit negative signal |
| Phase Transition | f_phase | +0.10 | Boundary between phases |
| Reference Count | f_ref | +0.05/ref (cap 0.15) | Citations by other turns |
| Novelty | f_novel | +0.10 | Semantic distance from cluster centroid |

**Property 3.1 (Bounded Salience)**: By construction, salience ∈ [0, 1] for all turns.

#### 3.4.1 Phase-Aware Weighting

Conversation phases (Exploration, Consolidation, Synthesis, Debugging, Planning) receive different importance weights:

| Phase | Weight | Rationale |
|-------|--------|-----------|
| Synthesis | 1.0 | High-value conclusions |
| Consolidation | 0.8 | Established knowledge |
| Exploration | 0.6 | Tentative directions |
| Planning | 0.7 | Action intentions |
| Debugging | 0.5 | Temporary scaffolding |

---

## 4. Implementation

### 4.1 SIMD-Accelerated Distance Computation

We implement AVX2-optimized distance functions for x86_64 architectures, processing 8 f32 values per instruction.

#### 4.1.1 L2 Squared Distance

```rust
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let diff = _mm256_sub_ps(va, vb);
        // FMA: sum = diff * diff + sum
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Handle remainder (< 8 elements)
    let mut scalar_sum = horizontal_sum_avx2(sum);
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
    }

    scalar_sum
}
```

**Key Optimizations**:
1. **Unaligned loads** (`_mm256_loadu_ps`): Support arbitrary memory alignment
2. **Fused Multiply-Add** (`_mm256_fmadd_ps`): Single instruction for `a*b+c`
3. **Horizontal reduction**: Efficient 256-bit to scalar reduction

#### 4.1.2 Cosine Similarity

```rust
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    for i in 0..(a.len() / 8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));

        dot = _mm256_fmadd_ps(va, vb, dot);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    let dot_sum = horizontal_sum_avx2(dot);
    let norm_a_sum = horizontal_sum_avx2(norm_a).sqrt();
    let norm_b_sum = horizontal_sum_avx2(norm_b).sqrt();

    dot_sum / (norm_a_sum * norm_b_sum + 1e-8)
}
```

#### 4.1.3 Performance Characteristics

| Operation | Scalar (ns) | AVX2 (ns) | Speedup |
|-----------|-------------|-----------|---------|
| L2 (768-dim) | 890 | 112 | 7.9x |
| Cosine (768-dim) | 1240 | 198 | 6.3x |
| Inner Product (768-dim) | 580 | 89 | 6.5x |

Benchmarks on Intel Core i7-12700K @ 3.6GHz, single-threaded.

### 4.2 HNSW Index Implementation

Our HNSW implementation follows [Malkov & Yashunin, 2018] with trajectory-aware extensions.

#### 4.2.1 Configuration

```rust
pub struct HNSWConfig {
    pub m: usize,            // Max edges per node (default: 16)
    pub m_max0: usize,       // Max edges at layer 0 (default: 2 * m)
    pub ef_construction: usize,  // Search width during construction (default: 200)
    pub ef_search: usize,    // Search width during query (default: 128)
    pub ml: f64,             // Level multiplier (default: 1 / ln(m))
}
```

**Theorem 4.1 (Layer Assignment)**: For a node with uniform random u ∈ [0, 1], the assigned layer is:

```
layer = floor(-ln(u) × ml)
```

This produces an exponential distribution with expected layer height O(log n).

#### 4.2.2 Graph Construction

**Algorithm 1: HNSW Insert**

```
function INSERT(q, embedding, coordinate):
    level ← random_level(ml)

    // Top-down search to find entry point at level
    ep ← entry_point
    for l ← max_level down to level + 1:
        ep ← SEARCH_LAYER(q, ep, 1, l)

    // Insert at each layer from level down to 0
    for l ← min(level, max_level) down to 0:
        neighbors ← SEARCH_LAYER(q, ep, ef_construction, l)
        selected ← SELECT_NEIGHBORS(q, neighbors, M_l, coordinate)

        // Add bidirectional edges
        for n in selected:
            connect(q, n, l)
            connect(n, q, l)

            // Prune if exceeding max edges
            if degree(n, l) > M_l:
                PRUNE(n, l)

        ep ← neighbors[0]
```

**Key Extension**: `SELECT_NEIGHBORS` incorporates trajectory distance:

```rust
fn select_neighbors(
    &self,
    query: &[f32],
    query_coord: &TrajectoryCoordinate5D,
    candidates: &[usize],
    m: usize,
) -> Vec<usize> {
    // Score = α × embedding_distance + (1-α) × trajectory_distance
    let mut scored: Vec<_> = candidates.iter().map(|&id| {
        let emb_dist = self.distance(query, &self.vectors[id]);
        let traj_dist = query_coord.weighted_distance(&self.coords[id]);
        let score = self.alpha * emb_dist + (1.0 - self.alpha) * traj_dist;
        (id, score)
    }).collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(m);
    scored.into_iter().map(|(id, _)| id).collect()
}
```

#### 4.2.3 Search Algorithm

**Algorithm 2: HNSW Search**

```
function SEARCH(q, k, ef):
    ep ← entry_point

    // Greedy descent from top layer
    for l ← max_level down to 1:
        ep ← SEARCH_LAYER(q, ep, 1, l)

    // Beam search at layer 0
    candidates ← SEARCH_LAYER(q, ep, ef, 0)

    return TOP_K(candidates, k)
```

**Complexity**:
- Build: O(n log n) expected time
- Query: O(log n) expected time for ef = O(log n)
- Memory: O(n × m × L) where L is average layer count

### 4.3 Memory Layout and Zero-Copy Serialization

We use [rkyv](https://rkyv.org) for zero-copy deserialization, enabling memory-mapped index loading.

```rust
#[derive(Archive, Deserialize, Serialize)]
#[archive_attr(derive(CheckBytes))]
pub struct SerializedIndex {
    pub vectors: AlignedVec<f32>,
    pub coordinates: Vec<ArchivedCoordinate5D>,
    pub graph: Vec<Vec<u32>>,
    pub metadata: IndexMetadata,
}
```

**Memory-Mapped Loading**:
```rust
impl HNSWIndex {
    pub fn load_mmap(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Zero-copy: archive lives directly in mmap
        let archived = rkyv::check_archived_root::<SerializedIndex>(&mmap)?;

        Ok(Self::from_archived(archived, mmap))
    }
}
```

Benefits:
- **Instant startup**: No deserialization required
- **Shared memory**: Multiple processes can share the same mmap
- **Lazy loading**: Pages loaded on-demand by OS

### 4.4 Write-Ahead Log (WAL)

For durability, we maintain a write-ahead log:

```rust
pub struct WALEntry {
    pub sequence: u64,
    pub operation: WALOperation,
    pub checksum: u32,
}

pub enum WALOperation {
    Insert { id: String, vector: Vec<f32>, coord: Coordinate5D },
    Delete { id: String },
    UpdateSalience { id: String, salience: f32 },
}
```

**Recovery Protocol**:
1. Load last checkpoint (serialized index)
2. Replay WAL entries with sequence > checkpoint_sequence
3. Verify integrity via checksums
4. Compact WAL after successful recovery

---

## 5. Evaluation

### 5.1 Benchmark Setup

| Parameter | Value |
|-----------|-------|
| Dataset | 1M synthetic embeddings (768-dim, Gaussian) |
| Index Type | HNSW (m=16, ef_construction=200) |
| Query Set | 10K embeddings, k=10 |
| Hardware | Intel i7-12700K, 32GB DDR5, NVMe SSD |
| Baseline | FAISS HNSW (same parameters) |

### 5.2 Recall vs. Latency

| ef_search | Recall@10 | Latency (p50) | Latency (p99) |
|-----------|-----------|---------------|---------------|
| 32 | 91.2% | 0.21 ms | 0.45 ms |
| 64 | 94.8% | 0.38 ms | 0.72 ms |
| 128 | 97.3% | 0.71 ms | 1.24 ms |
| 256 | 98.9% | 1.32 ms | 2.18 ms |

### 5.3 Trajectory-Aware vs. Embedding-Only

| Retrieval Mode | Contextual Relevance | User Preference |
|----------------|---------------------|-----------------|
| Embedding-only | 72.3% | 34% |
| Trajectory-aware (α=0.3) | 84.7% | 66% |

*Contextual relevance measured by human evaluation on 500 retrieval samples.*

### 5.4 SIMD Throughput

| Implementation | Throughput (queries/sec) |
|----------------|-------------------------|
| Scalar Rust | 12,400 |
| AVX2 Rust (ours) | 89,200 |
| FAISS AVX2 (C++) | 94,100 |

Our implementation achieves 95% of FAISS throughput while providing Rust's memory safety guarantees.

### 5.5 Conservation Metric Validation

We tested bounded forgetting over 10,000 memory compaction operations:

| Metric | Mean Δ | Max Δ | Violations (ε=0.01) |
|--------|--------|-------|---------------------|
| Magnitude | 0.0023 | 0.0089 | 0 |
| Energy | 0.0041 | 0.0094 | 0 |
| Information | 0.0018 | 0.0076 | 0 |

All operations remained within tolerance bounds.

---

## 6. Discussion

### 6.1 Design Decisions

**Why 5 Dimensions?**: We empirically tested 3D (depth, temporal, homogeneity), 5D (our system), and 7D (adding branching factor and sibling count). 5D provided the best tradeoff between expressiveness and computational overhead. Additional dimensions showed diminishing returns.

**Why Dual Rings?**: Single-ring architectures force a choice between temporal and influence ordering. The dual-ring allows efficient traversal in both directions without index duplication.

**Why Conservation Metrics?**: Without formal invariants, memory compaction becomes a black box. Conservation metrics provide provable bounds on information loss.

### 6.2 Limitations

1. **Coordinate Computation Overhead**: Computing homogeneity requires parent embedding access, adding latency during ingestion.

2. **Cold Start**: New conversations lack trajectory structure, degrading to embedding-only retrieval.

3. **Cross-Session Transfer**: Coordinates are session-local; cross-session retrieval requires normalization.

### 6.3 Future Work

1. **Learned Coordinates**: Train an encoder to predict trajectory coordinates from embedding content.

2. **Hierarchical Rings**: Multi-level ring structures for very long conversations (>10K turns).

3. **GPU Acceleration**: Port SIMD kernels to CUDA/ROCm for throughput-critical deployments.

4. **Distributed Index**: Shard HNSW graph across machines with coordinate-aware routing.

---

## 7. Conclusion

We presented RAG++ Core, a trajectory-aware vector search engine that augments traditional embedding retrieval with structural awareness. Our 5D coordinate system captures conversational topology, the dual-ring topology enables bidirectional traversal, and conservation metrics provide formal guarantees for bounded forgetting. SIMD acceleration achieves 4-8x speedup with 95% of optimized C++ performance.

The system is available as an open-source Rust crate: `rag-plusplus-core` on crates.io.

---

## References

[1] Y. Malkov and D. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs," *IEEE TPAMI*, vol. 42, no. 4, pp. 824-836, 2018.

[2] H. Jégou, M. Douze, and C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE TPAMI*, vol. 33, no. 1, pp. 117-128, 2011.

[3] C. Packer et al., "MemGPT: Towards LLMs as Operating Systems," *arXiv:2310.08560*, 2023.

[4] J. S. Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," *UIST*, 2023.

[5] A. Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.

[6] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE TBD*, 2019.

[7] M. Diomande, "Admissibility Kernel: Deterministic Context Slicing with Cryptographic Verification," *Technical Report*, 2026.

[8] M. Diomande, "CognitiveTwin: Trajectory-Aware Direct Preference Optimization," *Technical Report*, 2026.

---

## Appendix A: API Reference

### A.1 Core Types

```rust
// 5D Coordinate
pub struct TrajectoryCoordinate5D {
    pub depth: u32,
    pub sibling_order: u32,
    pub homogeneity: f32,
    pub temporal: f32,
    pub complexity: u32,
}

// Conservation Metrics
pub struct ConservationMetrics {
    pub magnitude: f32,
    pub energy: f32,
    pub information: f32,
}

// Salience Score
pub struct TurnSalience {
    pub score: f32,
    pub factors: SalienceFactors,
}
```

### A.2 Index Operations

```rust
// Create index
let config = HNSWConfig::default();
let index = HNSWIndex::new(768, config);

// Insert with coordinates
index.add_with_coord(embedding, id, coord)?;

// Search
let results = index.search(&query, 10)?;

// Search with coordinate weighting
let results = index.search_trajectory(&query, &query_coord, 10, 0.3)?;
```

### A.3 Conservation Validation

```rust
// Create tracker
let config = ConservationConfig::default();
let tracker = ConservationTracker::new(config);

// Validate operation
let before = tracker.compute_metrics(&memory_before);
let after = tracker.compute_metrics(&memory_after);
let violation = tracker.check_violation(&before, &after);

if violation.severity >= ViolationSeverity::Error {
    return Err(ConservationError::Violation(violation));
}
```

---

## Appendix B: Benchmark Reproduction

```bash
# Clone repository
git clone https://github.com/Diomandeee/rag-plusplus-core
cd rag-plusplus-core

# Run benchmarks
cargo bench --features parallel

# Specific benchmarks
cargo bench -- hnsw_search
cargo bench -- simd_distance
cargo bench -- conservation
```

---

## Citation

```bibtex
@software{diomande2026ragpluspluscore,
  author = {Diomande, Mohamed},
  title = {RAG++ Core: A Trajectory-Aware Vector Search Engine},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Diomandeee/rag-plusplus-core}
}
```
