//! Sparse Retrieval Module
//!
//! Provides sparse (keyword-based) retrieval using BM25 algorithm.
//! Used in hybrid search to combine with dense vector retrieval.
//!
//! # Algorithm
//!
//! BM25 (Best Matching 25) is a probabilistic relevance function:
//!
//! score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
//!
//! Where:
//! - f(qi, D) = frequency of term qi in document D
//! - |D| = document length
//! - avgdl = average document length
//! - k1, b = tuning parameters (typically k1=1.2-2.0, b=0.75)
//! - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │              BM25Index                      │
//! ├─────────────────────────────────────────────┤
//! │  + add(id, text)                            │
//! │  + search(query, k) -> Vec<SparseResult>    │
//! │  + update_doc_stats()                       │
//! └─────────────────────────────────────────────┘
//! ```

mod bm25;
mod hybrid;
mod tokenizer;

pub use bm25::{BM25Config, BM25Index, SparseResult};
pub use hybrid::{HybridFusionStrategy, HybridResult, HybridSearchConfig, HybridSearcher};
pub use tokenizer::{SimpleTokenizer, Tokenizer};
