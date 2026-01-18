//! Filter Engine
//!
//! Provides metadata filtering for retrieval results.
//!
//! # Architecture
//!
//! ```text
//! Filter Expression (AST)
//!        │
//!        ▼
//! ┌─────────────────┐
//! │ FilterCompiler  │ ──▶ CompiledFilter (optimized)
//! └─────────────────┘
//!        │
//!        ▼
//! ┌─────────────────┐
//! │ FilterEvaluator │ ──▶ bool (matches/not)
//! └─────────────────┘
//! ```
//!
//! # Supported Operations
//!
//! - Equality: `field == value`
//! - Comparison: `field > value`, `field < value`
//! - Range: `field in [min, max]`
//! - Set membership: `field in [a, b, c]`
//! - Boolean: `AND`, `OR`, `NOT`

mod ast;
mod eval;

pub use ast::{FilterExpr, FilterOp, FilterValue};
pub use eval::{CompiledFilter, FilterEvaluator};
