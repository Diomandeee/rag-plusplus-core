//! RAG++ Error Types
//!
//! Structured error hierarchy for the entire library.
//! All errors implement std::error::Error and can be converted to anyhow::Error.

use thiserror::Error;

/// Main error type for RAG++
#[derive(Error, Debug)]
pub enum Error {
    // Index errors
    #[error("Index not found: {name}")]
    IndexNotFound { name: String },

    #[error("Duplicate index: {name}")]
    DuplicateIndex { name: String },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index build failed: {reason}")]
    IndexBuild { reason: String },

    #[error("Index corrupted: {details}")]
    IndexCorrupted { details: String },

    #[error("Index capacity exceeded: {current}/{max}")]
    IndexCapacity { current: usize, max: usize },

    // Query errors
    #[error("Invalid query: {reason}")]
    InvalidQuery { reason: String },

    #[error("Query timeout after {elapsed_ms}ms (budget: {budget_ms}ms)")]
    QueryTimeout { elapsed_ms: u64, budget_ms: u64 },

    #[error("Filter evaluation failed: {reason}")]
    FilterError { reason: String },

    // Storage errors
    #[error("Record not found: {record_id}")]
    RecordNotFound { record_id: String },

    #[error("Duplicate record: {record_id}")]
    DuplicateRecord { record_id: String },

    #[error("Serialization failed: {reason}")]
    Serialization { reason: String },

    // WAL errors
    #[error("WAL write failed: {reason}")]
    WalWrite { reason: String },

    #[error("WAL recovery failed: {reason}")]
    WalRecovery { reason: String },

    // Resource errors
    #[error("Memory limit exceeded: {used_bytes}/{limit_bytes} bytes")]
    MemoryLimit { used_bytes: usize, limit_bytes: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias using RAG++ Error
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Check if error is retriable
    #[must_use]
    pub const fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::QueryTimeout { .. } | Self::Io(_) | Self::MemoryLimit { .. }
        )
    }

    /// Get error code for metrics/logging
    #[must_use]
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::IndexNotFound { .. } => "INDEX_NOT_FOUND",
            Self::DuplicateIndex { .. } => "DUPLICATE_INDEX",
            Self::DimensionMismatch { .. } => "DIMENSION_MISMATCH",
            Self::IndexBuild { .. } => "INDEX_BUILD_FAILED",
            Self::IndexCorrupted { .. } => "INDEX_CORRUPTED",
            Self::IndexCapacity { .. } => "INDEX_CAPACITY",
            Self::InvalidQuery { .. } => "INVALID_QUERY",
            Self::QueryTimeout { .. } => "QUERY_TIMEOUT",
            Self::FilterError { .. } => "FILTER_ERROR",
            Self::RecordNotFound { .. } => "RECORD_NOT_FOUND",
            Self::DuplicateRecord { .. } => "DUPLICATE_RECORD",
            Self::Serialization { .. } => "SERIALIZATION_ERROR",
            Self::WalWrite { .. } => "WAL_WRITE_ERROR",
            Self::WalRecovery { .. } => "WAL_RECOVERY_ERROR",
            Self::MemoryLimit { .. } => "MEMORY_LIMIT",
            Self::Io(_) => "IO_ERROR",
        }
    }
}
