//! Write-Ahead Log (WAL)
//!
//! Provides durability guarantees for record operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   Write-Ahead Log (WAL)                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │  1. Log entry to WAL (fsync)                                 │
//! │  2. Apply to in-memory buffer                                │
//! │  3. Periodically checkpoint to persistent store              │
//! │  4. Truncate WAL after successful checkpoint                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Invariants
//!
//! - INV-003: WAL-before-buffer (operations logged before applied)
//!
//! # Entry Types
//!
//! - Insert: Add a new record
//! - UpdateStats: Update outcome statistics
//! - Delete: Mark record as deleted

mod entry;
mod reader;
mod writer;

pub use entry::{WalEntry, WalEntryType};
pub use reader::WalReader;
pub use writer::{WalConfig, WalWriter};
