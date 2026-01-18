//! Record Storage Module
//!
//! Provides persistent and in-memory storage for MemoryRecords.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      RecordStore Trait                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  + insert(record) -> Result<RecordId>                        │
//! │  + get(id) -> Option<&MemoryRecord>                          │
//! │  + update_stats(id, outcome) -> Result<()>                   │
//! │  + remove(id) -> Result<bool>                                │
//! │  + iter() -> impl Iterator<Item = &MemoryRecord>             │
//! └─────────────────────────────────────────────────────────────┘
//!            ▲                            ▲
//!            │                            │
//!     ┌──────┴──────┐            ┌────────┴────────┐
//!     │ InMemoryStore│            │ PersistentStore │
//!     │  (HashMap)   │            │  (mmap + rkyv)  │
//!     └─────────────┘            └─────────────────┘
//! ```
//!
//! # Invariants
//!
//! - INV-001: Record immutability (only stats updates allowed)
//! - INV-004: Index-record consistency

mod memory;
mod traits;

pub use memory::InMemoryStore;
pub use traits::{RecordStore, SharedStore};
