//! Write Buffer Module
//!
//! Provides buffered write operations for improved performance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     Write Buffer                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Operations:                                                 │
//! │    1. Write to WAL (durability)                              │
//! │    2. Buffer in memory                                       │
//! │    3. Flush when: capacity/time/explicit                     │
//! └─────────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼ flush
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    RecordStore + Index                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Thread Safety
//!
//! The buffer is thread-safe and can be shared across threads.

mod write_buffer;

pub use write_buffer::{WriteBuffer, WriteBufferConfig, BufferStats};
