//! Write Buffer Implementation
//!
//! Buffers write operations for batched flushing.

use crate::error::Result;
use crate::index::VectorIndex;
use crate::store::RecordStore;
use crate::types::{MemoryRecord, RecordId};
use crate::wal::WalWriter;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Buffered operation type.
#[derive(Debug, Clone)]
pub enum BufferedOp {
    /// Insert a new record
    Insert(MemoryRecord),
    /// Update statistics
    UpdateStats { id: RecordId, outcome: f64 },
    /// Delete a record
    Delete(RecordId),
}

/// Write buffer configuration.
#[derive(Debug, Clone)]
pub struct WriteBufferConfig {
    /// Maximum number of operations before auto-flush
    pub max_ops: usize,
    /// Maximum buffer size in bytes before auto-flush
    pub max_bytes: usize,
    /// Maximum time in milliseconds before auto-flush (0 = disabled)
    pub max_age_ms: u64,
    /// Whether to use WAL
    pub use_wal: bool,
}

impl Default for WriteBufferConfig {
    fn default() -> Self {
        Self {
            max_ops: 1000,
            max_bytes: 64 * 1024 * 1024, // 64 MB
            max_age_ms: 5000,             // 5 seconds
            use_wal: true,
        }
    }
}

impl WriteBufferConfig {
    /// Create new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum operations.
    #[must_use]
    pub const fn with_max_ops(mut self, max: usize) -> Self {
        self.max_ops = max;
        self
    }

    /// Set maximum bytes.
    #[must_use]
    pub const fn with_max_bytes(mut self, max: usize) -> Self {
        self.max_bytes = max;
        self
    }

    /// Set maximum age in milliseconds.
    #[must_use]
    pub const fn with_max_age_ms(mut self, max: u64) -> Self {
        self.max_age_ms = max;
        self
    }

    /// Disable WAL (for testing only).
    #[must_use]
    pub const fn without_wal(mut self) -> Self {
        self.use_wal = false;
        self
    }
}

/// Buffer statistics.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Number of operations currently buffered
    pub buffered_ops: usize,
    /// Estimated buffer size in bytes
    pub buffered_bytes: usize,
    /// Total inserts (including flushed)
    pub total_inserts: u64,
    /// Total updates (including flushed)
    pub total_updates: u64,
    /// Total deletes (including flushed)
    pub total_deletes: u64,
    /// Number of flushes performed
    pub flush_count: u64,
}

/// Write buffer for batched operations.
///
/// Provides durability through WAL and performance through batching.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::buffer::{WriteBuffer, WriteBufferConfig};
///
/// let wal = WalWriter::new(wal_config)?;
/// let mut store = InMemoryStore::new();
/// let mut index = FlatIndex::new(IndexConfig::new(128));
///
/// let mut buffer = WriteBuffer::new(
///     WriteBufferConfig::new(),
///     Arc::new(wal),
/// );
///
/// buffer.insert(record)?;
/// buffer.update_stats(&id, 0.9)?;
///
/// // Flush to store and index
/// buffer.flush(&mut store, &mut index)?;
/// ```
pub struct WriteBuffer {
    config: WriteBufferConfig,
    /// WAL writer (optional)
    wal: Option<Arc<WalWriter>>,
    /// Buffered operations
    ops: RwLock<VecDeque<BufferedOp>>,
    /// Estimated buffer size
    size_bytes: AtomicUsize,
    /// Buffer creation/last flush time
    last_flush: RwLock<Instant>,
    /// Statistics
    total_inserts: AtomicU64,
    total_updates: AtomicU64,
    total_deletes: AtomicU64,
    flush_count: AtomicU64,
}

impl std::fmt::Debug for WriteBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriteBuffer")
            .field("config", &self.config)
            .field("ops_count", &self.ops.read().len())
            .field("size_bytes", &self.size_bytes.load(Ordering::Relaxed))
            .finish()
    }
}

impl WriteBuffer {
    /// Create a new write buffer with WAL.
    #[must_use]
    pub fn new(config: WriteBufferConfig, wal: Arc<WalWriter>) -> Self {
        Self {
            config,
            wal: Some(wal),
            ops: RwLock::new(VecDeque::new()),
            size_bytes: AtomicUsize::new(0),
            last_flush: RwLock::new(Instant::now()),
            total_inserts: AtomicU64::new(0),
            total_updates: AtomicU64::new(0),
            total_deletes: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
        }
    }

    /// Create a write buffer without WAL (testing/in-memory only).
    #[must_use]
    pub fn without_wal(config: WriteBufferConfig) -> Self {
        Self {
            config,
            wal: None,
            ops: RwLock::new(VecDeque::new()),
            size_bytes: AtomicUsize::new(0),
            last_flush: RwLock::new(Instant::now()),
            total_inserts: AtomicU64::new(0),
            total_updates: AtomicU64::new(0),
            total_deletes: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
        }
    }

    /// Estimate size of an operation.
    fn estimate_op_size(op: &BufferedOp) -> usize {
        match op {
            BufferedOp::Insert(record) => {
                std::mem::size_of::<MemoryRecord>()
                    + record.embedding.len() * 4
                    + record.context.len()
                    + record.id.len()
            }
            BufferedOp::UpdateStats { .. } => 32,
            BufferedOp::Delete(_) => 32,
        }
    }

    /// Check if buffer should auto-flush.
    fn should_flush(&self) -> bool {
        let ops = self.ops.read();
        let size = self.size_bytes.load(Ordering::Relaxed);
        let last_flush = self.last_flush.read();

        // Check capacity
        if ops.len() >= self.config.max_ops {
            return true;
        }

        // Check size
        if size >= self.config.max_bytes {
            return true;
        }

        // Check age
        if self.config.max_age_ms > 0 {
            let age = last_flush.elapsed().as_millis() as u64;
            if age >= self.config.max_age_ms && !ops.is_empty() {
                return true;
            }
        }

        false
    }

    /// Insert a record.
    ///
    /// The record is logged to WAL (if enabled) and buffered.
    pub fn insert(&self, record: MemoryRecord) -> Result<()> {
        // Write to WAL first (INV-003)
        if let Some(wal) = &self.wal {
            wal.log_insert(&record)?;
        }

        // Buffer the operation
        let op = BufferedOp::Insert(record);
        let size = Self::estimate_op_size(&op);

        {
            let mut ops = self.ops.write();
            ops.push_back(op);
        }

        self.size_bytes.fetch_add(size, Ordering::Relaxed);
        self.total_inserts.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Update record statistics.
    pub fn update_stats(&self, id: &RecordId, outcome: f64) -> Result<()> {
        // Write to WAL first
        if let Some(wal) = &self.wal {
            wal.log_update_stats(id, outcome)?;
        }

        // Buffer the operation
        let op = BufferedOp::UpdateStats {
            id: id.clone(),
            outcome,
        };
        let size = Self::estimate_op_size(&op);

        {
            let mut ops = self.ops.write();
            ops.push_back(op);
        }

        self.size_bytes.fetch_add(size, Ordering::Relaxed);
        self.total_updates.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Delete a record.
    pub fn delete(&self, id: &RecordId) -> Result<()> {
        // Write to WAL first
        if let Some(wal) = &self.wal {
            wal.log_delete(id)?;
        }

        // Buffer the operation
        let op = BufferedOp::Delete(id.clone());
        let size = Self::estimate_op_size(&op);

        {
            let mut ops = self.ops.write();
            ops.push_back(op);
        }

        self.size_bytes.fetch_add(size, Ordering::Relaxed);
        self.total_deletes.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Flush buffer to store and index.
    ///
    /// All buffered operations are applied atomically.
    pub fn flush<S: RecordStore, I: VectorIndex>(
        &self,
        store: &mut S,
        index: &mut I,
    ) -> Result<FlushResult> {
        let ops: Vec<BufferedOp> = {
            let mut ops_guard = self.ops.write();
            std::mem::take(&mut *ops_guard).into()
        };

        if ops.is_empty() {
            return Ok(FlushResult::default());
        }

        let mut result = FlushResult::default();

        for op in ops {
            match op {
                BufferedOp::Insert(record) => {
                    // Add to index
                    index.add(record.id.to_string(), &record.embedding)?;

                    // Add to store
                    store.insert(record)?;

                    result.inserts += 1;
                }
                BufferedOp::UpdateStats { id, outcome } => {
                    store.update_stats(&id, outcome)?;
                    result.updates += 1;
                }
                BufferedOp::Delete(id) => {
                    // Remove from index
                    index.remove(id.as_str())?;

                    // Remove from store
                    store.remove(&id)?;

                    result.deletes += 1;
                }
            }
        }

        // Reset buffer state
        self.size_bytes.store(0, Ordering::SeqCst);
        *self.last_flush.write() = Instant::now();
        self.flush_count.fetch_add(1, Ordering::Relaxed);

        // Checkpoint WAL
        if let Some(wal) = &self.wal {
            wal.log_checkpoint()?;
        }

        Ok(result)
    }

    /// Flush only to store (no index update).
    pub fn flush_to_store<S: RecordStore>(&self, store: &mut S) -> Result<FlushResult> {
        let ops: Vec<BufferedOp> = {
            let mut ops_guard = self.ops.write();
            std::mem::take(&mut *ops_guard).into()
        };

        if ops.is_empty() {
            return Ok(FlushResult::default());
        }

        let mut result = FlushResult::default();

        for op in ops {
            match op {
                BufferedOp::Insert(record) => {
                    store.insert(record)?;
                    result.inserts += 1;
                }
                BufferedOp::UpdateStats { id, outcome } => {
                    store.update_stats(&id, outcome)?;
                    result.updates += 1;
                }
                BufferedOp::Delete(id) => {
                    store.remove(&id)?;
                    result.deletes += 1;
                }
            }
        }

        self.size_bytes.store(0, Ordering::SeqCst);
        *self.last_flush.write() = Instant::now();
        self.flush_count.fetch_add(1, Ordering::Relaxed);

        Ok(result)
    }

    /// Auto-flush if buffer thresholds are exceeded.
    ///
    /// Returns `true` if flush occurred.
    pub fn maybe_flush<S: RecordStore, I: VectorIndex>(
        &self,
        store: &mut S,
        index: &mut I,
    ) -> Result<bool> {
        if self.should_flush() {
            self.flush(store, index)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get buffer statistics.
    #[must_use]
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            buffered_ops: self.ops.read().len(),
            buffered_bytes: self.size_bytes.load(Ordering::Relaxed),
            total_inserts: self.total_inserts.load(Ordering::Relaxed),
            total_updates: self.total_updates.load(Ordering::Relaxed),
            total_deletes: self.total_deletes.load(Ordering::Relaxed),
            flush_count: self.flush_count.load(Ordering::Relaxed),
        }
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.read().is_empty()
    }

    /// Get number of buffered operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.read().len()
    }
}

/// Result of a flush operation.
#[derive(Debug, Clone, Default)]
pub struct FlushResult {
    /// Number of inserts applied
    pub inserts: usize,
    /// Number of updates applied
    pub updates: usize,
    /// Number of deletes applied
    pub deletes: usize,
}

impl FlushResult {
    /// Total operations applied.
    #[must_use]
    pub fn total(&self) -> usize {
        self.inserts + self.updates + self.deletes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{FlatIndex, IndexConfig};
    use crate::stats::OutcomeStats;
    use crate::store::InMemoryStore;
    use crate::types::RecordStatus;

    fn create_test_record(id: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.into(),
            embedding: vec![1.0, 2.0, 3.0],
            context: format!("Context for {id}"),
            outcome: 0.5,
            metadata: Default::default(),
            created_at: 1234567890,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        }
    }

    #[test]
    fn test_buffer_insert_and_flush() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        buffer.insert(create_test_record("rec-1")).unwrap();
        buffer.insert(create_test_record("rec-2")).unwrap();

        assert_eq!(buffer.len(), 2);

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));

        let result = buffer.flush(&mut store, &mut index).unwrap();

        assert_eq!(result.inserts, 2);
        assert_eq!(store.len(), 2);
        assert_eq!(index.len(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_update_stats() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        buffer.insert(create_test_record("rec-1")).unwrap();
        buffer.update_stats(&"rec-1".into(), 0.8).unwrap();
        buffer.update_stats(&"rec-1".into(), 0.9).unwrap();

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));

        let result = buffer.flush(&mut store, &mut index).unwrap();

        assert_eq!(result.inserts, 1);
        assert_eq!(result.updates, 2);

        let record = store.get(&"rec-1".into()).unwrap();
        assert_eq!(record.stats.count(), 2);
    }

    #[test]
    fn test_buffer_delete() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        buffer.insert(create_test_record("rec-1")).unwrap();
        buffer.insert(create_test_record("rec-2")).unwrap();

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));

        // First flush to populate store/index
        buffer.flush(&mut store, &mut index).unwrap();

        // Now delete
        buffer.delete(&"rec-1".into()).unwrap();
        let result = buffer.flush(&mut store, &mut index).unwrap();

        assert_eq!(result.deletes, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_auto_flush_by_ops() {
        let config = WriteBufferConfig::new()
            .without_wal()
            .with_max_ops(5);
        let buffer = WriteBuffer::without_wal(config);

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));

        for i in 0..4 {
            buffer.insert(create_test_record(&format!("rec-{i}"))).unwrap();
            buffer.maybe_flush(&mut store, &mut index).unwrap();
        }

        // Should not have flushed yet
        assert!(!buffer.is_empty());

        // 5th insert triggers flush check
        buffer.insert(create_test_record("rec-4")).unwrap();
        let flushed = buffer.maybe_flush(&mut store, &mut index).unwrap();

        assert!(flushed);
        assert!(buffer.is_empty());
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn test_buffer_stats() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        buffer.insert(create_test_record("rec-1")).unwrap();
        buffer.insert(create_test_record("rec-2")).unwrap();
        buffer.update_stats(&"rec-1".into(), 0.8).unwrap();
        buffer.delete(&"rec-2".into()).unwrap();

        let stats = buffer.stats();
        assert_eq!(stats.buffered_ops, 4);
        assert!(stats.buffered_bytes > 0);
        assert_eq!(stats.total_inserts, 2);
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.total_deletes, 1);

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));
        buffer.flush(&mut store, &mut index).unwrap();

        let stats_after = buffer.stats();
        assert_eq!(stats_after.buffered_ops, 0);
        assert_eq!(stats_after.flush_count, 1);
    }

    #[test]
    fn test_flush_to_store_only() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        buffer.insert(create_test_record("rec-1")).unwrap();

        let mut store = InMemoryStore::new();
        let result = buffer.flush_to_store(&mut store).unwrap();

        assert_eq!(result.inserts, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_empty_flush() {
        let config = WriteBufferConfig::new().without_wal();
        let buffer = WriteBuffer::without_wal(config);

        let mut store = InMemoryStore::new();
        let mut index = FlatIndex::new(IndexConfig::new(3));

        let result = buffer.flush(&mut store, &mut index).unwrap();
        assert_eq!(result.total(), 0);
    }
}
