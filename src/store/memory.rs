//! In-Memory Record Store
//!
//! High-performance in-memory storage for MemoryRecords using HashMap.
//!
//! # Thread Safety
//!
//! The store itself is not thread-safe. For concurrent access,
//! wrap it in `SharedStore` using `shared_store()`.
//!
//! # Performance
//!
//! - Insert: O(1) amortized
//! - Get: O(1)
//! - Remove: O(1)
//! - Memory: ~(record_size * n) + HashMap overhead

use crate::error::{Error, Result};
use crate::store::traits::RecordStore;
use crate::types::{MemoryRecord, RecordId, RecordStatus};
use ahash::AHashMap;

/// In-memory record store.
///
/// Stores records in a HashMap for O(1) access.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::store::{InMemoryStore, RecordStore};
/// use rag_plusplus_core::types::MemoryRecord;
///
/// let mut store = InMemoryStore::new();
/// store.insert(record)?;
///
/// if let Some(record) = store.get(&"my-id".into()) {
///     println!("Found: {}", record.context);
/// }
/// ```
#[derive(Debug, Default)]
pub struct InMemoryStore {
    /// Records by ID
    records: AHashMap<RecordId, MemoryRecord>,
    /// Track memory usage (approximate)
    estimated_bytes: usize,
}

impl InMemoryStore {
    /// Create a new empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: AHashMap::new(),
            estimated_bytes: 0,
        }
    }

    /// Create store with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            records: AHashMap::with_capacity(capacity),
            estimated_bytes: 0,
        }
    }

    /// Estimate memory usage of a record.
    fn estimate_record_size(record: &MemoryRecord) -> usize {
        // Base struct size
        let base = std::mem::size_of::<MemoryRecord>();

        // String allocations
        let id_size = record.id.capacity();
        let context_size = record.context.capacity();

        // Embedding vector
        let embedding_size = record.embedding.capacity() * std::mem::size_of::<f32>();

        // Metadata (rough estimate)
        let metadata_size = record.metadata.len() * 64; // Assume 64 bytes per entry

        base + id_size + context_size + embedding_size + metadata_size
    }

    /// Get statistics about the store.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        let active_count = self
            .records
            .values()
            .filter(|r| r.status == RecordStatus::Active)
            .count();

        let total_outcomes: u64 = self.records.values().map(|r| r.stats.count()).sum();

        StoreStats {
            total_records: self.records.len(),
            active_records: active_count,
            total_outcome_updates: total_outcomes,
            memory_bytes: self.estimated_bytes,
        }
    }

    /// Iterate over all records.
    pub fn iter(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.records.values()
    }

    /// Iterate over active records only.
    pub fn iter_active(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.records
            .values()
            .filter(|r| r.status == RecordStatus::Active)
    }

    /// Get mutable reference to a record (internal use only).
    fn get_mut(&mut self, id: &RecordId) -> Option<&mut MemoryRecord> {
        self.records.get_mut(id)
    }

    /// Estimated memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.estimated_bytes
    }
}

impl RecordStore for InMemoryStore {
    fn insert(&mut self, record: MemoryRecord) -> Result<RecordId> {
        let id = record.id.clone();

        if self.records.contains_key(&id) {
            return Err(Error::DuplicateRecord {
                record_id: id.to_string(),
            });
        }

        let size = Self::estimate_record_size(&record);
        self.records.insert(id.clone(), record);
        self.estimated_bytes += size;

        Ok(id)
    }

    fn get(&self, id: &RecordId) -> Option<MemoryRecord> {
        self.records.get(id).cloned()
    }

    fn contains(&self, id: &RecordId) -> bool {
        self.records.contains_key(id)
    }

    fn update_stats(&mut self, id: &RecordId, outcome: f64) -> Result<()> {
        let record = self.get_mut(id).ok_or_else(|| Error::RecordNotFound {
            record_id: id.to_string(),
        })?;

        // Only update stats, never modify core record data (INV-001)
        record.stats.update_scalar(outcome);

        Ok(())
    }

    fn remove(&mut self, id: &RecordId) -> Result<bool> {
        if let Some(record) = self.records.remove(id) {
            let size = Self::estimate_record_size(&record);
            self.estimated_bytes = self.estimated_bytes.saturating_sub(size);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn clear(&mut self) {
        self.records.clear();
        self.estimated_bytes = 0;
    }

    fn ids(&self) -> Vec<RecordId> {
        self.records.keys().cloned().collect()
    }

    fn memory_usage(&self) -> usize {
        self.estimated_bytes
    }
}

/// Store statistics.
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    /// Total number of records (including archived/deleted)
    pub total_records: usize,
    /// Number of active records
    pub active_records: usize,
    /// Total outcome updates across all records
    pub total_outcome_updates: u64,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::traits::tests::*;
    use crate::OutcomeStats;

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
    fn test_new_store() {
        let store = InMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let store = InMemoryStore::with_capacity(100);
        assert!(store.is_empty());
    }

    #[test]
    fn test_insert_and_get() {
        let mut store = InMemoryStore::new();
        let record = create_test_record("test-1");

        let id = store.insert(record).unwrap();
        assert_eq!(id.as_str(), "test-1");

        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.id.as_str(), "test-1");
        assert_eq!(retrieved.context, "Context for test-1");
    }

    #[test]
    fn test_duplicate_insert_error() {
        let mut store = InMemoryStore::new();
        let record = create_test_record("dup");

        store.insert(record.clone()).unwrap();
        let result = store.insert(record);

        assert!(result.is_err());
    }

    #[test]
    fn test_update_stats() {
        let mut store = InMemoryStore::new();
        store.insert(create_test_record("stats-test")).unwrap();

        let id: RecordId = "stats-test".into();

        store.update_stats(&id, 0.8).unwrap();
        store.update_stats(&id, 0.9).unwrap();
        store.update_stats(&id, 0.7).unwrap();

        let record = store.get(&id).unwrap();
        assert_eq!(record.stats.count(), 3);
        assert!((record.stats.mean_scalar().unwrap() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_update_stats_not_found() {
        let mut store = InMemoryStore::new();
        let result = store.update_stats(&"nonexistent".into(), 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove() {
        let mut store = InMemoryStore::new();
        store.insert(create_test_record("to-remove")).unwrap();

        assert_eq!(store.len(), 1);

        let removed = store.remove(&"to-remove".into()).unwrap();
        assert!(removed);
        assert_eq!(store.len(), 0);

        let removed_again = store.remove(&"to-remove".into()).unwrap();
        assert!(!removed_again);
    }

    #[test]
    fn test_iter() {
        let mut store = InMemoryStore::new();

        for i in 0..5 {
            store.insert(create_test_record(&format!("iter-{i}"))).unwrap();
        }

        let count = store.iter().count();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_iter_active() {
        let mut store = InMemoryStore::new();

        for i in 0..5 {
            let mut record = create_test_record(&format!("active-{i}"));
            if i % 2 == 0 {
                record.status = RecordStatus::Archived;
            }
            store.insert(record).unwrap();
        }

        let active_count = store.iter_active().count();
        assert_eq!(active_count, 2); // Only odd indices are active
    }

    #[test]
    fn test_stats() {
        let mut store = InMemoryStore::new();

        for i in 0..10 {
            store.insert(create_test_record(&format!("stat-{i}"))).unwrap();
        }

        store.update_stats(&"stat-0".into(), 0.5).unwrap();
        store.update_stats(&"stat-0".into(), 0.6).unwrap();
        store.update_stats(&"stat-1".into(), 0.7).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_records, 10);
        assert_eq!(stats.active_records, 10);
        assert_eq!(stats.total_outcome_updates, 3);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_memory_tracking() {
        let mut store = InMemoryStore::new();

        let initial = store.memory_usage();
        assert_eq!(initial, 0);

        store.insert(create_test_record("mem-1")).unwrap();
        let after_one = store.memory_usage();
        assert!(after_one > 0);

        store.insert(create_test_record("mem-2")).unwrap();
        let after_two = store.memory_usage();
        assert!(after_two > after_one);

        store.remove(&"mem-1".into()).unwrap();
        let after_remove = store.memory_usage();
        assert!(after_remove < after_two);
    }

    #[test]
    fn test_clear() {
        let mut store = InMemoryStore::new();

        for i in 0..10 {
            store.insert(create_test_record(&format!("clear-{i}"))).unwrap();
        }

        assert_eq!(store.len(), 10);
        assert!(store.memory_usage() > 0);

        store.clear();

        assert!(store.is_empty());
        assert_eq!(store.memory_usage(), 0);
    }

    #[test]
    fn test_ids() {
        let mut store = InMemoryStore::new();

        for i in 0..5 {
            store.insert(create_test_record(&format!("id-{i}"))).unwrap();
        }

        let ids = store.ids();
        assert_eq!(ids.len(), 5);
    }

    // Run trait compliance tests
    #[test]
    fn test_trait_basic_crud() {
        let mut store = InMemoryStore::new();
        test_basic_crud(&mut store);
    }

    #[test]
    fn test_trait_batch_operations() {
        let mut store = InMemoryStore::new();
        test_batch_operations(&mut store);
    }

    #[test]
    fn test_trait_duplicate_insert() {
        let mut store = InMemoryStore::new();
        test_duplicate_insert(&mut store);
    }
}
