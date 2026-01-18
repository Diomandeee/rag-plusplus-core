//! Store Traits
//!
//! Defines the core abstraction for record storage.

use crate::error::Result;
use crate::types::{MemoryRecord, RecordId};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::sync::Arc;

/// Core trait for record storage.
///
/// All storage implementations must implement this trait.
/// The trait enforces INV-001 (record immutability) by only allowing
/// stats updates after insertion.
pub trait RecordStore: Send + Sync + Debug {
    /// Insert a new record.
    ///
    /// # Errors
    ///
    /// Returns error if record with same ID already exists.
    fn insert(&mut self, record: MemoryRecord) -> Result<RecordId>;

    /// Insert multiple records in batch.
    ///
    /// Default implementation calls `insert` repeatedly.
    fn insert_batch(&mut self, records: Vec<MemoryRecord>) -> Result<Vec<RecordId>> {
        let mut ids = Vec::with_capacity(records.len());
        for record in records {
            ids.push(self.insert(record)?);
        }
        Ok(ids)
    }

    /// Get a record by ID.
    fn get(&self, id: &RecordId) -> Option<MemoryRecord>;

    /// Get multiple records by ID.
    fn get_batch(&self, ids: &[RecordId]) -> Vec<Option<MemoryRecord>> {
        ids.iter().map(|id| self.get(id)).collect()
    }

    /// Check if a record exists.
    fn contains(&self, id: &RecordId) -> bool;

    /// Update the outcome statistics for a record.
    ///
    /// This is the ONLY mutable operation allowed on a record after insertion
    /// (enforces INV-001).
    ///
    /// # Arguments
    ///
    /// * `id` - Record ID
    /// * `outcome` - New outcome value to incorporate
    fn update_stats(&mut self, id: &RecordId, outcome: f64) -> Result<()>;

    /// Remove a record (marks as deleted, may not physically remove).
    ///
    /// # Returns
    ///
    /// `true` if record was found and removed.
    fn remove(&mut self, id: &RecordId) -> Result<bool>;

    /// Number of records in the store.
    fn len(&self) -> usize;

    /// Whether the store is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all records.
    fn clear(&mut self);

    /// Get all record IDs.
    fn ids(&self) -> Vec<RecordId>;

    /// Get memory usage estimate in bytes.
    fn memory_usage(&self) -> usize;
}

/// Thread-safe shared store using Arc<RwLock>.
pub type SharedStore<S> = Arc<RwLock<S>>;

/// Create a shared store from a store instance.
#[allow(dead_code)]
pub fn shared_store<S: RecordStore>(store: S) -> SharedStore<S> {
    Arc::new(RwLock::new(store))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::OutcomeStats;

    // Trait tests that any implementation should pass
    pub fn test_basic_crud<S: RecordStore>(store: &mut S) {
        use crate::types::RecordStatus;

        // Insert
        let record = MemoryRecord {
            id: "test-1".into(),
            embedding: vec![1.0, 2.0, 3.0],
            context: "test context".into(),
            outcome: 0.8,
            metadata: Default::default(),
            created_at: 0,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        };

        let id = store.insert(record.clone()).unwrap();
        assert_eq!(id.as_str(), "test-1");

        // Get
        let retrieved = store.get(&id).unwrap();
        assert_eq!(retrieved.id.as_str(), "test-1");
        assert!((retrieved.outcome - 0.8).abs() < 0.001);

        // Contains
        assert!(store.contains(&id));
        assert!(!store.contains(&"nonexistent".into()));

        // Length
        assert_eq!(store.len(), 1);

        // Update stats
        store.update_stats(&id, 0.9).unwrap();
        store.update_stats(&id, 0.7).unwrap();

        let updated = store.get(&id).unwrap();
        assert_eq!(updated.stats.count(), 2);

        // Remove
        assert!(store.remove(&id).unwrap());
        assert!(!store.contains(&id));
        assert_eq!(store.len(), 0);
    }

    pub fn test_batch_operations<S: RecordStore>(store: &mut S) {
        use crate::types::RecordStatus;

        let records: Vec<MemoryRecord> = (0..10)
            .map(|i| MemoryRecord {
                id: format!("batch-{i}").into(),
                embedding: vec![i as f32; 3],
                context: format!("context {i}"),
                outcome: i as f64 / 10.0,
                metadata: Default::default(),
                created_at: i as u64,
                status: RecordStatus::Active,
                stats: OutcomeStats::new(1),
            })
            .collect();

        let ids = store.insert_batch(records).unwrap();
        assert_eq!(ids.len(), 10);
        assert_eq!(store.len(), 10);

        // Get batch
        let retrieved = store.get_batch(&ids[..3]);
        assert_eq!(retrieved.len(), 3);
        assert!(retrieved.iter().all(|r| r.is_some()));

        // Get IDs
        let all_ids = store.ids();
        assert_eq!(all_ids.len(), 10);

        // Clear
        store.clear();
        assert!(store.is_empty());
    }

    pub fn test_duplicate_insert<S: RecordStore>(store: &mut S) {
        use crate::types::RecordStatus;

        let record = MemoryRecord {
            id: "dup-test".into(),
            embedding: vec![1.0],
            context: "test".into(),
            outcome: 0.5,
            metadata: Default::default(),
            created_at: 0,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        };

        store.insert(record.clone()).unwrap();
        let result = store.insert(record);
        assert!(result.is_err());
    }
}
