//! WAL Entry Types
//!
//! Defines the structure of write-ahead log entries.

use crate::types::{MemoryRecord, RecordId};
use rkyv::{Archive, Deserialize, Serialize};

/// Type of WAL entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum WalEntryType {
    /// Insert a new record
    Insert,
    /// Update statistics for a record
    UpdateStats,
    /// Delete a record
    Delete,
    /// Checkpoint marker
    Checkpoint,
}

/// A single WAL entry.
///
/// Each entry is self-contained with all information needed to replay it.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct WalEntry {
    /// Monotonically increasing sequence number
    pub sequence: u64,
    /// Entry type
    pub entry_type: WalEntryType,
    /// Timestamp (Unix epoch millis)
    pub timestamp_ms: u64,
    /// Record ID (for all entry types)
    pub record_id: String,
    /// Full record data (for Insert)
    pub record_data: Option<WalRecordData>,
    /// Outcome value (for UpdateStats)
    pub outcome: Option<f64>,
    /// CRC32 checksum for integrity
    pub checksum: u32,
}

/// Serializable record data for WAL.
///
/// Subset of MemoryRecord needed for reconstruction.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct WalRecordData {
    /// Record ID
    pub id: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Context string
    pub context: String,
    /// Initial outcome
    pub outcome: f64,
    /// Metadata (serialized as JSON string for simplicity)
    pub metadata_json: String,
    /// Creation timestamp
    pub created_at: u64,
}

impl WalRecordData {
    /// Create from a MemoryRecord.
    #[must_use]
    pub fn from_record(record: &MemoryRecord) -> Self {
        Self {
            id: record.id.to_string(),
            embedding: record.embedding.clone(),
            context: record.context.clone(),
            outcome: record.outcome,
            metadata_json: "{}".to_string(), // TODO: serialize metadata
            created_at: record.created_at,
        }
    }
}

impl WalEntry {
    /// Create a new Insert entry.
    #[must_use]
    pub fn insert(sequence: u64, record: &MemoryRecord) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Insert,
            timestamp_ms: current_time_ms(),
            record_id: record.id.to_string(),
            record_data: Some(WalRecordData::from_record(record)),
            outcome: None,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a new UpdateStats entry.
    #[must_use]
    pub fn update_stats(sequence: u64, record_id: &RecordId, outcome: f64) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::UpdateStats,
            timestamp_ms: current_time_ms(),
            record_id: record_id.to_string(),
            record_data: None,
            outcome: Some(outcome),
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a new Delete entry.
    #[must_use]
    pub fn delete(sequence: u64, record_id: &RecordId) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Delete,
            timestamp_ms: current_time_ms(),
            record_id: record_id.to_string(),
            record_data: None,
            outcome: None,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a checkpoint marker.
    #[must_use]
    pub fn checkpoint(sequence: u64) -> Self {
        let mut entry = Self {
            sequence,
            entry_type: WalEntryType::Checkpoint,
            timestamp_ms: current_time_ms(),
            record_id: String::new(),
            record_data: None,
            outcome: None,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32 checksum for this entry.
    fn compute_checksum(&self) -> u32 {
        use xxhash_rust::xxh32::xxh32;

        // Build a byte buffer of key fields
        let mut data = Vec::new();

        // Hash key fields
        data.extend_from_slice(&self.sequence.to_le_bytes());
        data.push(self.entry_type as u8);
        data.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        data.extend_from_slice(self.record_id.as_bytes());

        if let Some(ref record_data) = self.record_data {
            data.extend_from_slice(record_data.id.as_bytes());
            data.extend_from_slice(record_data.context.as_bytes());
            data.extend_from_slice(&record_data.outcome.to_bits().to_le_bytes());
        }

        if let Some(outcome) = self.outcome {
            data.extend_from_slice(&outcome.to_bits().to_le_bytes());
        }

        xxh32(&data, 0)
    }

    /// Verify the checksum.
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        let mut copy = self.clone();
        copy.checksum = 0;
        copy.checksum = copy.compute_checksum();
        copy.checksum == self.checksum
    }

    /// Serialize to bytes using rkyv.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 256>(self)
            .expect("WAL entry serialization should not fail")
            .to_vec()
    }

    /// Deserialize from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails or checksum is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WalError> {
        let archived = rkyv::check_archived_root::<Self>(bytes)
            .map_err(|e| WalError::Corrupted(format!("Failed to deserialize: {e}")))?;

        let entry: Self = archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|_| WalError::Corrupted("Deserialization failed".into()))?;

        if !entry.verify_checksum() {
            return Err(WalError::ChecksumMismatch);
        }

        Ok(entry)
    }
}

/// WAL-specific errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WalError {
    #[error("WAL entry corrupted: {0}")]
    Corrupted(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("IO error: {0}")]
    Io(String),

    #[error("WAL is full")]
    Full,
}

/// Get current time in milliseconds.
fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::OutcomeStats;
    use crate::types::RecordStatus;

    fn create_test_record() -> MemoryRecord {
        MemoryRecord {
            id: "test-record".into(),
            embedding: vec![1.0, 2.0, 3.0],
            context: "Test context".into(),
            outcome: 0.8,
            metadata: Default::default(),
            created_at: 1234567890,
            status: RecordStatus::Active,
            stats: OutcomeStats::new(1),
        }
    }

    #[test]
    fn test_insert_entry() {
        let record = create_test_record();
        let entry = WalEntry::insert(1, &record);

        assert_eq!(entry.sequence, 1);
        assert_eq!(entry.entry_type, WalEntryType::Insert);
        assert_eq!(entry.record_id, "test-record");
        assert!(entry.record_data.is_some());
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_update_stats_entry() {
        let entry = WalEntry::update_stats(2, &"rec-1".into(), 0.9);

        assert_eq!(entry.sequence, 2);
        assert_eq!(entry.entry_type, WalEntryType::UpdateStats);
        assert_eq!(entry.record_id, "rec-1");
        assert_eq!(entry.outcome, Some(0.9));
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_delete_entry() {
        let entry = WalEntry::delete(3, &"rec-2".into());

        assert_eq!(entry.sequence, 3);
        assert_eq!(entry.entry_type, WalEntryType::Delete);
        assert_eq!(entry.record_id, "rec-2");
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_checkpoint_entry() {
        let entry = WalEntry::checkpoint(100);

        assert_eq!(entry.sequence, 100);
        assert_eq!(entry.entry_type, WalEntryType::Checkpoint);
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let record = create_test_record();
        let entry = WalEntry::insert(1, &record);

        let bytes = entry.to_bytes();
        let restored = WalEntry::from_bytes(&bytes).unwrap();

        assert_eq!(restored.sequence, entry.sequence);
        assert_eq!(restored.entry_type, entry.entry_type);
        assert_eq!(restored.record_id, entry.record_id);
        assert!(restored.verify_checksum());
    }

    #[test]
    fn test_corrupted_bytes() {
        let record = create_test_record();
        let entry = WalEntry::insert(1, &record);

        let mut bytes = entry.to_bytes();
        // Corrupt a byte
        if !bytes.is_empty() {
            let mid = bytes.len() / 2;
            bytes[mid] ^= 0xFF;
        }

        let result = WalEntry::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_tamper_detection() {
        let mut entry = WalEntry::update_stats(1, &"test".into(), 0.5);

        // Tamper with data after checksum
        entry.outcome = Some(0.99);

        assert!(!entry.verify_checksum());
    }
}
