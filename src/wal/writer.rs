//! WAL Writer
//!
//! Appends entries to the write-ahead log with durability guarantees.

use crate::error::{Error, Result};
use crate::types::{MemoryRecord, RecordId};
use crate::wal::entry::WalEntry;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// WAL configuration.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory for WAL files
    pub directory: PathBuf,
    /// Maximum WAL file size before rotation (bytes)
    pub max_file_size: u64,
    /// Whether to sync after every write
    pub sync_on_write: bool,
    /// Buffer size for writes
    pub buffer_size: usize,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("./wal"),
            max_file_size: 64 * 1024 * 1024, // 64 MB
            sync_on_write: true,
            buffer_size: 64 * 1024, // 64 KB buffer
        }
    }
}

impl WalConfig {
    /// Create new config with directory.
    #[must_use]
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            ..Default::default()
        }
    }

    /// Set maximum file size.
    #[must_use]
    pub const fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set sync behavior.
    #[must_use]
    pub const fn with_sync_on_write(mut self, sync: bool) -> Self {
        self.sync_on_write = sync;
        self
    }
}

/// Write-ahead log writer.
///
/// Thread-safe WAL writer with durable append operations.
///
/// # Durability
///
/// When `sync_on_write` is enabled, each write is followed by an fsync
/// to ensure the data is persisted to disk before returning.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::wal::{WalWriter, WalConfig};
///
/// let config = WalConfig::new("./wal");
/// let mut writer = WalWriter::new(config)?;
///
/// writer.log_insert(&record)?;
/// writer.log_update_stats(&id, 0.9)?;
/// ```
pub struct WalWriter {
    config: WalConfig,
    /// Current sequence number
    sequence: AtomicU64,
    /// Current file handle
    file: Mutex<Option<BufWriter<File>>>,
    /// Current file size
    file_size: AtomicU64,
    /// Current file number
    file_number: AtomicU64,
}

impl std::fmt::Debug for WalWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalWriter")
            .field("config", &self.config)
            .field("sequence", &self.sequence)
            .field("file_size", &self.file_size)
            .field("file_number", &self.file_number)
            .finish()
    }
}

impl WalWriter {
    /// Create a new WAL writer.
    ///
    /// Creates the WAL directory if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns error if directory creation or file opening fails.
    pub fn new(config: WalConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.directory).map_err(|e| Error::WalWrite {
            reason: format!("Failed to create WAL directory: {e}"),
        })?;

        let writer = Self {
            config,
            sequence: AtomicU64::new(0),
            file: Mutex::new(None),
            file_size: AtomicU64::new(0),
            file_number: AtomicU64::new(0),
        };

        // Find highest existing sequence number
        writer.recover_sequence()?;

        Ok(writer)
    }

    /// Recover the sequence number from existing WAL files.
    fn recover_sequence(&self) -> Result<()> {
        let mut max_seq = 0u64;
        let mut max_file = 0u64;

        if let Ok(entries) = std::fs::read_dir(&self.config.directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Some(num_str) = name.strip_prefix("wal_").and_then(|s| s.strip_suffix(".log")) {
                        if let Ok(num) = num_str.parse::<u64>() {
                            max_file = max_file.max(num);

                            // Read last entry to get sequence
                            if let Ok(last_seq) = self.get_last_sequence(&path) {
                                max_seq = max_seq.max(last_seq);
                            }
                        }
                    }
                }
            }
        }

        self.sequence.store(max_seq, Ordering::SeqCst);
        self.file_number.store(max_file + 1, Ordering::SeqCst);

        Ok(())
    }

    /// Get the last sequence number from a WAL file.
    fn get_last_sequence(&self, path: &Path) -> Result<u64> {
        let reader = crate::wal::reader::WalReader::open(path)?;
        let entries: Vec<_> = reader.collect();

        if let Some(Ok(last)) = entries.last() {
            Ok(last.sequence)
        } else {
            Ok(0)
        }
    }

    /// Get the current WAL file path.
    fn current_file_path(&self) -> PathBuf {
        let num = self.file_number.load(Ordering::Relaxed);
        self.config.directory.join(format!("wal_{num:08}.log"))
    }

    /// Ensure we have an open file.
    fn ensure_file(&self) -> Result<()> {
        let mut file_guard = self.file.lock();

        if file_guard.is_none() {
            let path = self.current_file_path();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| Error::WalWrite {
                    reason: format!("Failed to open WAL file: {e}"),
                })?;

            let size = file.metadata().map(|m| m.len()).unwrap_or(0);
            self.file_size.store(size, Ordering::SeqCst);

            *file_guard = Some(BufWriter::with_capacity(self.config.buffer_size, file));
        }

        Ok(())
    }

    /// Rotate to a new WAL file if needed.
    fn maybe_rotate(&self) -> Result<()> {
        let size = self.file_size.load(Ordering::Relaxed);

        if size >= self.config.max_file_size {
            let mut file_guard = self.file.lock();

            // Flush and close current file
            if let Some(mut f) = file_guard.take() {
                f.flush().map_err(|e| Error::WalWrite {
                    reason: format!("Failed to flush WAL: {e}"),
                })?;
            }

            // Increment file number
            self.file_number.fetch_add(1, Ordering::SeqCst);
            self.file_size.store(0, Ordering::SeqCst);
        }

        Ok(())
    }

    /// Write an entry to the WAL.
    fn write_entry(&self, entry: &WalEntry) -> Result<()> {
        self.maybe_rotate()?;
        self.ensure_file()?;

        let bytes = entry.to_bytes();
        let entry_size = bytes.len() as u64;

        let mut file_guard = self.file.lock();
        let writer = file_guard.as_mut().ok_or_else(|| Error::WalWrite {
            reason: "WAL file not open".into(),
        })?;

        // Write length prefix (4 bytes, little-endian)
        writer
            .write_all(&(bytes.len() as u32).to_le_bytes())
            .map_err(|e| Error::WalWrite {
                reason: format!("Failed to write length: {e}"),
            })?;

        // Write entry data
        writer.write_all(&bytes).map_err(|e| Error::WalWrite {
            reason: format!("Failed to write entry: {e}"),
        })?;

        // Optionally sync
        if self.config.sync_on_write {
            writer.flush().map_err(|e| Error::WalWrite {
                reason: format!("Failed to flush: {e}"),
            })?;

            // fsync the underlying file
            writer.get_ref().sync_all().map_err(|e| Error::WalWrite {
                reason: format!("Failed to sync: {e}"),
            })?;
        }

        // Update file size
        self.file_size
            .fetch_add(4 + entry_size, Ordering::Relaxed);

        Ok(())
    }

    /// Get the next sequence number.
    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Log an insert operation.
    ///
    /// # Errors
    ///
    /// Returns error if write fails.
    pub fn log_insert(&self, record: &MemoryRecord) -> Result<u64> {
        let seq = self.next_sequence();
        let entry = WalEntry::insert(seq, record);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Log a stats update operation.
    pub fn log_update_stats(&self, record_id: &RecordId, outcome: f64) -> Result<u64> {
        let seq = self.next_sequence();
        let entry = WalEntry::update_stats(seq, record_id, outcome);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Log a delete operation.
    pub fn log_delete(&self, record_id: &RecordId) -> Result<u64> {
        let seq = self.next_sequence();
        let entry = WalEntry::delete(seq, record_id);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Log a checkpoint marker.
    pub fn log_checkpoint(&self) -> Result<u64> {
        let seq = self.next_sequence();
        let entry = WalEntry::checkpoint(seq);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Get current sequence number.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }

    /// Flush pending writes.
    pub fn flush(&self) -> Result<()> {
        let mut file_guard = self.file.lock();
        if let Some(writer) = file_guard.as_mut() {
            writer.flush().map_err(|e| Error::WalWrite {
                reason: format!("Failed to flush: {e}"),
            })?;
            writer.get_ref().sync_all().map_err(|e| Error::WalWrite {
                reason: format!("Failed to sync: {e}"),
            })?;
        }
        Ok(())
    }

    /// Close the WAL writer.
    pub fn close(&self) -> Result<()> {
        self.flush()?;
        let mut file_guard = self.file.lock();
        *file_guard = None;
        Ok(())
    }

    /// Get WAL directory.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.config.directory
    }

    /// Get list of WAL files.
    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.config.directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "log") {
                    files.push(path);
                }
            }
        }

        files.sort();
        Ok(files)
    }

    /// Truncate WAL files up to (but not including) a checkpoint.
    ///
    /// This removes entries that have been successfully checkpointed.
    pub fn truncate_before(&self, checkpoint_seq: u64) -> Result<()> {
        let files = self.list_files()?;

        for file_path in files {
            // Check if all entries in this file are before the checkpoint
            let reader = crate::wal::reader::WalReader::open(&file_path)?;
            let entries: Vec<_> = reader.collect();

            if let Some(Ok(last)) = entries.last() {
                if last.sequence < checkpoint_seq {
                    // Safe to delete this file
                    std::fs::remove_file(&file_path).map_err(|e| Error::WalWrite {
                        reason: format!("Failed to remove old WAL file: {e}"),
                    })?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::OutcomeStats;
    use crate::types::RecordStatus;
    use tempfile::TempDir;

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
    fn test_wal_writer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path());
        let writer = WalWriter::new(config).unwrap();

        assert_eq!(writer.sequence(), 0);
    }

    #[test]
    fn test_log_insert() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path());
        let writer = WalWriter::new(config).unwrap();

        let record = create_test_record("test-1");
        let seq = writer.log_insert(&record).unwrap();

        assert_eq!(seq, 1);
        assert_eq!(writer.sequence(), 1);
    }

    #[test]
    fn test_log_multiple_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path());
        let writer = WalWriter::new(config).unwrap();

        writer.log_insert(&create_test_record("rec-1")).unwrap();
        writer.log_insert(&create_test_record("rec-2")).unwrap();
        writer.log_update_stats(&"rec-1".into(), 0.8).unwrap();
        writer.log_delete(&"rec-2".into()).unwrap();

        assert_eq!(writer.sequence(), 4);
    }

    #[test]
    fn test_wal_file_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path());
        let writer = WalWriter::new(config).unwrap();

        writer.log_insert(&create_test_record("test")).unwrap();
        writer.flush().unwrap();

        let files = writer.list_files().unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_sequence_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Write some entries
        {
            let config = WalConfig::new(temp_dir.path());
            let writer = WalWriter::new(config).unwrap();
            writer.log_insert(&create_test_record("rec-1")).unwrap();
            writer.log_insert(&create_test_record("rec-2")).unwrap();
            writer.log_insert(&create_test_record("rec-3")).unwrap();
            writer.flush().unwrap();
        }

        // Reopen and check sequence
        {
            let config = WalConfig::new(temp_dir.path());
            let writer = WalWriter::new(config).unwrap();
            assert_eq!(writer.sequence(), 3);

            // New entries should continue from 4
            let seq = writer.log_insert(&create_test_record("rec-4")).unwrap();
            assert_eq!(seq, 4);
        }
    }

    #[test]
    fn test_file_rotation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(1024); // Very small for testing

        let writer = WalWriter::new(config).unwrap();

        // Write many entries to trigger rotation
        for i in 0..50 {
            writer
                .log_insert(&create_test_record(&format!("rec-{i}")))
                .unwrap();
        }
        writer.flush().unwrap();

        let files = writer.list_files().unwrap();
        assert!(files.len() > 1, "Expected multiple WAL files after rotation");
    }

    #[test]
    fn test_checkpoint_and_truncate() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig::new(temp_dir.path())
            .with_max_file_size(512); // Small for rotation

        let writer = WalWriter::new(config).unwrap();

        // Write entries across multiple files
        for i in 0..20 {
            writer
                .log_insert(&create_test_record(&format!("rec-{i}")))
                .unwrap();
        }

        // Add checkpoint
        let checkpoint_seq = writer.log_checkpoint().unwrap();
        writer.flush().unwrap();

        // Count files before truncation
        let files_before = writer.list_files().unwrap().len();

        // Truncate before checkpoint
        writer.truncate_before(checkpoint_seq).unwrap();

        // Should have fewer files
        let files_after = writer.list_files().unwrap().len();
        assert!(files_after <= files_before);
    }
}
