//! WAL Reader
//!
//! Reads and replays WAL entries for recovery.

use crate::error::{Error, Result};
use crate::wal::entry::WalEntry;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// WAL file reader.
///
/// Provides iteration over WAL entries for recovery.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::wal::WalReader;
///
/// let reader = WalReader::open("./wal/wal_00000001.log")?;
///
/// for entry in reader {
///     match entry {
///         Ok(e) => println!("Seq {}: {:?}", e.sequence, e.entry_type),
///         Err(e) => eprintln!("Error: {}", e),
///     }
/// }
/// ```
pub struct WalReader {
    reader: BufReader<File>,
    entries_read: u64,
}

impl WalReader {
    /// Open a WAL file for reading.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::WalRecovery {
            reason: format!("Failed to open WAL file: {e}"),
        })?;

        Ok(Self {
            reader: BufReader::new(file),
            entries_read: 0,
        })
    }

    /// Read the next entry from the WAL.
    fn read_entry(&mut self) -> Option<Result<WalEntry>> {
        // Read length prefix (4 bytes)
        let mut len_buf = [0u8; 4];
        match self.reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return None; // End of file
            }
            Err(e) => {
                return Some(Err(Error::WalRecovery {
                    reason: format!("Failed to read length: {e}"),
                }));
            }
        }

        let len = u32::from_le_bytes(len_buf) as usize;

        // Sanity check length
        if len > 10 * 1024 * 1024 {
            // 10 MB max entry size
            return Some(Err(Error::WalRecovery {
                reason: format!("Entry size too large: {len}"),
            }));
        }

        // Read entry data
        let mut data = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut data) {
            return Some(Err(Error::WalRecovery {
                reason: format!("Failed to read entry data: {e}"),
            }));
        }

        // Deserialize
        match WalEntry::from_bytes(&data) {
            Ok(entry) => {
                self.entries_read += 1;
                Some(Ok(entry))
            }
            Err(e) => Some(Err(Error::WalRecovery {
                reason: format!("Failed to deserialize entry: {e}"),
            })),
        }
    }

    /// Number of entries successfully read.
    #[must_use]
    pub fn entries_read(&self) -> u64 {
        self.entries_read
    }
}

impl Iterator for WalReader {
    type Item = Result<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_entry()
    }
}

/// Read all entries from multiple WAL files.
///
/// Files are read in sorted order (by filename).
#[allow(dead_code)]
pub struct MultiFileReader {
    files: Vec<std::path::PathBuf>,
    current_reader: Option<WalReader>,
    current_index: usize,
    total_entries: u64,
}

#[allow(dead_code)]
impl MultiFileReader {
    /// Create reader from a directory of WAL files.
    ///
    /// # Errors
    ///
    /// Returns error if directory cannot be read.
    pub fn from_directory(dir: impl AsRef<Path>) -> Result<Self> {
        let mut files = Vec::new();

        let entries = std::fs::read_dir(dir.as_ref()).map_err(|e| Error::WalRecovery {
            reason: format!("Failed to read WAL directory: {e}"),
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "log") {
                files.push(path);
            }
        }

        // Sort by filename for proper ordering
        files.sort();

        Ok(Self {
            files,
            current_reader: None,
            current_index: 0,
            total_entries: 0,
        })
    }

    /// Move to the next file.
    fn next_file(&mut self) -> Option<Result<()>> {
        if self.current_index >= self.files.len() {
            return None;
        }

        let path = &self.files[self.current_index];
        self.current_index += 1;

        match WalReader::open(path) {
            Ok(reader) => {
                self.current_reader = Some(reader);
                Some(Ok(()))
            }
            Err(e) => Some(Err(e)),
        }
    }

    /// Get total entries read across all files.
    #[must_use]
    pub fn total_entries(&self) -> u64 {
        self.total_entries
    }
}

impl Iterator for MultiFileReader {
    type Item = Result<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try current reader first
            if let Some(reader) = &mut self.current_reader {
                if let Some(entry) = reader.next() {
                    if entry.is_ok() {
                        self.total_entries += 1;
                    }
                    return Some(entry);
                }
            }

            // Move to next file
            match self.next_file() {
                Some(Ok(())) => continue,
                Some(Err(e)) => return Some(Err(e)),
                None => return None,
            }
        }
    }
}

/// Replay WAL entries to recover state.
///
/// # Arguments
///
/// * `reader` - Source of WAL entries
/// * `on_insert` - Called for Insert entries
/// * `on_update` - Called for UpdateStats entries
/// * `on_delete` - Called for Delete entries
///
/// # Returns
///
/// The last sequence number seen, or 0 if no entries.
#[allow(dead_code)]
pub fn replay_wal<I, FI, FU, FD>(
    reader: I,
    mut on_insert: FI,
    mut on_update: FU,
    mut on_delete: FD,
) -> Result<ReplayStats>
where
    I: Iterator<Item = Result<WalEntry>>,
    FI: FnMut(&WalEntry) -> Result<()>,
    FU: FnMut(&str, f64) -> Result<()>,
    FD: FnMut(&str) -> Result<()>,
{
    use crate::wal::entry::WalEntryType;

    let mut stats = ReplayStats::default();

    for entry_result in reader {
        let entry = entry_result?;
        stats.last_sequence = stats.last_sequence.max(entry.sequence);

        match entry.entry_type {
            WalEntryType::Insert => {
                on_insert(&entry)?;
                stats.inserts += 1;
            }
            WalEntryType::UpdateStats => {
                if let Some(outcome) = entry.outcome {
                    on_update(&entry.record_id, outcome)?;
                    stats.updates += 1;
                }
            }
            WalEntryType::Delete => {
                on_delete(&entry.record_id)?;
                stats.deletes += 1;
            }
            WalEntryType::Checkpoint => {
                stats.checkpoints += 1;
                stats.last_checkpoint_seq = Some(entry.sequence);
            }
        }

        stats.total_entries += 1;
    }

    Ok(stats)
}

/// Statistics from WAL replay.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ReplayStats {
    /// Total entries processed
    pub total_entries: u64,
    /// Insert operations
    pub inserts: u64,
    /// Update operations
    pub updates: u64,
    /// Delete operations
    pub deletes: u64,
    /// Checkpoint markers
    pub checkpoints: u64,
    /// Last sequence number seen
    pub last_sequence: u64,
    /// Last checkpoint sequence (if any)
    pub last_checkpoint_seq: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::OutcomeStats;
    use crate::types::RecordStatus;
    use crate::wal::{WalConfig, WalWriter};
    use crate::types::MemoryRecord;
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
    fn test_read_written_entries() {
        let temp_dir = TempDir::new().unwrap();

        // Write entries
        {
            let config = WalConfig::new(temp_dir.path());
            let writer = WalWriter::new(config).unwrap();
            writer.log_insert(&create_test_record("rec-1")).unwrap();
            writer.log_insert(&create_test_record("rec-2")).unwrap();
            writer.log_update_stats(&"rec-1".into(), 0.8).unwrap();
            writer.log_delete(&"rec-2".into()).unwrap();
            writer.flush().unwrap();
        }

        // Read entries
        let files = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map_or(false, |e| e == "log"))
            .unwrap();

        let reader = WalReader::open(files).unwrap();
        let entries: Vec<_> = reader.filter_map(|e| e.ok()).collect();

        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].record_id, "rec-1");
        assert_eq!(entries[1].record_id, "rec-2");
        assert_eq!(entries[2].record_id, "rec-1");
        assert_eq!(entries[3].record_id, "rec-2");
    }

    #[test]
    fn test_multi_file_reader() {
        let temp_dir = TempDir::new().unwrap();

        // Write to multiple files
        {
            let config = WalConfig::new(temp_dir.path())
                .with_max_file_size(256); // Small for rotation

            let writer = WalWriter::new(config).unwrap();

            for i in 0..20 {
                writer
                    .log_insert(&create_test_record(&format!("rec-{i}")))
                    .unwrap();
            }
            writer.flush().unwrap();

            let files = writer.list_files().unwrap();
            assert!(files.len() > 1);
        }

        // Read all with multi-file reader
        let reader = MultiFileReader::from_directory(temp_dir.path()).unwrap();
        let entries: Vec<_> = reader.filter_map(|e| e.ok()).collect();

        assert_eq!(entries.len(), 20);

        // Verify sequence is continuous
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.sequence, (i + 1) as u64);
        }
    }

    #[test]
    fn test_replay_wal() {
        let temp_dir = TempDir::new().unwrap();

        // Write entries
        {
            let config = WalConfig::new(temp_dir.path());
            let writer = WalWriter::new(config).unwrap();
            writer.log_insert(&create_test_record("rec-1")).unwrap();
            writer.log_insert(&create_test_record("rec-2")).unwrap();
            writer.log_update_stats(&"rec-1".into(), 0.8).unwrap();
            writer.log_checkpoint().unwrap();
            writer.log_delete(&"rec-2".into()).unwrap();
            writer.flush().unwrap();
        }

        // Replay
        let reader = MultiFileReader::from_directory(temp_dir.path()).unwrap();

        let mut inserts = Vec::new();
        let mut updates = Vec::new();
        let mut deletes = Vec::new();

        let stats = replay_wal(
            reader,
            |e| {
                inserts.push(e.record_id.clone());
                Ok(())
            },
            |id, outcome| {
                updates.push((id.to_string(), outcome));
                Ok(())
            },
            |id| {
                deletes.push(id.to_string());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(inserts.len(), 2);
        assert_eq!(updates.len(), 1);
        assert_eq!(deletes.len(), 1);
        assert_eq!(stats.checkpoints, 1);
        assert_eq!(stats.last_sequence, 5);
        assert_eq!(stats.last_checkpoint_seq, Some(4));
    }
}
