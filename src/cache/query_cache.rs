//! Query Cache Implementation
//!
//! LRU cache with TTL expiration for query results.

use crate::retrieval::engine::QueryResponse;
use ahash::AHashMap;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Time-to-live for entries
    pub ttl: Duration,
    /// Whether to cache queries with filters
    pub cache_filtered: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: Duration::from_secs(300), // 5 minutes
            cache_filtered: true,
        }
    }
}

impl CacheConfig {
    /// Create new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max entries.
    #[must_use]
    pub const fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set TTL.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Set whether to cache filtered queries.
    #[must_use]
    pub const fn with_cache_filtered(mut self, cache: bool) -> Self {
        self.cache_filtered = cache;
        self
    }
}

/// Cache key for queries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Hash of embedding vector
    embedding_hash: u64,
    /// Number of results
    k: usize,
    /// Hash of filter (0 if no filter)
    filter_hash: u64,
    /// Hash of index names (0 if all indexes)
    indexes_hash: u64,
}

impl CacheKey {
    /// Create a cache key from query parameters.
    #[must_use]
    pub fn new(
        embedding: &[f32],
        k: usize,
        filter_hash: Option<u64>,
        indexes: Option<&[String]>,
    ) -> Self {
        Self {
            embedding_hash: Self::hash_embedding(embedding),
            k,
            filter_hash: filter_hash.unwrap_or(0),
            indexes_hash: indexes.map(Self::hash_indexes).unwrap_or(0),
        }
    }

    /// Hash an embedding vector.
    fn hash_embedding(embedding: &[f32]) -> u64 {
        let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);

        for &value in embedding {
            hasher.write(&value.to_le_bytes());
        }

        hasher.finish()
    }

    /// Hash index names.
    fn hash_indexes(indexes: &[String]) -> u64 {
        let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);

        for name in indexes {
            hasher.write(name.as_bytes());
        }

        hasher.finish()
    }
}

/// Cached entry with metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cached response
    pub response: QueryResponse,
    /// When the entry was created
    pub created_at: Instant,
    /// Number of times this entry was accessed
    pub access_count: u64,
}

impl CacheEntry {
    /// Check if entry is expired.
    #[must_use]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of entries currently in cache
    pub entries: usize,
    /// Number of evictions
    pub evictions: u64,
    /// Number of expired entries removed
    pub expirations: u64,
}

impl CacheStats {
    /// Calculate hit ratio.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// LRU query cache with TTL expiration.
///
/// Thread-safe cache for query results.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::cache::{QueryCache, CacheConfig, CacheKey};
///
/// let cache = QueryCache::new(CacheConfig::default());
///
/// let key = CacheKey::new(&embedding, 10, None, None);
///
/// // Try cache first
/// if let Some(response) = cache.get(&key) {
///     return response;
/// }
///
/// // Execute query
/// let response = engine.query(request)?;
///
/// // Cache result
/// cache.put(key, response.clone());
/// ```
pub struct QueryCache {
    config: CacheConfig,
    /// Cache entries
    entries: RwLock<AHashMap<CacheKey, CacheEntry>>,
    /// LRU order (front = oldest)
    order: RwLock<VecDeque<CacheKey>>,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
}

impl std::fmt::Debug for QueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCache")
            .field("config", &self.config)
            .field("entries", &self.entries.read().len())
            .finish()
    }
}

impl QueryCache {
    /// Create a new cache.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(AHashMap::new()),
            order: RwLock::new(VecDeque::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
        }
    }

    /// Create with default config.
    #[must_use]
    pub fn default_cache() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get a cached response.
    ///
    /// Returns `Some(response)` if found and not expired, `None` otherwise.
    pub fn get(&self, key: &CacheKey) -> Option<QueryResponse> {
        // Check for entry
        let entries = self.entries.read();

        if let Some(entry) = entries.get(key) {
            // Check expiration
            if entry.is_expired(self.config.ttl) {
                drop(entries);
                self.remove(key);
                self.expirations.fetch_add(1, Ordering::Relaxed);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            self.hits.fetch_add(1, Ordering::Relaxed);

            // Move to back of LRU (update access)
            drop(entries);
            self.touch(key);

            // Re-read after touch
            let entries = self.entries.read();
            entries.get(key).map(|e| e.response.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Put a response in the cache.
    pub fn put(&self, key: CacheKey, response: QueryResponse) {
        // Evict if necessary
        self.maybe_evict();

        let entry = CacheEntry {
            response,
            created_at: Instant::now(),
            access_count: 1,
        };

        {
            let mut entries = self.entries.write();
            let mut order = self.order.write();

            // Remove old entry if exists
            if entries.contains_key(&key) {
                order.retain(|k| k != &key);
            }

            entries.insert(key.clone(), entry);
            order.push_back(key);
        }
    }

    /// Remove an entry.
    pub fn remove(&self, key: &CacheKey) -> Option<CacheEntry> {
        let mut entries = self.entries.write();
        let mut order = self.order.write();

        order.retain(|k| k != key);
        entries.remove(key)
    }

    /// Touch an entry (move to back of LRU).
    fn touch(&self, key: &CacheKey) {
        let mut order = self.order.write();

        // Remove from current position
        order.retain(|k| k != key);
        // Add to back
        order.push_back(key.clone());
    }

    /// Evict oldest entries if over capacity.
    fn maybe_evict(&self) {
        let entries = self.entries.read();
        let current_size = entries.len();
        drop(entries);

        if current_size >= self.config.max_entries {
            // Evict 10% of entries
            let to_evict = self.config.max_entries / 10;
            self.evict_oldest(to_evict.max(1));
        }
    }

    /// Evict the n oldest entries.
    fn evict_oldest(&self, n: usize) {
        let mut entries = self.entries.write();
        let mut order = self.order.write();

        for _ in 0..n {
            if let Some(key) = order.pop_front() {
                entries.remove(&key);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        let mut order = self.order.write();

        entries.clear();
        order.clear();
    }

    /// Remove expired entries.
    pub fn cleanup_expired(&self) {
        let entries_snapshot: Vec<CacheKey> = {
            let entries = self.entries.read();
            entries
                .iter()
                .filter(|(_, entry)| entry.is_expired(self.config.ttl))
                .map(|(key, _)| key.clone())
                .collect()
        };

        for key in entries_snapshot {
            self.remove(&key);
            self.expirations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: self.entries.read().len(),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
        }
    }

    /// Get current size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::default_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::engine::RetrievedRecord;
    use crate::stats::OutcomeStats;
    use crate::types::{MemoryRecord, RecordStatus};

    fn create_test_response(result_count: usize) -> QueryResponse {
        let results: Vec<RetrievedRecord> = (0..result_count)
            .map(|i| RetrievedRecord {
                record: MemoryRecord {
                    id: format!("rec-{i}").into(),
                    embedding: vec![1.0],
                    context: format!("ctx-{i}"),
                    outcome: 0.5,
                    metadata: Default::default(),
                    created_at: 0,
                    status: RecordStatus::Active,
                    stats: OutcomeStats::new(1),
                },
                score: 0.9 - (i as f32 * 0.1),
                rank: i + 1,
                source_index: "test".into(),
            })
            .collect();

        QueryResponse {
            results,
            priors: None,
            latency: Duration::from_millis(10),
            indexes_searched: 1,
            candidates_considered: result_count,
        }
    }

    #[test]
    fn test_cache_key() {
        let key1 = CacheKey::new(&[1.0, 2.0, 3.0], 10, None, None);
        let key2 = CacheKey::new(&[1.0, 2.0, 3.0], 10, None, None);
        let key3 = CacheKey::new(&[1.0, 2.0, 4.0], 10, None, None);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_put_and_get() {
        let cache = QueryCache::default_cache();
        let key = CacheKey::new(&[1.0, 2.0], 5, None, None);
        let response = create_test_response(5);

        cache.put(key.clone(), response);

        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().results.len(), 5);
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryCache::default_cache();
        let key = CacheKey::new(&[1.0, 2.0], 5, None, None);

        let cached = cache.get(&key);
        assert!(cached.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_cache_hit() {
        let cache = QueryCache::default_cache();
        let key = CacheKey::new(&[1.0, 2.0], 5, None, None);

        cache.put(key.clone(), create_test_response(5));
        cache.get(&key);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_ttl_expiration() {
        let config = CacheConfig::new().with_ttl(Duration::from_millis(50));
        let cache = QueryCache::new(config);

        let key = CacheKey::new(&[1.0], 5, None, None);
        cache.put(key.clone(), create_test_response(5));

        // Should hit
        assert!(cache.get(&key).is_some());

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(60));

        // Should miss (expired)
        assert!(cache.get(&key).is_none());

        let stats = cache.stats();
        assert_eq!(stats.expirations, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig::new().with_max_entries(5);
        let cache = QueryCache::new(config);

        // Fill cache
        for i in 0..5 {
            let key = CacheKey::new(&[i as f32], 1, None, None);
            cache.put(key, create_test_response(1));
        }

        assert_eq!(cache.len(), 5);

        // Add one more (should trigger eviction)
        let key = CacheKey::new(&[100.0], 1, None, None);
        cache.put(key, create_test_response(1));

        // Cache should not exceed max
        assert!(cache.len() <= 5);
    }

    #[test]
    fn test_clear() {
        let cache = QueryCache::default_cache();

        for i in 0..10 {
            let key = CacheKey::new(&[i as f32], 1, None, None);
            cache.put(key, create_test_response(1));
        }

        assert_eq!(cache.len(), 10);

        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_ratio() {
        let cache = QueryCache::default_cache();
        let key = CacheKey::new(&[1.0], 5, None, None);

        cache.put(key.clone(), create_test_response(5));

        // 3 hits
        cache.get(&key);
        cache.get(&key);
        cache.get(&key);

        // 1 miss
        cache.get(&CacheKey::new(&[999.0], 5, None, None));

        let stats = cache.stats();
        assert!((stats.hit_ratio() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_remove() {
        let cache = QueryCache::default_cache();
        let key = CacheKey::new(&[1.0], 5, None, None);

        cache.put(key.clone(), create_test_response(5));
        assert!(cache.get(&key).is_some());

        cache.remove(&key);
        assert!(cache.get(&key).is_none());
    }
}
