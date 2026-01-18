//! Index Registry
//!
//! Manages multiple named indexes for multi-index retrieval scenarios.
//!
//! # Overview
//!
//! The registry provides a centralized way to manage multiple vector indexes,
//! enabling multi-modal retrieval (e.g., separate indexes for text embeddings,
//! code embeddings, and image embeddings).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    IndexRegistry                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  indexes: HashMap<String, Box<dyn VectorIndex>>              │
//! ├─────────────────────────────────────────────────────────────┤
//! │  + register(name, index)                                     │
//! │  + get(name) -> &dyn VectorIndex                             │
//! │  + get_mut(name) -> &mut dyn VectorIndex                     │
//! │  + remove(name) -> Option<Box<dyn VectorIndex>>              │
//! │  + list() -> Vec<&str>                                       │
//! │  + search_all(query, k) -> MultiIndexResults                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::error::{Error, Result};
use crate::index::traits::{DistanceType, SearchResult, VectorIndex};
use ahash::AHashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Information about a registered index.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Index name
    pub name: String,
    /// Vector dimension
    pub dimension: usize,
    /// Distance metric
    pub distance_type: DistanceType,
    /// Number of vectors
    pub size: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
}

/// Results from a multi-index search.
#[derive(Debug, Clone)]
pub struct MultiIndexResult {
    /// Index name this result came from
    pub index_name: String,
    /// Search results from this index
    pub results: Vec<SearchResult>,
}

/// Results aggregated from multiple indexes.
#[derive(Debug, Clone, Default)]
pub struct MultiIndexResults {
    /// Results per index
    pub by_index: Vec<MultiIndexResult>,
    /// Total results across all indexes
    pub total_count: usize,
}

impl MultiIndexResults {
    /// Create empty results.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add results from an index.
    pub fn add(&mut self, index_name: String, results: Vec<SearchResult>) {
        self.total_count += results.len();
        self.by_index.push(MultiIndexResult {
            index_name,
            results,
        });
    }

    /// Flatten all results into a single vector.
    ///
    /// Note: Results are not re-ranked; use fusion for proper merging.
    #[must_use]
    pub fn flatten(&self) -> Vec<(String, SearchResult)> {
        self.by_index
            .iter()
            .flat_map(|mir| {
                mir.results
                    .iter()
                    .cloned()
                    .map(|r| (mir.index_name.clone(), r))
            })
            .collect()
    }
}

/// Thread-safe registry for managing multiple named indexes.
///
/// # Example
///
/// ```ignore
/// use rag_plusplus_core::index::{IndexRegistry, FlatIndex, IndexConfig};
///
/// let mut registry = IndexRegistry::new();
///
/// // Register indexes for different modalities
/// let text_index = FlatIndex::new(IndexConfig::new(768));
/// let code_index = FlatIndex::new(IndexConfig::new(512));
///
/// registry.register("text_embeddings", text_index)?;
/// registry.register("code_embeddings", code_index)?;
///
/// // Search specific index
/// let results = registry.search("text_embeddings", &query, 10)?;
///
/// // Search all indexes
/// let all_results = registry.search_all(&query, 10)?;
/// ```
#[derive(Debug, Default)]
pub struct IndexRegistry {
    /// Named indexes
    indexes: AHashMap<String, Box<dyn VectorIndex>>,
}

impl IndexRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            indexes: AHashMap::new(),
        }
    }

    /// Create registry with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indexes: AHashMap::with_capacity(capacity),
        }
    }

    /// Register a new index with the given name.
    ///
    /// # Errors
    ///
    /// Returns error if an index with the same name already exists.
    pub fn register<I: VectorIndex + 'static>(
        &mut self,
        name: impl Into<String>,
        index: I,
    ) -> Result<()> {
        let name = name.into();
        if self.indexes.contains_key(&name) {
            return Err(Error::DuplicateIndex { name });
        }
        self.indexes.insert(name, Box::new(index));
        Ok(())
    }

    /// Register or replace an index.
    ///
    /// Returns the previous index if one existed.
    pub fn register_or_replace<I: VectorIndex + 'static>(
        &mut self,
        name: impl Into<String>,
        index: I,
    ) -> Option<Box<dyn VectorIndex>> {
        self.indexes.insert(name.into(), Box::new(index))
    }

    /// Get a reference to an index by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn VectorIndex> {
        self.indexes.get(name).map(AsRef::as_ref)
    }

    /// Remove an index by name.
    ///
    /// Returns the removed index if it existed.
    pub fn remove(&mut self, name: &str) -> Option<Box<dyn VectorIndex>> {
        self.indexes.remove(name)
    }

    /// Check if an index exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.indexes.contains_key(name)
    }

    /// List all registered index names.
    #[must_use]
    pub fn list(&self) -> Vec<&str> {
        self.indexes.keys().map(String::as_str).collect()
    }

    /// Get information about all registered indexes.
    #[must_use]
    pub fn info(&self) -> Vec<IndexInfo> {
        self.indexes
            .iter()
            .map(|(name, index)| IndexInfo {
                name: name.clone(),
                dimension: index.dimension(),
                distance_type: index.distance_type(),
                size: index.len(),
                memory_bytes: index.memory_usage(),
            })
            .collect()
    }

    /// Number of registered indexes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indexes.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// Total number of vectors across all indexes.
    #[must_use]
    pub fn total_vectors(&self) -> usize {
        self.indexes.values().map(|i| i.len()).sum()
    }

    /// Total memory usage across all indexes.
    #[must_use]
    pub fn total_memory(&self) -> usize {
        self.indexes.values().map(|i| i.memory_usage()).sum()
    }

    /// Search a specific index by name.
    ///
    /// # Errors
    ///
    /// Returns error if index doesn't exist or search fails.
    pub fn search(&self, name: &str, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let index = self.indexes.get(name).ok_or_else(|| Error::IndexNotFound {
            name: name.to_string(),
        })?;
        index.search(query, k)
    }

    /// Search all indexes with the same query.
    ///
    /// Note: This is a sequential search. For parallel search, use `parallel_search_all`.
    ///
    /// # Errors
    ///
    /// Returns error if any search fails.
    pub fn search_all(&self, query: &[f32], k: usize) -> Result<MultiIndexResults> {
        let mut results = MultiIndexResults::new();

        for (name, index) in &self.indexes {
            // Skip indexes with incompatible dimensions
            if index.dimension() != query.len() {
                continue;
            }

            let index_results = index.search(query, k)?;
            results.add(name.clone(), index_results);
        }

        Ok(results)
    }

    /// Search multiple specific indexes.
    ///
    /// # Arguments
    ///
    /// * `names` - Index names to search
    /// * `query` - Query vector
    /// * `k` - Number of results per index
    ///
    /// # Errors
    ///
    /// Returns error if any index doesn't exist or search fails.
    pub fn search_indexes(
        &self,
        names: &[&str],
        query: &[f32],
        k: usize,
    ) -> Result<MultiIndexResults> {
        let mut results = MultiIndexResults::new();

        for name in names {
            let index = self.indexes.get(*name).ok_or_else(|| Error::IndexNotFound {
                name: (*name).to_string(),
            })?;

            // Check dimension compatibility
            if index.dimension() != query.len() {
                return Err(Error::DimensionMismatch {
                    expected: index.dimension(),
                    got: query.len(),
                });
            }

            let index_results = index.search(query, k)?;
            results.add((*name).to_string(), index_results);
        }

        Ok(results)
    }

    /// Add a vector to a specific index.
    ///
    /// # Errors
    ///
    /// Returns error if index doesn't exist or add fails.
    pub fn add(&mut self, index_name: &str, id: String, vector: &[f32]) -> Result<()> {
        let index = self
            .indexes
            .get_mut(index_name)
            .ok_or_else(|| Error::IndexNotFound {
                name: index_name.to_string(),
            })?;
        index.add(id, vector)
    }

    /// Remove a vector from a specific index.
    ///
    /// # Errors
    ///
    /// Returns error if index doesn't exist.
    pub fn remove_vector(&mut self, index_name: &str, id: &str) -> Result<bool> {
        let index = self
            .indexes
            .get_mut(index_name)
            .ok_or_else(|| Error::IndexNotFound {
                name: index_name.to_string(),
            })?;
        index.remove(id)
    }

    /// Clear all vectors from all indexes.
    pub fn clear_all(&mut self) {
        for index in self.indexes.values_mut() {
            index.clear();
        }
    }
}

/// Thread-safe shared registry using Arc<RwLock>.
pub type SharedRegistry = Arc<RwLock<IndexRegistry>>;

/// Create a new shared registry.
#[must_use]
pub fn shared_registry() -> SharedRegistry {
    Arc::new(RwLock::new(IndexRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{FlatIndex, IndexConfig};

    fn create_test_index(dim: usize) -> FlatIndex {
        FlatIndex::new(IndexConfig::new(dim))
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = IndexRegistry::new();
        let index = create_test_index(128);

        registry.register("test", index).unwrap();

        assert!(registry.contains("test"));
        assert!(!registry.contains("other"));
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get("test").unwrap();
        assert_eq!(retrieved.dimension(), 128);
    }

    #[test]
    fn test_duplicate_register_error() {
        let mut registry = IndexRegistry::new();

        registry.register("test", create_test_index(128)).unwrap();
        let result = registry.register("test", create_test_index(256));

        assert!(result.is_err());
    }

    #[test]
    fn test_register_or_replace() {
        let mut registry = IndexRegistry::new();

        // First registration
        let old = registry.register_or_replace("test", create_test_index(128));
        assert!(old.is_none());

        // Replace
        let old = registry.register_or_replace("test", create_test_index(256));
        assert!(old.is_some());
        assert_eq!(old.unwrap().dimension(), 128);

        // New index has new dimension
        assert_eq!(registry.get("test").unwrap().dimension(), 256);
    }

    #[test]
    fn test_remove() {
        let mut registry = IndexRegistry::new();
        registry.register("test", create_test_index(128)).unwrap();

        let removed = registry.remove("test");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().dimension(), 128);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_list_and_info() {
        let mut registry = IndexRegistry::new();
        registry.register("a", create_test_index(128)).unwrap();
        registry.register("b", create_test_index(256)).unwrap();

        let names = registry.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));

        let info = registry.info();
        assert_eq!(info.len(), 2);
    }

    #[test]
    fn test_search_specific_index() {
        let mut registry = IndexRegistry::new();
        let mut index = create_test_index(4);

        // Add vectors
        index.add("v1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add("v2".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();

        registry.register("test", index).unwrap();

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = registry.search("test", &query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // Closest to query
    }

    #[test]
    fn test_search_nonexistent_index() {
        let registry = IndexRegistry::new();
        let result = registry.search("nonexistent", &[1.0], 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_search_all() {
        let mut registry = IndexRegistry::new();

        // Create two indexes with same dimension
        let mut index1 = create_test_index(4);
        index1.add("a1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut index2 = create_test_index(4);
        index2.add("b1".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();

        registry.register("index1", index1).unwrap();
        registry.register("index2", index2).unwrap();

        let query = [0.5, 0.5, 0.0, 0.0];
        let results = registry.search_all(&query, 10).unwrap();

        assert_eq!(results.by_index.len(), 2);
        assert_eq!(results.total_count, 2);
    }

    #[test]
    fn test_search_all_skips_incompatible_dimensions() {
        let mut registry = IndexRegistry::new();

        let mut index1 = create_test_index(4);
        index1.add("a1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut index2 = create_test_index(8); // Different dimension
        index2
            .add("b1".to_string(), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        registry.register("index1", index1).unwrap();
        registry.register("index2", index2).unwrap();

        // Query with dimension 4 - should only search index1
        let query = [0.5, 0.5, 0.0, 0.0];
        let results = registry.search_all(&query, 10).unwrap();

        assert_eq!(results.by_index.len(), 1);
        assert_eq!(results.by_index[0].index_name, "index1");
    }

    #[test]
    fn test_search_indexes() {
        let mut registry = IndexRegistry::new();

        let mut index1 = create_test_index(4);
        index1.add("a1".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut index2 = create_test_index(4);
        index2.add("b1".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let mut index3 = create_test_index(4);
        index3.add("c1".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();

        registry.register("idx1", index1).unwrap();
        registry.register("idx2", index2).unwrap();
        registry.register("idx3", index3).unwrap();

        // Only search idx1 and idx2
        let query = [0.5, 0.5, 0.0, 0.0];
        let results = registry
            .search_indexes(&["idx1", "idx2"], &query, 10)
            .unwrap();

        assert_eq!(results.by_index.len(), 2);
        assert_eq!(results.total_count, 2);
    }

    #[test]
    fn test_add_to_index() {
        let mut registry = IndexRegistry::new();
        registry.register("test", create_test_index(4)).unwrap();

        registry
            .add("test", "v1".to_string(), &[1.0, 0.0, 0.0, 0.0])
            .unwrap();

        assert_eq!(registry.get("test").unwrap().len(), 1);
    }

    #[test]
    fn test_multi_index_results_flatten() {
        let mut results = MultiIndexResults::new();

        results.add(
            "idx1".to_string(),
            vec![SearchResult::new("a".to_string(), 0.5, DistanceType::L2)],
        );
        results.add(
            "idx2".to_string(),
            vec![SearchResult::new("b".to_string(), 0.3, DistanceType::L2)],
        );

        let flat = results.flatten();
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].0, "idx1");
        assert_eq!(flat[0].1.id, "a");
        assert_eq!(flat[1].0, "idx2");
        assert_eq!(flat[1].1.id, "b");
    }

    #[test]
    fn test_total_vectors_and_memory() {
        let mut registry = IndexRegistry::new();

        let mut index1 = create_test_index(4);
        index1.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index1.add("b".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let mut index2 = create_test_index(4);
        index2.add("c".to_string(), &[0.0, 0.0, 1.0, 0.0]).unwrap();

        registry.register("idx1", index1).unwrap();
        registry.register("idx2", index2).unwrap();

        assert_eq!(registry.total_vectors(), 3);
        assert!(registry.total_memory() > 0);
    }

    #[test]
    fn test_clear_all() {
        let mut registry = IndexRegistry::new();

        let mut index1 = create_test_index(4);
        index1.add("a".to_string(), &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut index2 = create_test_index(4);
        index2.add("b".to_string(), &[0.0, 1.0, 0.0, 0.0]).unwrap();

        registry.register("idx1", index1).unwrap();
        registry.register("idx2", index2).unwrap();

        assert_eq!(registry.total_vectors(), 2);

        registry.clear_all();

        assert_eq!(registry.total_vectors(), 0);
        assert_eq!(registry.len(), 2); // Indexes still exist, just empty
    }

    #[test]
    fn test_shared_registry() {
        let registry = shared_registry();

        // Write access
        {
            let mut reg = registry.write();
            reg.register("test", create_test_index(128)).unwrap();
        }

        // Read access
        {
            let reg = registry.read();
            assert!(reg.contains("test"));
        }
    }
}
