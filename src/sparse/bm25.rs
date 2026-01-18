//! BM25 Sparse Retrieval Index
//!
//! Implements Okapi BM25 algorithm for keyword-based document retrieval.
//! Designed for hybrid search with dense vector retrieval.

use ahash::AHashMap;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::tokenizer::{SimpleTokenizer, Tokenizer};

/// BM25 configuration parameters.
#[derive(Debug, Clone)]
pub struct BM25Config {
    /// Term frequency saturation parameter (default: 1.2)
    pub k1: f32,
    /// Length normalization parameter (default: 0.75)
    pub b: f32,
    /// Minimum IDF threshold (default: 0.0, disable with negative value)
    pub min_idf: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            min_idf: 0.0,
        }
    }
}

impl BM25Config {
    /// Create config with custom k1.
    #[must_use]
    pub const fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Create config with custom b.
    #[must_use]
    pub const fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }
}

/// Sparse retrieval result.
#[derive(Debug, Clone)]
pub struct SparseResult {
    /// Document ID
    pub id: String,
    /// BM25 score
    pub score: f32,
}

/// Document statistics for BM25.
#[derive(Debug, Clone)]
struct DocStats {
    /// Document ID
    id: String,
    /// Term frequencies
    term_freqs: AHashMap<String, u32>,
    /// Document length (number of tokens)
    length: u32,
}

/// Inverted index entry.
#[derive(Debug, Clone, Default)]
struct PostingList {
    /// Document indices containing this term
    doc_indices: Vec<usize>,
    /// Document frequency (number of documents containing term)
    doc_freq: u32,
}

/// BM25 Index for sparse retrieval.
pub struct BM25Index {
    /// Configuration
    config: BM25Config,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer>,
    /// All documents
    documents: Vec<DocStats>,
    /// ID to index mapping
    id_to_idx: AHashMap<String, usize>,
    /// Inverted index: term -> posting list
    inverted_index: AHashMap<String, PostingList>,
    /// Total number of documents
    num_docs: usize,
    /// Average document length
    avg_doc_length: f32,
    /// Total document length sum (for incremental avg calculation)
    total_doc_length: u64,
}

impl std::fmt::Debug for BM25Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BM25Index")
            .field("config", &self.config)
            .field("num_docs", &self.num_docs)
            .field("num_terms", &self.inverted_index.len())
            .field("avg_doc_length", &self.avg_doc_length)
            .finish_non_exhaustive()
    }
}

impl BM25Index {
    /// Create a new BM25 index with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(BM25Config::default())
    }

    /// Create a new BM25 index with custom configuration.
    #[must_use]
    pub fn with_config(config: BM25Config) -> Self {
        Self {
            config,
            tokenizer: Box::new(SimpleTokenizer::new()),
            documents: Vec::new(),
            id_to_idx: AHashMap::new(),
            inverted_index: AHashMap::new(),
            num_docs: 0,
            avg_doc_length: 0.0,
            total_doc_length: 0,
        }
    }

    /// Set a custom tokenizer.
    pub fn with_tokenizer<T: Tokenizer + 'static>(mut self, tokenizer: T) -> Self {
        self.tokenizer = Box::new(tokenizer);
        self
    }

    /// Add a document to the index.
    ///
    /// # Arguments
    /// * `id` - Unique document identifier
    /// * `text` - Document text content
    ///
    /// # Returns
    /// `true` if document was added, `false` if ID already exists
    pub fn add(&mut self, id: String, text: &str) -> bool {
        if self.id_to_idx.contains_key(&id) {
            return false;
        }

        let term_freqs: AHashMap<String, u32> = self.tokenizer
            .tokenize_with_freq(text)
            .into_iter()
            .collect();

        let doc_length: u32 = term_freqs.values().sum();
        let doc_idx = self.documents.len();

        // Update inverted index
        for term in term_freqs.keys() {
            let posting = self.inverted_index.entry(term.clone()).or_default();
            posting.doc_indices.push(doc_idx);
            posting.doc_freq += 1;
        }

        // Store document stats
        let doc = DocStats {
            id: id.clone(),
            term_freqs,
            length: doc_length,
        };

        self.documents.push(doc);
        self.id_to_idx.insert(id, doc_idx);

        // Update global stats
        self.num_docs += 1;
        self.total_doc_length += doc_length as u64;
        self.avg_doc_length = self.total_doc_length as f32 / self.num_docs as f32;

        true
    }

    /// Add multiple documents.
    pub fn add_batch<I, S>(&mut self, documents: I)
    where
        I: IntoIterator<Item = (S, String)>,
        S: Into<String>,
    {
        for (id, text) in documents {
            self.add(id.into(), &text);
        }
    }

    /// Search for documents matching query.
    ///
    /// # Arguments
    /// * `query` - Search query text
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    /// Top-k documents sorted by BM25 score (descending)
    pub fn search(&self, query: &str, k: usize) -> Vec<SparseResult> {
        if self.num_docs == 0 {
            return vec![];
        }

        let query_terms = self.tokenizer.tokenize(query);
        if query_terms.is_empty() {
            return vec![];
        }

        // Calculate BM25 scores for all documents containing query terms
        let mut scores: AHashMap<usize, f32> = AHashMap::new();

        for term in &query_terms {
            if let Some(posting) = self.inverted_index.get(term) {
                let idf = self.calculate_idf(posting.doc_freq);

                if idf < self.config.min_idf {
                    continue;
                }

                for &doc_idx in &posting.doc_indices {
                    let doc = &self.documents[doc_idx];
                    if let Some(&tf) = doc.term_freqs.get(term) {
                        let tf_component = self.calculate_tf_component(tf, doc.length);
                        let term_score = idf * tf_component;
                        *scores.entry(doc_idx).or_insert(0.0) += term_score;
                    }
                }
            }
        }

        // Get top-k results using a min-heap
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::with_capacity(k + 1);

        for (doc_idx, score) in scores {
            heap.push(Reverse((OrderedFloat(score), doc_idx)));
            if heap.len() > k {
                heap.pop();
            }
        }

        // Convert to sorted results (highest score first)
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|Reverse((score, idx))| SparseResult {
                id: self.documents[idx].id.clone(),
                score: score.0,
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Calculate IDF (Inverse Document Frequency) for a term.
    #[inline]
    fn calculate_idf(&self, doc_freq: u32) -> f32 {
        let n = self.num_docs as f32;
        let df = doc_freq as f32;
        // BM25 IDF formula with smoothing
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Calculate term frequency component of BM25.
    #[inline]
    fn calculate_tf_component(&self, tf: u32, doc_length: u32) -> f32 {
        let tf = tf as f32;
        let dl = doc_length as f32;
        let avgdl = self.avg_doc_length;
        let k1 = self.config.k1;
        let b = self.config.b;

        let length_norm = 1.0 - b + b * (dl / avgdl);
        (tf * (k1 + 1.0)) / (tf + k1 * length_norm)
    }

    /// Get number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.num_docs
    }

    /// Check if index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Get number of unique terms.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.inverted_index.len()
    }

    /// Get average document length.
    #[must_use]
    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    /// Check if document exists.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_idx.contains_key(id)
    }

    /// Clear all documents.
    pub fn clear(&mut self) {
        self.documents.clear();
        self.id_to_idx.clear();
        self.inverted_index.clear();
        self.num_docs = 0;
        self.avg_doc_length = 0.0;
        self.total_doc_length = 0;
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_corpus() -> BM25Index {
        let mut index = BM25Index::new();

        index.add("doc1".to_string(), "the quick brown fox jumps over the lazy dog");
        index.add("doc2".to_string(), "a quick brown dog outpaces the fox");
        index.add("doc3".to_string(), "the lazy dog sleeps all day");
        index.add("doc4".to_string(), "machine learning is transforming the world");
        index.add("doc5".to_string(), "deep learning neural networks are powerful");

        index
    }

    #[test]
    fn test_add_documents() {
        let index = create_test_corpus();

        assert_eq!(index.len(), 5);
        assert!(index.contains("doc1"));
        assert!(!index.contains("nonexistent"));
    }

    #[test]
    fn test_duplicate_add() {
        let mut index = BM25Index::new();

        assert!(index.add("doc1".to_string(), "hello world"));
        assert!(!index.add("doc1".to_string(), "different text"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_basic_search() {
        let index = create_test_corpus();

        let results = index.search("quick brown fox", 3);

        assert!(!results.is_empty());
        assert!(results.len() <= 3);

        // doc1 and doc2 should be top results (contain most query terms)
        let top_ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"doc1") || top_ids.contains(&"doc2"));
    }

    #[test]
    fn test_search_relevance() {
        let index = create_test_corpus();

        // "machine learning" should find doc4
        let results = index.search("machine learning", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc4");

        // "deep learning" should find doc5
        let results = index.search("deep learning neural", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc5");
    }

    #[test]
    fn test_empty_query() {
        let index = create_test_corpus();
        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_matching_terms() {
        let index = create_test_corpus();
        let results = index.search("xyzabc nonexistent terms", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let index = BM25Index::new();
        let results = index.search("any query", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_score_ordering() {
        let index = create_test_corpus();
        let results = index.search("dog", 5);

        // Scores should be in descending order
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_custom_config() {
        let config = BM25Config::default()
            .with_k1(2.0)
            .with_b(0.5);

        let mut index = BM25Index::with_config(config);
        index.add("doc1".to_string(), "hello world hello");
        index.add("doc2".to_string(), "hello");

        let results = index.search("hello", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_num_terms() {
        let index = create_test_corpus();
        // Should have multiple unique terms
        assert!(index.num_terms() > 10);
    }

    #[test]
    fn test_avg_doc_length() {
        let index = create_test_corpus();
        // Average should be positive
        assert!(index.avg_doc_length() > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut index = create_test_corpus();
        assert!(!index.is_empty());

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.num_terms(), 0);
    }

    #[test]
    fn test_long_documents() {
        let mut index = BM25Index::new();

        // Add documents with varying lengths
        index.add("short".to_string(), "hello");
        index.add("medium".to_string(), "hello world this is a test");
        index.add("long".to_string(), "hello world this is a longer test document with many more words to test length normalization");

        let results = index.search("hello", 3);

        // All should match
        assert_eq!(results.len(), 3);
        // Shorter documents should potentially score higher (BM25 length normalization)
    }
}
