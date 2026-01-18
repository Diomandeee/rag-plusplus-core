//! Text Tokenization for Sparse Retrieval
//!
//! Provides tokenization utilities for BM25 and keyword matching.

use ahash::AHashSet;

/// Trait for text tokenizers.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text into terms.
    fn tokenize(&self, text: &str) -> Vec<String>;

    /// Tokenize and return unique terms with frequencies.
    fn tokenize_with_freq(&self, text: &str) -> Vec<(String, u32)> {
        let tokens = self.tokenize(text);
        let mut freq_map = ahash::AHashMap::new();

        for token in tokens {
            *freq_map.entry(token).or_insert(0u32) += 1;
        }

        freq_map.into_iter().collect()
    }
}

/// Simple whitespace + punctuation tokenizer with optional lowercasing.
#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    /// Convert to lowercase
    lowercase: bool,
    /// Minimum token length
    min_length: usize,
    /// Maximum token length
    max_length: usize,
    /// Stop words to filter out
    stop_words: AHashSet<String>,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lowercase: true,
            min_length: 1,
            max_length: 100,
            stop_words: default_stop_words(),
        }
    }

    /// Set lowercase option.
    #[must_use]
    pub const fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Set minimum token length.
    #[must_use]
    pub const fn with_min_length(mut self, min: usize) -> Self {
        self.min_length = min;
        self
    }

    /// Set maximum token length.
    #[must_use]
    pub const fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = max;
        self
    }

    /// Set stop words.
    #[must_use]
    pub fn with_stop_words(mut self, stop_words: AHashSet<String>) -> Self {
        self.stop_words = stop_words;
        self
    }

    /// Disable stop word filtering.
    #[must_use]
    pub fn without_stop_words(mut self) -> Self {
        self.stop_words.clear();
        self
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let processed = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        processed
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() >= self.min_length && s.len() <= self.max_length)
            .map(|s| s.to_string())
            .filter(|s| !self.stop_words.contains(s))
            .collect()
    }
}

/// Default English stop words.
fn default_stop_words() -> AHashSet<String> {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "that", "this", "these", "those", "it", "its", "they", "them", "their",
        "he", "she", "him", "her", "his", "we", "us", "our", "you", "your",
        "i", "me", "my", "not", "no", "nor", "so", "if", "then", "than",
        "when", "where", "what", "which", "who", "whom", "how", "why",
        "all", "each", "every", "both", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "just", "also", "very",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_basic() {
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.tokenize("Hello World! This is a test.");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stop words should be filtered
        assert!(!tokens.contains(&"this".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenizer_without_stop_words() {
        let tokenizer = SimpleTokenizer::new().without_stop_words();
        let tokens = tokenizer.tokenize("this is a test");

        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"is".to_string()));
        assert!(tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"test".to_string()));
    }

    #[test]
    fn test_tokenizer_case_sensitive() {
        let tokenizer = SimpleTokenizer::new()
            .with_lowercase(false)
            .without_stop_words();
        let tokens = tokenizer.tokenize("Hello World");

        assert!(tokens.contains(&"Hello".to_string()));
        assert!(tokens.contains(&"World".to_string()));
        assert!(!tokens.contains(&"hello".to_string()));
    }

    #[test]
    fn test_tokenizer_min_length() {
        let tokenizer = SimpleTokenizer::new()
            .with_min_length(3)
            .without_stop_words();
        let tokens = tokenizer.tokenize("a ab abc abcd");

        assert!(!tokens.contains(&"a".to_string()));
        assert!(!tokens.contains(&"ab".to_string()));
        assert!(tokens.contains(&"abc".to_string()));
        assert!(tokens.contains(&"abcd".to_string()));
    }

    #[test]
    fn test_tokenize_with_freq() {
        let tokenizer = SimpleTokenizer::new().without_stop_words();
        let freq = tokenizer.tokenize_with_freq("hello hello world");

        let freq_map: ahash::AHashMap<_, _> = freq.into_iter().collect();
        assert_eq!(freq_map.get("hello"), Some(&2));
        assert_eq!(freq_map.get("world"), Some(&1));
    }

    #[test]
    fn test_punctuation_handling() {
        let tokenizer = SimpleTokenizer::new().without_stop_words();
        let tokens = tokenizer.tokenize("hello, world! test-case foo_bar");

        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"case".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
    }
}
