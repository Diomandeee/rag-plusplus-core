//! Phase Inference Rules Engine
//!
//! Automatically infers trajectory phases from episode patterns.
//! Per SPECIFICATION_HARD.md Section 8.
//!
//! # Phases
//!
//! | Phase         | Indicators                                          |
//! |---------------|-----------------------------------------------------|
//! | Exploration   | Questions, short turns, topic shifts                |
//! | Consolidation | Long responses, code blocks, file references        |
//! | Synthesis     | "Decision:", "Let's go with", summaries             |
//! | Debugging     | "Traceback", "Exception", "doesn't work", traces    |
//! | Planning      | "Phase", "roadmap", "steps", numbered lists         |
//!
//! # Performance
//!
//! This module uses compiled regex patterns and efficient string matching
//! for high-throughput inference over large corpora.

use std::collections::HashMap;

/// Trajectory phase as defined in specification.
///
/// Represents the qualitative state within a trajectory:
/// - **Exploration**: Initial inquiry, questions, topic discovery
/// - **Consolidation**: Building understanding, code implementation
/// - **Synthesis**: Summarization, decisions, conclusions
/// - **Debugging**: Error resolution, troubleshooting
/// - **Planning**: Roadmaps, structured plans, task organization
///
/// # Example
///
/// ```
/// use rag_plusplus_core::trajectory::TrajectoryPhase;
///
/// let phase = TrajectoryPhase::from_str("exploration").unwrap();
/// assert_eq!(phase.as_str(), "exploration");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrajectoryPhase {
    /// Initial inquiry, questions, topic discovery
    Exploration,
    /// Building understanding, code implementation
    Consolidation,
    /// Summarization, decisions, conclusions
    Synthesis,
    /// Error resolution, troubleshooting
    Debugging,
    /// Roadmaps, structured plans, task organization
    Planning,
}

/// Backward-compatible type alias.
#[deprecated(since = "0.2.0", note = "Use TrajectoryPhase instead")]
pub type ConversationPhase = TrajectoryPhase;

impl TrajectoryPhase {
    /// Convert to database-compatible string.
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Exploration => "exploration",
            Self::Consolidation => "consolidation",
            Self::Synthesis => "synthesis",
            Self::Debugging => "debugging",
            Self::Planning => "planning",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "exploration" => Some(Self::Exploration),
            "consolidation" => Some(Self::Consolidation),
            "synthesis" => Some(Self::Synthesis),
            "debugging" => Some(Self::Debugging),
            "planning" => Some(Self::Planning),
            _ => None,
        }
    }
}

/// Features extracted from an episode for phase inference.
#[derive(Debug, Clone, Default)]
pub struct TurnFeatures {
    /// Episode identifier
    pub turn_id: u64,
    /// Role: "user" or "assistant"
    pub role: String,
    /// Content length in characters
    pub content_length: usize,
    /// Number of question marks
    pub question_count: usize,
    /// Number of code blocks (```)
    pub code_block_count: usize,
    /// Number of list items (numbered or bullet)
    pub list_item_count: usize,
    /// Contains error-related keywords
    pub has_error_keywords: bool,
    /// Contains decision keywords
    pub has_decision_keywords: bool,
    /// Contains planning keywords
    pub has_planning_keywords: bool,
    /// Contains file references (.rs, .py, etc.)
    pub has_file_references: bool,
    /// Word count
    pub word_count: usize,
    /// Average sentence length
    pub avg_sentence_length: f32,
}

impl TurnFeatures {
    /// Extract features from raw content.
    pub fn from_content(turn_id: u64, role: &str, content: &str) -> Self {
        let content_lower = content.to_lowercase();

        // Count questions
        let question_count = content.chars().filter(|&c| c == '?').count();

        // Count code blocks
        let code_block_count = content.matches("```").count() / 2;

        // Count list items (lines starting with - * or number.)
        let list_item_count = content.lines()
            .filter(|line| {
                let trimmed = line.trim();
                trimmed.starts_with("- ") ||
                trimmed.starts_with("* ") ||
                trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) &&
                    trimmed.chars().nth(1).map_or(false, |c| c == '.' || c == ')')
            })
            .count();

        // Error keywords
        let error_keywords = [
            "error", "exception", "traceback", "failed", "doesn't work",
            "bug", "fix", "broken", "crash", "panic", "undefined",
        ];
        let has_error_keywords = error_keywords.iter().any(|k| content_lower.contains(k));

        // Decision keywords
        let decision_keywords = [
            "decision:", "let's go with", "we'll use", "decided to",
            "the approach", "final answer", "conclusion", "summary",
        ];
        let has_decision_keywords = decision_keywords.iter().any(|k| content_lower.contains(k));

        // Planning keywords
        let planning_keywords = [
            "phase", "roadmap", "step", "plan", "milestone", "todo",
            "first,", "second,", "third,", "then", "next step",
        ];
        let has_planning_keywords = planning_keywords.iter().any(|k| content_lower.contains(k));

        // File references
        let file_extensions = [".rs", ".py", ".ts", ".js", ".go", ".java", ".cpp", ".h"];
        let has_file_references = file_extensions.iter().any(|ext| content.contains(ext));

        // Word count
        let word_count = content.split_whitespace().count();

        // Average sentence length
        let sentences: Vec<&str> = content.split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
        let avg_sentence_length = if sentences.is_empty() {
            0.0
        } else {
            sentences.iter().map(|s| s.split_whitespace().count()).sum::<usize>() as f32
                / sentences.len() as f32
        };

        Self {
            turn_id,
            role: role.to_string(),
            content_length: content.len(),
            question_count,
            code_block_count,
            list_item_count,
            has_error_keywords,
            has_decision_keywords,
            has_planning_keywords,
            has_file_references,
            word_count,
            avg_sentence_length,
        }
    }
}

/// Phase transition detection result.
///
/// Captures when the trajectory shifts from one phase to another,
/// which is often a high-salience moment in the trajectory.
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Episode identifier where transition occurs
    pub turn_id: u64,
    /// Previous phase (None if this is the first episode)
    pub from_phase: Option<TrajectoryPhase>,
    /// New phase after transition
    pub to_phase: TrajectoryPhase,
    /// Confidence score [0, 1] for the transition detection
    pub confidence: f32,
}

/// Configuration for phase inference.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    /// Minimum confidence to assign a phase
    pub min_confidence: f32,
    /// Short turn threshold (characters)
    pub short_turn_threshold: usize,
    /// Long turn threshold (characters)
    pub long_turn_threshold: usize,
    /// High question density threshold
    pub high_question_threshold: usize,
    /// Version string for tracking
    pub version: String,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            short_turn_threshold: 200,
            long_turn_threshold: 1000,
            high_question_threshold: 2,
            version: "v1.0".to_string(),
        }
    }
}

/// Phase inference engine.
#[derive(Debug, Clone)]
pub struct PhaseInferencer {
    config: PhaseConfig,
}

impl PhaseInferencer {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: PhaseConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: PhaseConfig) -> Self {
        Self { config }
    }

    /// Infer phase for a single episode.
    ///
    /// Returns (phase, confidence) or None if no phase can be determined.
    pub fn infer_single(&self, features: &TurnFeatures) -> Option<(TrajectoryPhase, f32)> {
        let mut scores: HashMap<TrajectoryPhase, f32> = HashMap::new();

        // Debugging: error keywords are strong signal
        if features.has_error_keywords {
            *scores.entry(TrajectoryPhase::Debugging).or_default() += 0.6;
        }

        // Planning: planning keywords + lists
        if features.has_planning_keywords {
            *scores.entry(TrajectoryPhase::Planning).or_default() += 0.4;
        }
        if features.list_item_count > 3 {
            *scores.entry(TrajectoryPhase::Planning).or_default() += 0.3;
        }

        // Synthesis: decision keywords
        if features.has_decision_keywords {
            *scores.entry(TrajectoryPhase::Synthesis).or_default() += 0.5;
        }

        // Consolidation: long responses with code
        if features.content_length > self.config.long_turn_threshold {
            *scores.entry(TrajectoryPhase::Consolidation).or_default() += 0.3;
        }
        if features.code_block_count > 0 {
            *scores.entry(TrajectoryPhase::Consolidation).or_default() += 0.3;
        }
        if features.has_file_references {
            *scores.entry(TrajectoryPhase::Consolidation).or_default() += 0.2;
        }

        // Exploration: questions, short turns (especially user)
        if features.role == "user" {
            if features.question_count >= self.config.high_question_threshold {
                *scores.entry(TrajectoryPhase::Exploration).or_default() += 0.5;
            }
            if features.content_length < self.config.short_turn_threshold {
                *scores.entry(TrajectoryPhase::Exploration).or_default() += 0.2;
            }
        }

        // Find highest scoring phase
        scores.into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .filter(|(_, score)| *score >= self.config.min_confidence)
    }

    /// Infer phases for a sequence of episodes.
    ///
    /// Uses context from surrounding episodes for better accuracy.
    pub fn infer_sequence(&self, turns: &[TurnFeatures]) -> Vec<(u64, TrajectoryPhase, f32)> {
        let mut results = Vec::with_capacity(turns.len());

        for (i, features) in turns.iter().enumerate() {
            // Get context from window
            let window_start = i.saturating_sub(2);
            let window_end = (i + 3).min(turns.len());
            let window = &turns[window_start..window_end];

            // Compute window-level signals
            let window_questions: usize = window.iter().map(|t| t.question_count).sum();
            let window_errors = window.iter().filter(|t| t.has_error_keywords).count();
            let window_code: usize = window.iter().map(|t| t.code_block_count).sum();

            // Start with single-turn inference
            let (mut phase, mut confidence) = self.infer_single(features)
                .unwrap_or((TrajectoryPhase::Exploration, 0.3));

            // Adjust based on window context
            if window_errors > 1 {
                if phase != TrajectoryPhase::Debugging {
                    phase = TrajectoryPhase::Debugging;
                    confidence = 0.5;
                } else {
                    confidence += 0.1;
                }
            }

            // High question density suggests exploration
            if window_questions > 3 && phase != TrajectoryPhase::Debugging {
                phase = TrajectoryPhase::Exploration;
                confidence = 0.5;
            }

            if window_code > 2 && phase == TrajectoryPhase::Exploration {
                phase = TrajectoryPhase::Consolidation;
                confidence = 0.4;
            }

            results.push((features.turn_id, phase, confidence.min(1.0)));
        }

        results
    }

    /// Detect phase transitions in a sequence.
    pub fn detect_transitions(&self, turns: &[TurnFeatures]) -> Vec<PhaseTransition> {
        let phases = self.infer_sequence(turns);
        let mut transitions = Vec::new();

        let mut prev_phase: Option<TrajectoryPhase> = None;
        for (turn_id, phase, confidence) in phases {
            if prev_phase != Some(phase) {
                transitions.push(PhaseTransition {
                    turn_id,
                    from_phase: prev_phase,
                    to_phase: phase,
                    confidence,
                });
            }
            prev_phase = Some(phase);
        }

        transitions
    }

    /// Get the version string for this inferencer.
    #[inline]
    pub fn version(&self) -> &str {
        &self.config.version
    }
}

impl Default for PhaseInferencer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_from_str() {
        assert_eq!(TrajectoryPhase::from_str("exploration"), Some(TrajectoryPhase::Exploration));
        assert_eq!(TrajectoryPhase::from_str("DEBUGGING"), Some(TrajectoryPhase::Debugging));
        assert_eq!(TrajectoryPhase::from_str("unknown"), None);
    }

    #[test]
    fn test_feature_extraction() {
        let content = "What is the error? Can you fix it?\n\n```python\nprint('hello')\n```";
        let features = TurnFeatures::from_content(1, "user", content);

        assert_eq!(features.question_count, 2);
        assert_eq!(features.code_block_count, 1);
        assert!(features.has_error_keywords);
    }

    #[test]
    fn test_exploration_inference() {
        let inferencer = PhaseInferencer::new();
        let features = TurnFeatures::from_content(1, "user", "What is this? How does it work?");

        let (phase, _) = inferencer.infer_single(&features).unwrap();
        assert_eq!(phase, TrajectoryPhase::Exploration);
    }

    #[test]
    fn test_debugging_inference() {
        let inferencer = PhaseInferencer::new();
        let features = TurnFeatures::from_content(1, "user", "I'm getting this error: Traceback... Exception raised");

        let (phase, _) = inferencer.infer_single(&features).unwrap();
        assert_eq!(phase, TrajectoryPhase::Debugging);
    }

    #[test]
    fn test_planning_inference() {
        let inferencer = PhaseInferencer::new();
        let content = r#"
Here's the roadmap:
1. First, implement the parser
2. Second, add validation
3. Third, write tests
4. Finally, deploy
"#;
        let features = TurnFeatures::from_content(1, "assistant", content);

        let (phase, _) = inferencer.infer_single(&features).unwrap();
        assert_eq!(phase, TrajectoryPhase::Planning);
    }

    #[test]
    fn test_sequence_inference() {
        let inferencer = PhaseInferencer::new();
        let turns = vec![
            TurnFeatures::from_content(1, "user", "What does this code do?"),
            TurnFeatures::from_content(2, "assistant", "This code implements a parser..."),
            TurnFeatures::from_content(3, "user", "I get this error: Exception"),
            TurnFeatures::from_content(4, "assistant", "The error is caused by..."),
        ];

        let results = inferencer.infer_sequence(&turns);
        assert_eq!(results.len(), 4);

        // Should detect exploration -> debugging transition
        let transitions = inferencer.detect_transitions(&turns);
        assert!(!transitions.is_empty());
    }
}
