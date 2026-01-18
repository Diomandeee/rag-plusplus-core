//! Trajectory Coordinate System
//!
//! 4D and 5D positioning systems for episodes in a trajectory, inspired by IRCP/RCP
//! DLM coordinates and TPO (Topological Preference Optimization).
//!
//! # Coordinate Dimensions
//!
//! ## 4D Coordinates (IRCP/RCP)
//!
//! | Dimension | Type | Meaning |
//! |-----------|------|---------|
//! | depth | u32 | Distance from root in trajectory tree (0 = root) |
//! | sibling_order | u32 | Position among siblings at branching points |
//! | homogeneity | f32 | Semantic similarity to parent [0, 1] |
//! | temporal | f32 | Normalized timestamp within trajectory [0, 1] |
//!
//! ## 5D Coordinates (TPO Extension)
//!
//! | Dimension | Type | Meaning |
//! |-----------|------|---------|
//! | depth | u32 | Distance from root in trajectory tree (0 = root) |
//! | sibling_order | u32 | Position among siblings at branching points |
//! | homogeneity | f32 | Semantic similarity to parent [0, 1] |
//! | temporal | f32 | Normalized timestamp within trajectory [0, 1] |
//! | complexity | u32 | Message complexity (n_parts: text + images + code blocks) |
//!
//! # Usage
//!
//! ```
//! use rag_plusplus_core::trajectory::{TrajectoryCoordinate, TrajectoryCoordinate5D};
//!
//! // 4D coordinates (IRCP/RCP)
//! let coord_a = TrajectoryCoordinate::new(3, 0, 0.85, 0.5);
//! let coord_b = TrajectoryCoordinate::new(4, 1, 0.72, 0.6);
//! let distance = coord_a.distance(&coord_b);
//!
//! // 5D coordinates (TPO extension with complexity)
//! let coord_5d_a = TrajectoryCoordinate5D::new(3, 0, 0.85, 0.5, 1);
//! let coord_5d_b = TrajectoryCoordinate5D::new(4, 1, 0.72, 0.6, 3); // multimodal
//! let distance_5d = coord_5d_a.distance(&coord_5d_b);
//! ```

use crate::distance::cosine_similarity_fast;

// ============================================================================
// DLM Weight Configuration (from DLM package analysis)
// ============================================================================

/// DLM coordinate weight configuration.
///
/// These weights control how each dimension contributes to distance calculations.
/// Default values are taken from the DLM package configuration.
///
/// # DLM Default Weights
///
/// | Dimension | Weight | Rationale |
/// |-----------|--------|-----------|
/// | depth | 0.25 | Structural position in tree |
/// | sibling | 0.15 | Branch selection importance |
/// | homogeneity | 0.30 | Semantic continuity (highest weight) |
/// | temporal | 0.20 | Time-based proximity |
/// | complexity | 0.10 | Information density |
///
/// # Example
///
/// ```
/// use rag_plusplus_core::trajectory::DLMWeights;
///
/// // Use default weights
/// let default = DLMWeights::default();
///
/// // Custom weights emphasizing semantic similarity
/// let semantic = DLMWeights::semantic_focused();
///
/// // Fully custom weights
/// let custom = DLMWeights::new(0.2, 0.2, 0.2, 0.2, 0.2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DLMWeights {
    /// Weight for depth dimension
    pub depth: f32,
    /// Weight for sibling_order dimension
    pub sibling: f32,
    /// Weight for homogeneity dimension
    pub homogeneity: f32,
    /// Weight for temporal dimension
    pub temporal: f32,
    /// Weight for complexity dimension
    pub complexity: f32,
}

impl DLMWeights {
    /// Create custom weights.
    ///
    /// Weights are automatically normalized to sum to 1.0.
    #[inline]
    pub fn new(depth: f32, sibling: f32, homogeneity: f32, temporal: f32, complexity: f32) -> Self {
        let total = depth + sibling + homogeneity + temporal + complexity;
        if total > 0.0 {
            Self {
                depth: depth / total,
                sibling: sibling / total,
                homogeneity: homogeneity / total,
                temporal: temporal / total,
                complexity: complexity / total,
            }
        } else {
            Self::default()
        }
    }

    /// Create weights from array [depth, sibling, homogeneity, temporal, complexity].
    #[inline]
    pub fn from_array(weights: [f32; 5]) -> Self {
        Self::new(weights[0], weights[1], weights[2], weights[3], weights[4])
    }

    /// Convert to array [depth, sibling, homogeneity, temporal, complexity].
    #[inline]
    pub fn to_array(&self) -> [f32; 5] {
        [self.depth, self.sibling, self.homogeneity, self.temporal, self.complexity]
    }

    /// Semantic-focused weights (emphasize homogeneity).
    ///
    /// Useful when semantic similarity is most important.
    #[inline]
    pub fn semantic_focused() -> Self {
        Self::new(0.15, 0.10, 0.50, 0.15, 0.10)
    }

    /// Structural-focused weights (emphasize depth and sibling).
    ///
    /// Useful when tree position is most important.
    #[inline]
    pub fn structural_focused() -> Self {
        Self::new(0.35, 0.25, 0.15, 0.15, 0.10)
    }

    /// Temporal-focused weights (emphasize temporal).
    ///
    /// Useful for time-based retrieval.
    #[inline]
    pub fn temporal_focused() -> Self {
        Self::new(0.15, 0.10, 0.20, 0.45, 0.10)
    }

    /// Equal weights for all dimensions.
    #[inline]
    pub fn equal() -> Self {
        Self::new(0.2, 0.2, 0.2, 0.2, 0.2)
    }
}

impl Default for DLMWeights {
    /// DLM default weights from configuration.
    fn default() -> Self {
        Self {
            depth: 0.25,
            sibling: 0.15,
            homogeneity: 0.30,
            temporal: 0.20,
            complexity: 0.10,
        }
    }
}

/// 4D coordinate for positioning an episode within a trajectory.
///
/// Captures both structural (depth, sibling) and semantic (homogeneity, temporal)
/// dimensions of an episode's position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryCoordinate {
    /// Depth in the trajectory tree (0 = root)
    pub depth: u32,
    /// Position among siblings at this level (0-indexed)
    pub sibling_order: u32,
    /// Semantic similarity to parent episode [0, 1]
    /// Computed as cosine similarity of embeddings
    pub homogeneity: f32,
    /// Normalized temporal position within trajectory [0, 1]
    /// 0 = start of trajectory, 1 = end
    pub temporal: f32,
}

impl TrajectoryCoordinate {
    /// Create a new trajectory coordinate.
    ///
    /// # Arguments
    ///
    /// * `depth` - Distance from root (0 = root)
    /// * `sibling_order` - Position among siblings (0-indexed)
    /// * `homogeneity` - Semantic similarity to parent [0, 1]
    /// * `temporal` - Normalized timestamp [0, 1]
    ///
    /// # Panics
    ///
    /// Panics if homogeneity or temporal are outside [0, 1].
    #[inline]
    pub fn new(depth: u32, sibling_order: u32, homogeneity: f32, temporal: f32) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&homogeneity),
            "homogeneity must be in [0, 1], got {}",
            homogeneity
        );
        debug_assert!(
            (0.0..=1.0).contains(&temporal),
            "temporal must be in [0, 1], got {}",
            temporal
        );

        Self {
            depth,
            sibling_order,
            homogeneity,
            temporal,
        }
    }

    /// Create a root coordinate (depth=0, sibling=0, homogeneity=1, temporal=0).
    #[inline]
    pub fn root() -> Self {
        Self {
            depth: 0,
            sibling_order: 0,
            homogeneity: 1.0,
            temporal: 0.0,
        }
    }

    /// Compute Euclidean distance between two coordinates.
    ///
    /// Uses normalized dimensions so each contributes equally.
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let d_depth = (self.depth as f32 - other.depth as f32).abs();
        let d_sibling = (self.sibling_order as f32 - other.sibling_order as f32).abs();
        let d_homo = self.homogeneity - other.homogeneity;
        let d_time = self.temporal - other.temporal;

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2)).sqrt()
    }

    /// Compute weighted distance with custom dimension weights.
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    /// * `weights` - (depth_weight, sibling_weight, homogeneity_weight, temporal_weight)
    #[inline]
    pub fn weighted_distance(&self, other: &Self, weights: (f32, f32, f32, f32)) -> f32 {
        let d_depth = (self.depth as f32 - other.depth as f32).abs() * weights.0;
        let d_sibling = (self.sibling_order as f32 - other.sibling_order as f32).abs() * weights.1;
        let d_homo = (self.homogeneity - other.homogeneity).abs() * weights.2;
        let d_time = (self.temporal - other.temporal).abs() * weights.3;

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2)).sqrt()
    }

    /// Compute Manhattan distance between two coordinates.
    #[inline]
    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        let d_depth = (self.depth as i64 - other.depth as i64).unsigned_abs() as f32;
        let d_sibling = (self.sibling_order as i64 - other.sibling_order as i64).unsigned_abs() as f32;
        let d_homo = (self.homogeneity - other.homogeneity).abs();
        let d_time = (self.temporal - other.temporal).abs();

        d_depth + d_sibling + d_homo + d_time
    }

    /// Compute coordinate from episode data.
    ///
    /// # Arguments
    ///
    /// * `depth` - Tree depth
    /// * `sibling_order` - Position among siblings
    /// * `embedding` - Episode embedding
    /// * `parent_embedding` - Parent episode embedding (for homogeneity)
    /// * `timestamp` - Episode timestamp
    /// * `trajectory_start` - First timestamp in trajectory
    /// * `trajectory_end` - Last timestamp in trajectory
    pub fn from_episode(
        depth: u32,
        sibling_order: u32,
        embedding: &[f32],
        parent_embedding: Option<&[f32]>,
        timestamp: i64,
        trajectory_start: i64,
        trajectory_end: i64,
    ) -> Self {
        // Compute homogeneity as cosine similarity to parent
        let homogeneity = parent_embedding
            .map(|parent| cosine_similarity_fast(embedding, parent).clamp(0.0, 1.0))
            .unwrap_or(1.0); // Root has homogeneity 1.0

        // Compute normalized temporal position
        let temporal = if trajectory_end > trajectory_start {
            let duration = (trajectory_end - trajectory_start) as f64;
            let elapsed = (timestamp - trajectory_start) as f64;
            (elapsed / duration).clamp(0.0, 1.0) as f32
        } else {
            0.0 // Single episode trajectory
        };

        Self {
            depth,
            sibling_order,
            homogeneity,
            temporal,
        }
    }

    /// Normalize depth relative to a maximum depth.
    ///
    /// Useful for comparing coordinates across trajectories of different depths.
    #[inline]
    pub fn normalize_depth(&self, max_depth: u32) -> f32 {
        if max_depth == 0 {
            0.0
        } else {
            self.depth as f32 / max_depth as f32
        }
    }

    /// Normalize sibling order relative to total siblings.
    #[inline]
    pub fn normalize_sibling(&self, total_siblings: u32) -> f32 {
        if total_siblings <= 1 {
            0.0
        } else {
            self.sibling_order as f32 / (total_siblings - 1) as f32
        }
    }

    /// Create a fully normalized coordinate for comparison across trajectories.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum depth in trajectory
    /// * `total_siblings` - Number of siblings at this level
    pub fn normalized(&self, max_depth: u32, total_siblings: u32) -> NormalizedCoordinate {
        NormalizedCoordinate {
            depth: self.normalize_depth(max_depth),
            sibling_order: self.normalize_sibling(total_siblings),
            homogeneity: self.homogeneity,
            temporal: self.temporal,
        }
    }
}

impl Default for TrajectoryCoordinate {
    fn default() -> Self {
        Self::root()
    }
}

/// Fully normalized coordinate where all dimensions are [0, 1].
///
/// Useful for comparing coordinates across trajectories of different sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedCoordinate {
    /// Normalized depth [0, 1]
    pub depth: f32,
    /// Normalized sibling order [0, 1]
    pub sibling_order: f32,
    /// Semantic homogeneity [0, 1]
    pub homogeneity: f32,
    /// Temporal position [0, 1]
    pub temporal: f32,
}

impl NormalizedCoordinate {
    /// Compute Euclidean distance between normalized coordinates.
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let d_depth = self.depth - other.depth;
        let d_sibling = self.sibling_order - other.sibling_order;
        let d_homo = self.homogeneity - other.homogeneity;
        let d_time = self.temporal - other.temporal;

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2)).sqrt()
    }

    /// Convert to array for batch operations.
    #[inline]
    pub fn to_array(&self) -> [f32; 4] {
        [self.depth, self.sibling_order, self.homogeneity, self.temporal]
    }
}

// ============================================================================
// 5D Coordinates (TPO Extension)
// ============================================================================

/// 5D coordinate for positioning an episode within a trajectory.
///
/// Extends the 4D IRCP/RCP coordinate with a complexity dimension from TPO
/// (Topological Preference Optimization). The complexity dimension captures
/// message information density (number of content parts: text, images, code blocks).
///
/// # TPO Integration
///
/// In TPO, the N dimension (n_parts) affects:
/// - **Path quality scoring**: Complex messages contribute more information
/// - **Preference strength**: Choosing paths with complex content over simple
/// - **Salience weighting**: Higher complexity = higher information density
///
/// # Example
///
/// ```
/// use rag_plusplus_core::trajectory::TrajectoryCoordinate5D;
///
/// // Simple text-only message
/// let simple = TrajectoryCoordinate5D::new(1, 0, 0.9, 0.3, 1);
///
/// // Complex multimodal message (text + 2 images + code block)
/// let complex = TrajectoryCoordinate5D::new(2, 0, 0.8, 0.5, 4);
///
/// // Complexity affects distance
/// let distance = simple.distance(&complex);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryCoordinate5D {
    /// Depth in the trajectory tree (0 = root)
    pub depth: u32,
    /// Position among siblings at this level (0-indexed)
    pub sibling_order: u32,
    /// Semantic similarity to parent episode [0, 1]
    pub homogeneity: f32,
    /// Normalized temporal position within trajectory [0, 1]
    pub temporal: f32,
    /// Message complexity (n_parts): number of content components
    /// - 1 = simple text
    /// - 2+ = multimodal (text + images, code blocks, etc.)
    pub complexity: u32,
}

impl TrajectoryCoordinate5D {
    /// Create a new 5D trajectory coordinate.
    ///
    /// # Arguments
    ///
    /// * `depth` - Distance from root (0 = root)
    /// * `sibling_order` - Position among siblings (0-indexed)
    /// * `homogeneity` - Semantic similarity to parent [0, 1]
    /// * `temporal` - Normalized timestamp [0, 1]
    /// * `complexity` - Number of content parts (1 = simple text)
    #[inline]
    pub fn new(depth: u32, sibling_order: u32, homogeneity: f32, temporal: f32, complexity: u32) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&homogeneity),
            "homogeneity must be in [0, 1], got {}",
            homogeneity
        );
        debug_assert!(
            (0.0..=1.0).contains(&temporal),
            "temporal must be in [0, 1], got {}",
            temporal
        );
        debug_assert!(complexity >= 1, "complexity must be at least 1");

        Self {
            depth,
            sibling_order,
            homogeneity,
            temporal,
            complexity,
        }
    }

    /// Create from a 4D coordinate with default complexity (1).
    #[inline]
    pub fn from_4d(coord: &TrajectoryCoordinate) -> Self {
        Self {
            depth: coord.depth,
            sibling_order: coord.sibling_order,
            homogeneity: coord.homogeneity,
            temporal: coord.temporal,
            complexity: 1,
        }
    }

    /// Create from a 4D coordinate with specified complexity.
    #[inline]
    pub fn from_4d_with_complexity(coord: &TrajectoryCoordinate, complexity: u32) -> Self {
        Self {
            depth: coord.depth,
            sibling_order: coord.sibling_order,
            homogeneity: coord.homogeneity,
            temporal: coord.temporal,
            complexity: complexity.max(1),
        }
    }

    /// Convert to 4D coordinate (discards complexity).
    #[inline]
    pub fn to_4d(&self) -> TrajectoryCoordinate {
        TrajectoryCoordinate {
            depth: self.depth,
            sibling_order: self.sibling_order,
            homogeneity: self.homogeneity,
            temporal: self.temporal,
        }
    }

    /// Create a root coordinate.
    #[inline]
    pub fn root() -> Self {
        Self {
            depth: 0,
            sibling_order: 0,
            homogeneity: 1.0,
            temporal: 0.0,
            complexity: 1,
        }
    }

    /// Compute Euclidean distance between two 5D coordinates.
    ///
    /// Uses log-scaled complexity to prevent domination by very complex messages.
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let d_depth = (self.depth as f32 - other.depth as f32).abs();
        let d_sibling = (self.sibling_order as f32 - other.sibling_order as f32).abs();
        let d_homo = self.homogeneity - other.homogeneity;
        let d_time = self.temporal - other.temporal;
        // Log-scale complexity to prevent domination (ln(1)=0, ln(e)=1, ln(10)≈2.3)
        let d_complexity = ((self.complexity as f32).ln_1p() - (other.complexity as f32).ln_1p()).abs();

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2) + d_complexity.powi(2)).sqrt()
    }

    /// Compute weighted distance with custom dimension weights.
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    /// * `weights` - (depth, sibling, homogeneity, temporal, complexity)
    #[inline]
    pub fn weighted_distance(&self, other: &Self, weights: (f32, f32, f32, f32, f32)) -> f32 {
        let d_depth = (self.depth as f32 - other.depth as f32).abs() * weights.0;
        let d_sibling = (self.sibling_order as f32 - other.sibling_order as f32).abs() * weights.1;
        let d_homo = (self.homogeneity - other.homogeneity).abs() * weights.2;
        let d_time = (self.temporal - other.temporal).abs() * weights.3;
        let d_complexity = ((self.complexity as f32).ln_1p() - (other.complexity as f32).ln_1p()).abs() * weights.4;

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2) + d_complexity.powi(2)).sqrt()
    }

    /// Compute weighted Euclidean distance using DLMWeights configuration.
    ///
    /// This is the primary distance metric used by DLM for coordinate-based retrieval.
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    /// * `weights` - DLMWeights configuration
    ///
    /// # Example
    ///
    /// ```
    /// use rag_plusplus_core::trajectory::{TrajectoryCoordinate5D, DLMWeights};
    ///
    /// let a = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.3, 1);
    /// let b = TrajectoryCoordinate5D::new(4, 1, 0.7, 0.6, 3);
    ///
    /// // Use default DLM weights
    /// let dist = a.dlm_distance(&b, &DLMWeights::default());
    ///
    /// // Use semantic-focused weights
    /// let semantic_dist = a.dlm_distance(&b, &DLMWeights::semantic_focused());
    /// ```
    #[inline]
    pub fn dlm_distance(&self, other: &Self, weights: &DLMWeights) -> f32 {
        self.weighted_distance(
            other,
            (weights.depth, weights.sibling, weights.homogeneity, weights.temporal, weights.complexity),
        )
    }

    /// Compute weighted Manhattan distance (L1 norm).
    ///
    /// DLM uses this for certain distance calculations where L1 is preferred
    /// over L2 (Euclidean).
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    /// * `weights` - DLMWeights configuration
    ///
    /// # Example
    ///
    /// ```
    /// use rag_plusplus_core::trajectory::{TrajectoryCoordinate5D, DLMWeights};
    ///
    /// let a = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.3, 1);
    /// let b = TrajectoryCoordinate5D::new(4, 1, 0.7, 0.6, 3);
    ///
    /// let dist = a.weighted_manhattan(&b, &DLMWeights::default());
    /// ```
    #[inline]
    pub fn weighted_manhattan(&self, other: &Self, weights: &DLMWeights) -> f32 {
        let d_depth = (self.depth as f32 - other.depth as f32).abs() * weights.depth;
        let d_sibling = (self.sibling_order as f32 - other.sibling_order as f32).abs() * weights.sibling;
        let d_homo = (self.homogeneity - other.homogeneity).abs() * weights.homogeneity;
        let d_time = (self.temporal - other.temporal).abs() * weights.temporal;
        let d_complexity = ((self.complexity as f32).ln_1p() - (other.complexity as f32).ln_1p()).abs() * weights.complexity;

        d_depth + d_sibling + d_homo + d_time + d_complexity
    }

    /// Compute unweighted Manhattan distance (L1 norm).
    #[inline]
    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        let d_depth = (self.depth as i64 - other.depth as i64).unsigned_abs() as f32;
        let d_sibling = (self.sibling_order as i64 - other.sibling_order as i64).unsigned_abs() as f32;
        let d_homo = (self.homogeneity - other.homogeneity).abs();
        let d_time = (self.temporal - other.temporal).abs();
        let d_complexity = ((self.complexity as f32).ln_1p() - (other.complexity as f32).ln_1p()).abs();

        d_depth + d_sibling + d_homo + d_time + d_complexity
    }

    /// Convert coordinate to a 5D vector for vector operations.
    ///
    /// Returns [depth, sibling_order, homogeneity, temporal, ln(1+complexity)].
    /// Uses log-scaled complexity for balanced vector representation.
    #[inline]
    pub fn as_vector(&self) -> [f32; 5] {
        [
            self.depth as f32,
            self.sibling_order as f32,
            self.homogeneity,
            self.temporal,
            (self.complexity as f32).ln_1p(),
        ]
    }

    /// Convert coordinate to a normalized 5D vector.
    ///
    /// All dimensions scaled to approximately [0, 1] range.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum depth in trajectory
    /// * `max_siblings` - Maximum number of siblings
    /// * `max_complexity` - Maximum complexity value
    #[inline]
    pub fn as_normalized_vector(&self, max_depth: u32, max_siblings: u32, max_complexity: u32) -> [f32; 5] {
        let depth_norm = if max_depth > 0 { self.depth as f32 / max_depth as f32 } else { 0.0 };
        let sibling_norm = if max_siblings > 0 { self.sibling_order as f32 / max_siblings as f32 } else { 0.0 };
        let complexity_norm = if max_complexity > 1 {
            (self.complexity as f32).ln_1p() / (max_complexity as f32).ln_1p()
        } else {
            0.0
        };

        [depth_norm, sibling_norm, self.homogeneity, self.temporal, complexity_norm]
    }

    /// Compute cosine similarity between two coordinates as vectors.
    ///
    /// DLM supports this for comparing coordinate "directions" rather than magnitudes.
    /// Useful for finding coordinates with similar relative positions.
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    ///
    /// # Returns
    ///
    /// Cosine similarity in range [-1, 1], where 1 = identical direction.
    ///
    /// # Example
    ///
    /// ```
    /// use rag_plusplus_core::trajectory::TrajectoryCoordinate5D;
    ///
    /// let a = TrajectoryCoordinate5D::new(2, 0, 0.9, 0.3, 1);
    /// let b = TrajectoryCoordinate5D::new(4, 0, 0.9, 0.6, 2);  // Similar direction
    ///
    /// let sim = a.cosine_similarity(&b);
    /// assert!(sim > 0.9);  // High similarity
    /// ```
    #[inline]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let a = self.as_vector();
        let b = other.as_vector();

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Compute combined distance using both Euclidean and cosine components.
    ///
    /// This hybrid metric balances magnitude (Euclidean) and direction (cosine)
    /// for more nuanced coordinate comparison.
    ///
    /// # Arguments
    ///
    /// * `other` - Other coordinate
    /// * `weights` - DLMWeights for Euclidean component
    /// * `cosine_weight` - Weight for cosine component [0, 1]
    ///
    /// # Returns
    ///
    /// Combined distance where higher = less similar.
    #[inline]
    pub fn hybrid_distance(&self, other: &Self, weights: &DLMWeights, cosine_weight: f32) -> f32 {
        let euclidean = self.dlm_distance(other, weights);
        let cosine_dist = 1.0 - self.cosine_similarity(other); // Convert similarity to distance

        let cw = cosine_weight.clamp(0.0, 1.0);
        (1.0 - cw) * euclidean + cw * cosine_dist
    }

    /// Compute normalized complexity factor [0, 1].
    ///
    /// Uses log scaling: complexity 1 → 0.0, complexity 10 → ~0.7, complexity 100 → ~0.9
    #[inline]
    pub fn complexity_factor(&self) -> f32 {
        // ln(1+x)/ln(1+max_typical) where max_typical ≈ 10
        (self.complexity as f32).ln_1p() / 10.0_f32.ln_1p()
    }

    /// Check if this is a complex message (more than just text).
    #[inline]
    pub fn is_multimodal(&self) -> bool {
        self.complexity > 1
    }

    /// Normalize all dimensions to [0, 1] for cross-trajectory comparison.
    pub fn normalized(&self, max_depth: u32, total_siblings: u32, max_complexity: u32) -> NormalizedCoordinate5D {
        let depth = if max_depth == 0 {
            0.0
        } else {
            self.depth as f32 / max_depth as f32
        };

        let sibling_order = if total_siblings <= 1 {
            0.0
        } else {
            self.sibling_order as f32 / (total_siblings - 1) as f32
        };

        let complexity = if max_complexity <= 1 {
            0.0
        } else {
            (self.complexity as f32).ln_1p() / (max_complexity as f32).ln_1p()
        };

        NormalizedCoordinate5D {
            depth,
            sibling_order,
            homogeneity: self.homogeneity,
            temporal: self.temporal,
            complexity,
        }
    }
}

impl Default for TrajectoryCoordinate5D {
    fn default() -> Self {
        Self::root()
    }
}

impl From<TrajectoryCoordinate> for TrajectoryCoordinate5D {
    fn from(coord: TrajectoryCoordinate) -> Self {
        Self::from_4d(&coord)
    }
}

impl From<&TrajectoryCoordinate> for TrajectoryCoordinate5D {
    fn from(coord: &TrajectoryCoordinate) -> Self {
        Self::from_4d(coord)
    }
}

/// Fully normalized 5D coordinate where all dimensions are [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedCoordinate5D {
    /// Normalized depth [0, 1]
    pub depth: f32,
    /// Normalized sibling order [0, 1]
    pub sibling_order: f32,
    /// Semantic homogeneity [0, 1]
    pub homogeneity: f32,
    /// Temporal position [0, 1]
    pub temporal: f32,
    /// Normalized complexity [0, 1] (log-scaled)
    pub complexity: f32,
}

impl NormalizedCoordinate5D {
    /// Compute Euclidean distance between normalized 5D coordinates.
    #[inline]
    pub fn distance(&self, other: &Self) -> f32 {
        let d_depth = self.depth - other.depth;
        let d_sibling = self.sibling_order - other.sibling_order;
        let d_homo = self.homogeneity - other.homogeneity;
        let d_time = self.temporal - other.temporal;
        let d_complexity = self.complexity - other.complexity;

        (d_depth.powi(2) + d_sibling.powi(2) + d_homo.powi(2) + d_time.powi(2) + d_complexity.powi(2)).sqrt()
    }

    /// Convert to array for batch operations.
    #[inline]
    pub fn to_array(&self) -> [f32; 5] {
        [self.depth, self.sibling_order, self.homogeneity, self.temporal, self.complexity]
    }
}

/// Compute coordinates for all episodes in a trajectory.
///
/// # Arguments
///
/// * `episodes` - List of (depth, sibling_order, embedding, timestamp)
/// * `parent_indices` - Parent index for each episode (None for root)
///
/// # Returns
///
/// Vector of TrajectoryCoordinate for each episode.
pub fn compute_trajectory_coordinates(
    episodes: &[(u32, u32, &[f32], i64)],
    parent_indices: &[Option<usize>],
) -> Vec<TrajectoryCoordinate> {
    if episodes.is_empty() {
        return Vec::new();
    }

    // Find trajectory time bounds
    let trajectory_start = episodes.iter().map(|(_, _, _, t)| *t).min().unwrap_or(0);
    let trajectory_end = episodes.iter().map(|(_, _, _, t)| *t).max().unwrap_or(0);

    let mut coordinates = Vec::with_capacity(episodes.len());

    for (i, (depth, sibling_order, embedding, timestamp)) in episodes.iter().enumerate() {
        let parent_embedding = parent_indices.get(i)
            .and_then(|p| *p)
            .and_then(|p| episodes.get(p))
            .map(|(_, _, e, _)| *e);

        let coord = TrajectoryCoordinate::from_episode(
            *depth,
            *sibling_order,
            embedding,
            parent_embedding,
            *timestamp,
            trajectory_start,
            trajectory_end,
        );

        coordinates.push(coord);
    }

    coordinates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_coordinate() {
        let coord = TrajectoryCoordinate::new(3, 1, 0.85, 0.5);
        assert_eq!(coord.depth, 3);
        assert_eq!(coord.sibling_order, 1);
        assert!((coord.homogeneity - 0.85).abs() < 1e-6);
        assert!((coord.temporal - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_root_coordinate() {
        let root = TrajectoryCoordinate::root();
        assert_eq!(root.depth, 0);
        assert_eq!(root.sibling_order, 0);
        assert!((root.homogeneity - 1.0).abs() < 1e-6);
        assert!((root.temporal - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_identical() {
        let coord = TrajectoryCoordinate::new(5, 2, 0.7, 0.3);
        assert!((coord.distance(&coord) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_different() {
        let a = TrajectoryCoordinate::new(0, 0, 1.0, 0.0);
        let b = TrajectoryCoordinate::new(1, 0, 1.0, 0.0);

        // Distance should be 1.0 (only depth differs by 1)
        assert!((a.distance(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_all_dimensions() {
        let a = TrajectoryCoordinate::new(0, 0, 0.0, 0.0);
        let b = TrajectoryCoordinate::new(1, 1, 1.0, 1.0);

        // sqrt(1 + 1 + 1 + 1) = 2.0
        assert!((a.distance(&b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = TrajectoryCoordinate::new(0, 0, 0.0, 0.0);
        let b = TrajectoryCoordinate::new(1, 1, 1.0, 1.0);

        // 1 + 1 + 1 + 1 = 4.0
        assert!((a.manhattan_distance(&b) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_distance() {
        let a = TrajectoryCoordinate::new(0, 0, 0.5, 0.5);
        let b = TrajectoryCoordinate::new(2, 0, 0.5, 0.5);

        // Only depth differs by 2, weight of 0.5 → 1.0
        let weights = (0.5, 1.0, 1.0, 1.0);
        assert!((a.weighted_distance(&b, weights) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_depth() {
        let coord = TrajectoryCoordinate::new(5, 0, 0.5, 0.5);
        assert!((coord.normalize_depth(10) - 0.5).abs() < 1e-6);
        assert!((coord.normalize_depth(5) - 1.0).abs() < 1e-6);
        assert!((coord.normalize_depth(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_sibling() {
        let coord = TrajectoryCoordinate::new(0, 2, 0.5, 0.5);
        assert!((coord.normalize_sibling(5) - 0.5).abs() < 1e-6); // 2/4 = 0.5
        assert!((coord.normalize_sibling(3) - 1.0).abs() < 1e-6); // 2/2 = 1.0
        assert!((coord.normalize_sibling(1) - 0.0).abs() < 1e-6); // Only child
    }

    #[test]
    fn test_normalized_coordinate() {
        let coord = TrajectoryCoordinate::new(5, 2, 0.8, 0.6);
        let normalized = coord.normalized(10, 5);

        assert!((normalized.depth - 0.5).abs() < 1e-6);
        assert!((normalized.sibling_order - 0.5).abs() < 1e-6);
        assert!((normalized.homogeneity - 0.8).abs() < 1e-6);
        assert!((normalized.temporal - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_from_episode() {
        // Create test embeddings (unit vectors)
        let parent_embedding = vec![1.0, 0.0, 0.0];
        let child_embedding = vec![0.8, 0.6, 0.0]; // cos = 0.8

        let coord = TrajectoryCoordinate::from_episode(
            3,
            1,
            &child_embedding,
            Some(&parent_embedding),
            500,
            0,
            1000,
        );

        assert_eq!(coord.depth, 3);
        assert_eq!(coord.sibling_order, 1);
        assert!((coord.homogeneity - 0.8).abs() < 0.01);
        assert!((coord.temporal - 0.5).abs() < 1e-6);
    }

    // ========== 5D Coordinate Tests (TPO Extension) ==========

    #[test]
    fn test_5d_new_coordinate() {
        let coord = TrajectoryCoordinate5D::new(3, 1, 0.85, 0.5, 2);
        assert_eq!(coord.depth, 3);
        assert_eq!(coord.sibling_order, 1);
        assert!((coord.homogeneity - 0.85).abs() < 1e-6);
        assert!((coord.temporal - 0.5).abs() < 1e-6);
        assert_eq!(coord.complexity, 2);
    }

    #[test]
    fn test_5d_root_coordinate() {
        let root = TrajectoryCoordinate5D::root();
        assert_eq!(root.depth, 0);
        assert_eq!(root.sibling_order, 0);
        assert!((root.homogeneity - 1.0).abs() < 1e-6);
        assert!((root.temporal - 0.0).abs() < 1e-6);
        assert_eq!(root.complexity, 1);
    }

    #[test]
    fn test_5d_from_4d() {
        let coord_4d = TrajectoryCoordinate::new(2, 1, 0.7, 0.4);
        let coord_5d = TrajectoryCoordinate5D::from_4d(&coord_4d);

        assert_eq!(coord_5d.depth, 2);
        assert_eq!(coord_5d.sibling_order, 1);
        assert!((coord_5d.homogeneity - 0.7).abs() < 1e-6);
        assert!((coord_5d.temporal - 0.4).abs() < 1e-6);
        assert_eq!(coord_5d.complexity, 1); // Default
    }

    #[test]
    fn test_5d_from_4d_with_complexity() {
        let coord_4d = TrajectoryCoordinate::new(2, 1, 0.7, 0.4);
        let coord_5d = TrajectoryCoordinate5D::from_4d_with_complexity(&coord_4d, 5);

        assert_eq!(coord_5d.complexity, 5);
    }

    #[test]
    fn test_5d_to_4d() {
        let coord_5d = TrajectoryCoordinate5D::new(3, 2, 0.8, 0.6, 4);
        let coord_4d = coord_5d.to_4d();

        assert_eq!(coord_4d.depth, 3);
        assert_eq!(coord_4d.sibling_order, 2);
        assert!((coord_4d.homogeneity - 0.8).abs() < 1e-6);
        assert!((coord_4d.temporal - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_5d_distance_identical() {
        let coord = TrajectoryCoordinate5D::new(5, 2, 0.7, 0.3, 3);
        assert!((coord.distance(&coord) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_5d_distance_same_complexity() {
        // Same complexity should reduce to 4D distance
        let a = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(1, 0, 1.0, 0.0, 1);

        // Distance should be 1.0 (only depth differs by 1, complexity same)
        assert!((a.distance(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_5d_distance_complexity_affects() {
        // Different complexity should increase distance
        let a = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 10);

        // Distance should be ln(11) - ln(2) ≈ 1.7
        let d = a.distance(&b);
        assert!(d > 1.0); // Complexity adds to distance
        assert!(d < 3.0); // But log scaling prevents domination
    }

    #[test]
    fn test_5d_complexity_factor() {
        let simple = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 1);
        let medium = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 5);
        let complex = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 10);

        // Complexity 1 → ~0.0 (ln(2)/ln(11))
        assert!(simple.complexity_factor() < 0.3);

        // Complexity 5 → ~0.5-0.6
        assert!(medium.complexity_factor() > 0.4);
        assert!(medium.complexity_factor() < 0.8);

        // Complexity 10 → ~0.9-1.0
        assert!(complex.complexity_factor() > 0.85);
    }

    #[test]
    fn test_5d_is_multimodal() {
        let simple = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 1);
        let multimodal = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 3);

        assert!(!simple.is_multimodal());
        assert!(multimodal.is_multimodal());
    }

    #[test]
    fn test_5d_from_trait() {
        let coord_4d = TrajectoryCoordinate::new(1, 0, 0.9, 0.2);
        let coord_5d: TrajectoryCoordinate5D = coord_4d.into();

        assert_eq!(coord_5d.depth, 1);
        assert_eq!(coord_5d.complexity, 1);
    }

    #[test]
    fn test_5d_normalized() {
        let coord = TrajectoryCoordinate5D::new(5, 2, 0.8, 0.6, 5);
        let normalized = coord.normalized(10, 5, 10);

        assert!((normalized.depth - 0.5).abs() < 1e-6);
        assert!((normalized.sibling_order - 0.5).abs() < 1e-6);
        assert!((normalized.homogeneity - 0.8).abs() < 1e-6);
        assert!((normalized.temporal - 0.6).abs() < 1e-6);
        // Complexity: ln(6)/ln(11) ≈ 0.75
        assert!(normalized.complexity > 0.7);
        assert!(normalized.complexity < 0.8);
    }

    #[test]
    fn test_normalized_5d_distance() {
        let a = NormalizedCoordinate5D {
            depth: 0.0,
            sibling_order: 0.0,
            homogeneity: 0.0,
            temporal: 0.0,
            complexity: 0.0,
        };
        let b = NormalizedCoordinate5D {
            depth: 1.0,
            sibling_order: 1.0,
            homogeneity: 1.0,
            temporal: 1.0,
            complexity: 1.0,
        };

        // sqrt(1 + 1 + 1 + 1 + 1) ≈ 2.236
        let d = a.distance(&b);
        assert!((d - 5.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_normalized_5d_to_array() {
        let coord = NormalizedCoordinate5D {
            depth: 0.1,
            sibling_order: 0.2,
            homogeneity: 0.3,
            temporal: 0.4,
            complexity: 0.5,
        };

        let arr = coord.to_array();
        assert_eq!(arr, [0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    // ========== DLM Weight Configuration Tests ==========

    #[test]
    fn test_dlm_weights_default() {
        let weights = DLMWeights::default();

        // Default weights from DLM config
        assert!((weights.depth - 0.25).abs() < 1e-6);
        assert!((weights.sibling - 0.15).abs() < 1e-6);
        assert!((weights.homogeneity - 0.30).abs() < 1e-6);
        assert!((weights.temporal - 0.20).abs() < 1e-6);
        assert!((weights.complexity - 0.10).abs() < 1e-6);

        // Weights should sum to 1.0
        let total = weights.depth + weights.sibling + weights.homogeneity + weights.temporal + weights.complexity;
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dlm_weights_normalization() {
        // Unnormalized weights
        let weights = DLMWeights::new(1.0, 1.0, 1.0, 1.0, 1.0);

        // Should normalize to equal weights
        assert!((weights.depth - 0.2).abs() < 1e-6);
        assert!((weights.sibling - 0.2).abs() < 1e-6);
        assert!((weights.homogeneity - 0.2).abs() < 1e-6);
        assert!((weights.temporal - 0.2).abs() < 1e-6);
        assert!((weights.complexity - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_dlm_weights_presets() {
        let semantic = DLMWeights::semantic_focused();
        assert!(semantic.homogeneity > semantic.depth);
        assert!(semantic.homogeneity > semantic.sibling);

        let structural = DLMWeights::structural_focused();
        assert!(structural.depth > structural.homogeneity);
        assert!(structural.sibling > structural.complexity);

        let temporal = DLMWeights::temporal_focused();
        assert!(temporal.temporal > temporal.depth);
        assert!(temporal.temporal > temporal.homogeneity);

        let equal = DLMWeights::equal();
        assert!((equal.depth - 0.2).abs() < 1e-6);
        assert!((equal.sibling - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_dlm_weights_array_conversion() {
        let weights = DLMWeights::default();
        let arr = weights.to_array();

        let reconstructed = DLMWeights::from_array(arr);
        assert!((weights.depth - reconstructed.depth).abs() < 1e-6);
        assert!((weights.sibling - reconstructed.sibling).abs() < 1e-6);
        assert!((weights.homogeneity - reconstructed.homogeneity).abs() < 1e-6);
        assert!((weights.temporal - reconstructed.temporal).abs() < 1e-6);
        assert!((weights.complexity - reconstructed.complexity).abs() < 1e-6);
    }

    // ========== DLM Distance Methods Tests ==========

    #[test]
    fn test_5d_dlm_distance() {
        let a = TrajectoryCoordinate5D::new(0, 0, 1.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(2, 1, 0.8, 0.3, 2);

        let weights = DLMWeights::default();
        let dist = a.dlm_distance(&b, &weights);

        // Distance should be positive and reasonable
        assert!(dist > 0.0);
        assert!(dist < 5.0);
    }

    #[test]
    fn test_5d_dlm_distance_identical() {
        let coord = TrajectoryCoordinate5D::new(3, 2, 0.7, 0.5, 3);
        let weights = DLMWeights::default();

        assert!((coord.dlm_distance(&coord, &weights) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_5d_weighted_manhattan() {
        let a = TrajectoryCoordinate5D::new(0, 0, 0.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(2, 2, 1.0, 1.0, 1);

        // With equal weights
        let weights = DLMWeights::equal();
        let dist = a.weighted_manhattan(&b, &weights);

        // Each dimension contributes: 2*0.2 + 2*0.2 + 1.0*0.2 + 1.0*0.2 + 0*0.2 = 1.2
        assert!((dist - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_5d_manhattan_distance() {
        let a = TrajectoryCoordinate5D::new(0, 0, 0.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(2, 2, 1.0, 1.0, 1);

        // Unweighted: 2 + 2 + 1.0 + 1.0 + 0 = 6.0
        let dist = a.manhattan_distance(&b);
        assert!((dist - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_5d_as_vector() {
        let coord = TrajectoryCoordinate5D::new(3, 2, 0.75, 0.5, 4);
        let vec = coord.as_vector();

        assert!((vec[0] - 3.0).abs() < 1e-6);
        assert!((vec[1] - 2.0).abs() < 1e-6);
        assert!((vec[2] - 0.75).abs() < 1e-6);
        assert!((vec[3] - 0.5).abs() < 1e-6);
        // ln(1+4) = ln(5) ≈ 1.609
        assert!((vec[4] - 5.0_f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_5d_as_normalized_vector() {
        let coord = TrajectoryCoordinate5D::new(5, 3, 0.8, 0.6, 5);
        let vec = coord.as_normalized_vector(10, 6, 10);

        assert!((vec[0] - 0.5).abs() < 1e-6);  // 5/10
        assert!((vec[1] - 0.5).abs() < 1e-6);  // 3/6
        assert!((vec[2] - 0.8).abs() < 1e-6);  // homogeneity unchanged
        assert!((vec[3] - 0.6).abs() < 1e-6);  // temporal unchanged
        // ln(6)/ln(11) ≈ 0.748
        assert!(vec[4] > 0.7 && vec[4] < 0.8);
    }

    #[test]
    fn test_5d_cosine_similarity_identical() {
        let coord = TrajectoryCoordinate5D::new(3, 2, 0.8, 0.5, 2);
        let sim = coord.cosine_similarity(&coord);

        // Identical vectors should have cosine similarity of 1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_5d_cosine_similarity_similar() {
        // Scaled version should have same direction
        let a = TrajectoryCoordinate5D::new(2, 1, 0.8, 0.4, 2);
        let b = TrajectoryCoordinate5D::new(4, 2, 0.8, 0.4, 2);  // Same direction, scaled

        let sim = a.cosine_similarity(&b);

        // Should be high but not exactly 1.0 due to log scaling of complexity
        assert!(sim > 0.95);
    }

    #[test]
    fn test_5d_cosine_similarity_different() {
        let a = TrajectoryCoordinate5D::new(10, 0, 0.0, 0.0, 1);
        let b = TrajectoryCoordinate5D::new(0, 10, 1.0, 1.0, 10);

        let sim = a.cosine_similarity(&b);

        // Very different directions should have low similarity
        assert!(sim < 0.5);
    }

    #[test]
    fn test_5d_hybrid_distance() {
        let a = TrajectoryCoordinate5D::new(2, 1, 0.9, 0.3, 1);
        let b = TrajectoryCoordinate5D::new(4, 2, 0.7, 0.6, 3);

        let weights = DLMWeights::default();

        // Pure Euclidean (cosine_weight = 0)
        let euclidean_only = a.hybrid_distance(&b, &weights, 0.0);

        // Pure cosine (cosine_weight = 1)
        let cosine_only = a.hybrid_distance(&b, &weights, 1.0);

        // Mixed (cosine_weight = 0.5)
        let mixed = a.hybrid_distance(&b, &weights, 0.5);

        // Mixed should be between the two extremes
        let min = euclidean_only.min(cosine_only);
        let max = euclidean_only.max(cosine_only);
        assert!(mixed >= min - 1e-6 && mixed <= max + 1e-6);
    }

    #[test]
    fn test_5d_hybrid_distance_identical() {
        let coord = TrajectoryCoordinate5D::new(3, 2, 0.8, 0.5, 2);
        let weights = DLMWeights::default();

        let dist = coord.hybrid_distance(&coord, &weights, 0.5);

        // Distance to self should be 0
        assert!((dist - 0.0).abs() < 1e-6);
    }
}
