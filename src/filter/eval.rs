//! Filter Evaluation
//!
//! Compiles and evaluates filter expressions against metadata.

use ahash::AHashSet;
use std::collections::HashMap;

use super::ast::{FilterExpr, FilterOp, FilterValue};
use crate::types::MetadataValue;

/// Compiled filter for fast evaluation.
#[derive(Debug, Clone)]
pub struct CompiledFilter {
    expr: FilterExpr,
    /// Pre-computed hash sets for IN operations
    in_sets: HashMap<String, AHashSet<String>>,
}

impl CompiledFilter {
    /// Compile a filter expression.
    #[must_use]
    pub fn compile(expr: FilterExpr) -> Self {
        let mut in_sets = HashMap::new();
        Self::extract_in_sets(&expr, &mut in_sets);
        Self { expr, in_sets }
    }

    /// Extract IN sets for fast lookup.
    fn extract_in_sets(expr: &FilterExpr, sets: &mut HashMap<String, AHashSet<String>>) {
        match expr {
            FilterExpr::Comparison { field, op: FilterOp::In, value: FilterValue::StringList(items) } => {
                sets.insert(field.clone(), items.iter().cloned().collect());
            }
            FilterExpr::And(exprs) | FilterExpr::Or(exprs) => {
                for e in exprs {
                    Self::extract_in_sets(e, sets);
                }
            }
            FilterExpr::Not(e) => Self::extract_in_sets(e, sets),
            _ => {}
        }
    }

    /// Evaluate the filter against metadata.
    #[must_use]
    pub fn evaluate(&self, metadata: &HashMap<String, MetadataValue>) -> bool {
        self.eval_expr(&self.expr, metadata)
    }

    fn eval_expr(&self, expr: &FilterExpr, metadata: &HashMap<String, MetadataValue>) -> bool {
        match expr {
            FilterExpr::True => true,
            FilterExpr::False => false,
            FilterExpr::Comparison { field, op, value } => {
                self.eval_comparison(field, *op, value, metadata)
            }
            FilterExpr::And(exprs) => exprs.iter().all(|e| self.eval_expr(e, metadata)),
            FilterExpr::Or(exprs) => exprs.iter().any(|e| self.eval_expr(e, metadata)),
            FilterExpr::Not(e) => !self.eval_expr(e, metadata),
        }
    }

    fn eval_comparison(
        &self,
        field: &str,
        op: FilterOp,
        filter_value: &FilterValue,
        metadata: &HashMap<String, MetadataValue>,
    ) -> bool {
        let Some(meta_value) = metadata.get(field) else {
            // Field not present - only NotIn and Ne can match
            return matches!(op, FilterOp::NotIn | FilterOp::Ne);
        };

        match (meta_value, filter_value) {
            // String comparisons
            (MetadataValue::String(s), FilterValue::String(v)) => {
                self.compare_strings(s, v, op)
            }
            
            // Integer comparisons
            (MetadataValue::Int(i), FilterValue::Int(v)) => {
                self.compare_numbers(*i as f64, *v as f64, op)
            }
            
            // Float comparisons
            (MetadataValue::Float(f), FilterValue::Float(v)) => {
                self.compare_numbers(*f, *v, op)
            }
            
            // Cross-type numeric
            (MetadataValue::Int(i), FilterValue::Float(v)) => {
                self.compare_numbers(*i as f64, *v, op)
            }
            (MetadataValue::Float(f), FilterValue::Int(v)) => {
                self.compare_numbers(*f, *v as f64, op)
            }
            
            // Boolean
            (MetadataValue::Bool(b), FilterValue::Bool(v)) => match op {
                FilterOp::Eq => b == v,
                FilterOp::Ne => b != v,
                _ => false,
            },
            
            // IN set (string)
            (MetadataValue::String(s), FilterValue::StringList(_)) => {
                match op {
                    FilterOp::In => self.in_sets.get(field)
                        .map(|set| set.contains(s))
                        .unwrap_or(false),
                    FilterOp::NotIn => self.in_sets.get(field)
                        .map(|set| !set.contains(s))
                        .unwrap_or(true),
                    _ => false,
                }
            }
            
            // Range
            (MetadataValue::Int(i), FilterValue::Range(min, max)) => {
                let f = *i as f64;
                matches!(op, FilterOp::InRange) && f >= *min && f <= *max
            }
            (MetadataValue::Float(f), FilterValue::Range(min, max)) => {
                matches!(op, FilterOp::InRange) && *f >= *min && *f <= *max
            }
            
            // Type mismatch
            _ => false,
        }
    }

    fn compare_strings(&self, s: &str, v: &str, op: FilterOp) -> bool {
        match op {
            FilterOp::Eq => s == v,
            FilterOp::Ne => s != v,
            FilterOp::Gt => s > v,
            FilterOp::Gte => s >= v,
            FilterOp::Lt => s < v,
            FilterOp::Lte => s <= v,
            FilterOp::Contains => s.contains(v),
            FilterOp::StartsWith => s.starts_with(v),
            FilterOp::EndsWith => s.ends_with(v),
            _ => false,
        }
    }

    fn compare_numbers(&self, a: f64, b: f64, op: FilterOp) -> bool {
        match op {
            FilterOp::Eq => (a - b).abs() < f64::EPSILON,
            FilterOp::Ne => (a - b).abs() >= f64::EPSILON,
            FilterOp::Gt => a > b,
            FilterOp::Gte => a >= b,
            FilterOp::Lt => a < b,
            FilterOp::Lte => a <= b,
            _ => false,
        }
    }
}

/// Filter evaluator with caching.
#[derive(Debug)]
pub struct FilterEvaluator {
    filters: HashMap<String, CompiledFilter>,
}

impl FilterEvaluator {
    /// Create a new filter evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
        }
    }

    /// Register a named filter.
    pub fn register(&mut self, name: impl Into<String>, expr: FilterExpr) {
        self.filters.insert(name.into(), CompiledFilter::compile(expr));
    }

    /// Evaluate a registered filter.
    #[must_use]
    pub fn evaluate(&self, name: &str, metadata: &HashMap<String, MetadataValue>) -> Option<bool> {
        self.filters.get(name).map(|f| f.evaluate(metadata))
    }

    /// Evaluate an expression directly.
    #[must_use]
    pub fn evaluate_expr(expr: &FilterExpr, metadata: &HashMap<String, MetadataValue>) -> bool {
        CompiledFilter::compile(expr.clone()).evaluate(metadata)
    }
}

impl Default for FilterEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_metadata() -> HashMap<String, MetadataValue> {
        let mut m = HashMap::new();
        m.insert("genre".to_string(), MetadataValue::String("electronic".to_string()));
        m.insert("tempo".to_string(), MetadataValue::Int(128));
        m.insert("duration".to_string(), MetadataValue::Float(180.5));
        m.insert("is_remix".to_string(), MetadataValue::Bool(false));
        m
    }

    #[test]
    fn test_eq_string() {
        let filter = CompiledFilter::compile(FilterExpr::eq("genre", FilterValue::string("electronic")));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata));
    }

    #[test]
    fn test_ne_string() {
        let filter = CompiledFilter::compile(FilterExpr::ne("genre", FilterValue::string("rock")));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata));
    }

    #[test]
    fn test_gt_int() {
        let filter = CompiledFilter::compile(FilterExpr::gt("tempo", FilterValue::int(120)));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata)); // 128 > 120
    }

    #[test]
    fn test_range() {
        let filter = CompiledFilter::compile(FilterExpr::in_range("tempo", 100.0, 140.0));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata)); // 128 in [100, 140]
    }

    #[test]
    fn test_in_set() {
        let filter = CompiledFilter::compile(FilterExpr::in_set(
            "genre",
            vec!["electronic".to_string(), "house".to_string()],
        ));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata));
    }

    #[test]
    fn test_and() {
        let filter = CompiledFilter::compile(FilterExpr::and(vec![
            FilterExpr::eq("genre", FilterValue::string("electronic")),
            FilterExpr::gt("tempo", FilterValue::int(100)),
        ]));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata));
    }

    #[test]
    fn test_or() {
        let filter = CompiledFilter::compile(FilterExpr::or(vec![
            FilterExpr::eq("genre", FilterValue::string("rock")),
            FilterExpr::gt("tempo", FilterValue::int(100)),
        ]));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata)); // Second clause matches
    }

    #[test]
    fn test_not() {
        let filter = CompiledFilter::compile(FilterExpr::not(
            FilterExpr::eq("is_remix", FilterValue::bool(true))
        ));
        let metadata = create_metadata();
        
        assert!(filter.evaluate(&metadata)); // is_remix is false
    }

    #[test]
    fn test_missing_field() {
        let filter = CompiledFilter::compile(FilterExpr::eq("artist", FilterValue::string("Unknown")));
        let metadata = create_metadata();
        
        assert!(!filter.evaluate(&metadata)); // Field doesn't exist
    }
}
