//! Filter AST (Abstract Syntax Tree)
//!
//! Defines the filter expression structure.

use std::collections::HashSet;

/// Value types supported in filters.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// List of strings (for IN operator)
    StringList(Vec<String>),
    /// Numeric range (min, max, inclusive)
    Range(f64, f64),
}

impl FilterValue {
    /// Create a string value.
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create an integer value.
    pub const fn int(i: i64) -> Self {
        Self::Int(i)
    }

    /// Create a float value.
    pub const fn float(f: f64) -> Self {
        Self::Float(f)
    }

    /// Create a boolean value.
    pub const fn bool(b: bool) -> Self {
        Self::Bool(b)
    }

    /// Create a string list value.
    pub fn string_list(items: Vec<String>) -> Self {
        Self::StringList(items)
    }

    /// Create a range value.
    pub const fn range(min: f64, max: f64) -> Self {
        Self::Range(min, max)
    }
}

/// Filter comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// In set
    In,
    /// Not in set
    NotIn,
    /// In range (inclusive)
    InRange,
    /// Contains (for string)
    Contains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
}

/// Filter expression.
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// Field comparison: field op value
    Comparison {
        field: String,
        op: FilterOp,
        value: FilterValue,
    },
    /// Logical AND
    And(Vec<FilterExpr>),
    /// Logical OR
    Or(Vec<FilterExpr>),
    /// Logical NOT
    Not(Box<FilterExpr>),
    /// Always true
    True,
    /// Always false
    False,
}

impl FilterExpr {
    /// Create an equality filter.
    pub fn eq(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Eq,
            value,
        }
    }

    /// Create a not-equal filter.
    pub fn ne(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Ne,
            value,
        }
    }

    /// Create a greater-than filter.
    pub fn gt(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Gt,
            value,
        }
    }

    /// Create a greater-than-or-equal filter.
    pub fn gte(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Gte,
            value,
        }
    }

    /// Create a less-than filter.
    pub fn lt(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Lt,
            value,
        }
    }

    /// Create a less-than-or-equal filter.
    pub fn lte(field: impl Into<String>, value: FilterValue) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::Lte,
            value,
        }
    }

    /// Create an IN filter.
    pub fn in_set(field: impl Into<String>, values: Vec<String>) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::In,
            value: FilterValue::StringList(values),
        }
    }

    /// Create a range filter.
    pub fn in_range(field: impl Into<String>, min: f64, max: f64) -> Self {
        Self::Comparison {
            field: field.into(),
            op: FilterOp::InRange,
            value: FilterValue::Range(min, max),
        }
    }

    /// Combine with AND.
    pub fn and(exprs: Vec<FilterExpr>) -> Self {
        if exprs.is_empty() {
            Self::True
        } else if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            Self::And(exprs)
        }
    }

    /// Combine with OR.
    pub fn or(exprs: Vec<FilterExpr>) -> Self {
        if exprs.is_empty() {
            Self::False
        } else if exprs.len() == 1 {
            exprs.into_iter().next().unwrap()
        } else {
            Self::Or(exprs)
        }
    }

    /// Negate.
    pub fn not(expr: FilterExpr) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Get all fields referenced by this expression.
    pub fn referenced_fields(&self) -> HashSet<String> {
        let mut fields = HashSet::new();
        self.collect_fields(&mut fields);
        fields
    }

    fn collect_fields(&self, fields: &mut HashSet<String>) {
        match self {
            Self::Comparison { field, .. } => {
                fields.insert(field.clone());
            }
            Self::And(exprs) | Self::Or(exprs) => {
                for expr in exprs {
                    expr.collect_fields(fields);
                }
            }
            Self::Not(expr) => expr.collect_fields(fields),
            Self::True | Self::False => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_filter() {
        let filter = FilterExpr::eq("genre", FilterValue::string("electronic"));
        
        if let FilterExpr::Comparison { field, op, value } = filter {
            assert_eq!(field, "genre");
            assert_eq!(op, FilterOp::Eq);
            assert_eq!(value, FilterValue::String("electronic".to_string()));
        } else {
            panic!("Expected Comparison");
        }
    }

    #[test]
    fn test_and_filter() {
        let filter = FilterExpr::and(vec![
            FilterExpr::eq("genre", FilterValue::string("electronic")),
            FilterExpr::gt("tempo", FilterValue::int(120)),
        ]);

        let fields = filter.referenced_fields();
        assert!(fields.contains("genre"));
        assert!(fields.contains("tempo"));
    }

    #[test]
    fn test_empty_and() {
        let filter = FilterExpr::and(vec![]);
        assert!(matches!(filter, FilterExpr::True));
    }
}
