//! Request errors with the Python exception class the reference raises, so error parity is comparable
//! (`status.json`: stage + error_kind; the message text is informational).

use std::fmt;

#[derive(Debug, Clone)]
pub struct LayaError {
    /// Python exception class name: `ValueError`, `TypeError`.
    pub kind: &'static str,
    /// SDC stage in which the reference raises it (`s01`, `s03`, ...).
    pub stage: &'static str,
    pub message: String,
}

impl LayaError {
    pub fn value(stage: &'static str, message: impl Into<String>) -> Self {
        Self { kind: "ValueError", stage, message: message.into() }
    }
    pub fn type_err(stage: &'static str, message: impl Into<String>) -> Self {
        Self { kind: "TypeError", stage, message: message.into() }
    }
}

impl fmt::Display for LayaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.kind, self.stage, self.message)
    }
}

impl std::error::Error for LayaError {}
