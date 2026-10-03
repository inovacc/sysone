//! s07: temperature lookup — `clamp_temperature` + `temp_bucket` (laya/common.py:367-388) and
//! `t_scale = temperature_by_options.get(bucket, temperature[qt])` (laya/agent.py:392-396, 768).
//! The checkpoint's `temperature` tensor is never read (docs/laya-model-analysis.md §9).

use serde_json::{Map, Value};

use crate::normalize::QType;

pub const TEMP_MIN: f64 = 0.5;
pub const TEMP_MAX: f64 = 5.0;

/// Python `float(t)` then clamp; non-numbers, NaN and ±inf become 1.0.
pub fn clamp_temperature(t: &Value) -> f64 {
    let f = match t {
        Value::Number(n) => n.to_string().parse::<f64>().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    };
    match f {
        Some(f) if f.is_finite() => TEMP_MAX.min(TEMP_MIN.max(f)),
        _ => 1.0,
    }
}

fn raw_f64(t: &Value) -> f64 {
    match t {
        Value::Number(n) => n.to_string().parse::<f64>().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

pub fn temp_bucket(qt: QType, k: usize) -> String {
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    format!("{}:{}", qt.name(), size)
}

pub struct Temperatures {
    per_type_raw: Vec<Value>,
    buckets_raw: Map<String, Value>,
}

pub struct Lookup {
    pub bucket: String,
    pub source: &'static str,
    pub t_raw: f64,
    pub t: f64,
}

impl Temperatures {
    /// From `rl_agent_config.json` (`temperature`, `temperature_by_options`).
    pub fn from_config(cfg: &Value) -> Self {
        let per_type_raw = cfg
            .get("temperature")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)]);
        let buckets_raw = cfg.get("temperature_by_options").and_then(Value::as_object).cloned().unwrap_or_default();
        Self { per_type_raw, buckets_raw }
    }

    pub fn lookup(&self, qt: QType, k: usize) -> Lookup {
        let bucket = temp_bucket(qt, k);
        match self.buckets_raw.get(&bucket) {
            Some(v) => Lookup { t_raw: raw_f64(v), t: clamp_temperature(v), source: "bucket", bucket },
            None => {
                let v = &self.per_type_raw[qt.code()];
                Lookup { t_raw: raw_f64(v), t: clamp_temperature(v), source: "per_type", bucket }
            }
        }
    }
}
