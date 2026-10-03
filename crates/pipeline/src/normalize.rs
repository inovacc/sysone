//! s01: question validation (`Agent._check_question`, laya/agent.py:556-624) and normalization
//! (`Agent._to_internal`, laya/agent.py:626-646), ported branch for branch.

use serde_json::{Map, Value};

use crate::error::LayaError;
use crate::pyjson::{dumps, py_eq_key, py_repr, py_str_repr, py_strip, py_type_name};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
    pub fn code(self) -> usize {
        self as usize
    }
}

/// Normalized criteria.
#[derive(Clone, Debug)]
pub enum Crit {
    /// Ordered (label, description-or-null); list criteria are deduplicated under Python equality.
    Choice(Vec<(Value, Value)>),
    /// Level descriptions, index 0 first.
    Score(Vec<Value>),
    /// `None` when omitted; keys lowered with `str(k).lower()`.
    Noul(Option<Map<String, Value>>),
}

/// The internal question dict `{"t", "ins", "crit"[, "labels"]}`.
#[derive(Clone, Debug)]
pub struct Internal {
    pub qid: String,
    pub t: QType,
    pub ins: String,
    pub crit: Crit,
    /// `Some` when the request had a `labels` key (its value may be null).
    pub labels: Option<Value>,
}

fn qrepr(qid: &str) -> String {
    py_str_repr(qid)
}

/// `_check_question` (laya/agent.py:565-624). The messages are the reference's own text: laya-serve sends
/// them to the client as the 422 `detail`, so a client sees the same words from either server.
pub fn check_question(qid: &str, qdef: &Value) -> Result<(), LayaError> {
    let err = |msg: String| LayaError::value("s01", msg);
    let Some(obj) = qdef.as_object() else {
        return Err(err(format!("question {}: definition must be a dict, got {}", qrepr(qid), py_type_name(qdef))));
    };
    let tval = obj.get("type").unwrap_or(&Value::Null);
    let t = match tval {
        // `t not in QTYPES` on a dict: TypeError("unhashable type: 'list'")
        Value::Array(_) | Value::Object(_) => {
            return Err(LayaError::type_err("s01", format!("unhashable type: {}", py_str_repr(py_type_name(tval)))));
        }
        Value::String(s) if matches!(s.as_str(), "choice" | "score" | "noul") => s.as_str(),
        _ => {
            return Err(err(format!("question {}: unknown type {}; use one of ['choice', 'noul', 'score']", qrepr(qid), py_repr(tval))));
        }
    };
    if !obj.contains_key("instructions") {
        return Err(err(format!("question {}: no 'instructions'; add the text the model should answer", qrepr(qid))));
    }
    let crit = obj.get("criteria").unwrap_or(&Value::Null);
    match t {
        "choice" => {
            let labels: Vec<&Value> = match crit {
                Value::Array(a) => a.iter().collect(),
                Value::Object(m) => m.keys().map(|_| &Value::Null).collect(), // str keys are always scalars
                _ => {
                    return Err(err(format!("question {}: a choice question takes 'criteria' as a dict of label -> description, \
                        or a list of labels", qrepr(qid))));
                }
            };
            if labels.is_empty() {
                return Err(err(format!("question {}: a choice question needs at least one criterion", qrepr(qid))));
            }
            for (i, l) in labels.iter().enumerate() {
                if matches!(l, Value::Array(_) | Value::Object(_)) {
                    return Err(err(format!("question {}: choice label {i} is a {}; a label is rendered as option text and used \
                        as the answer key, so it must be a scalar (a string, number or None), got {}",
                        qrepr(qid), py_type_name(l), py_repr(l))));
                }
            }
        }
        "score" => {
            let Some(a) = crit.as_array() else {
                return Err(err(format!("question {}: a score question takes 'criteria' as a list of level descriptions, \
                    index 0 first", qrepr(qid))));
            };
            if a.is_empty() {
                return Err(err(format!("question {}: a score question needs at least one level", qrepr(qid))));
            }
            if let Some(i) = a.iter().position(Value::is_null) {
                return Err(err(format!("question {}: score level {i} is null; give every level a description, index 0 first",
                    qrepr(qid))));
            }
        }
        _ => match crit {
            Value::Null => {}
            Value::Object(m) => {
                let mut keys: Vec<String> = m.keys().map(|k| k.to_lowercase()).collect();
                keys.sort();
                keys.dedup();
                if keys.iter().any(|k| !matches!(k.as_str(), "true" | "false")) {
                    let shown = format!("[{}]", keys.iter().map(|k| py_str_repr(k)).collect::<Vec<_>>().join(", "));
                    return Err(err(format!("question {}: a noul question takes 'criteria' keyed only 'true'/'false' (either or \
                        both, and omitted is fine), got {shown}. Those keys are the option texts the model reads; any other key \
                        was silently dropped and replaced with the defaults. If you want the answer worded differently, keep \
                        'criteria' keyed 'true'/'false' and set 'labels' instead.", qrepr(qid))));
                }
            }
            _ => {
                return Err(err(format!("question {}: a noul question takes 'criteria' as a dict with optional 'true'/'false' \
                    descriptions, or omits it", qrepr(qid))));
            }
        },
    }
    if let Some(labels) = obj.get("labels") {
        if t != "noul" {
            return Err(err(format!("question {}: 'labels' is only supported for noul questions", qrepr(qid))));
        }
        resolve_noul_labels(labels).map_err(|e| err(format!("question {}: {}", qrepr(qid), e.message)))?;
    }
    Ok(())
}

/// `_resolve_noul_labels` (laya/common.py:51-62): (false_label, true_label), stripped.
pub fn resolve_noul_labels(labels: &Value) -> Result<(String, String), LayaError> {
    let err = || LayaError::value("s01", "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
    let m = match labels {
        Value::Null => return Ok(("false".into(), "true".into())),
        Value::Object(m) => m,
        _ => return Err(err()),
    };
    if m.len() != 2 || !m.contains_key("false") || !m.contains_key("true") {
        return Err(err());
    }
    let (Some(Value::String(f)), Some(Value::String(t))) = (m.get("false"), m.get("true")) else {
        return Err(err());
    };
    let (f, t) = (py_strip(f), py_strip(t));
    if f.is_empty() || t.is_empty() || f == t {
        return Err(err());
    }
    Ok((f.to_string(), t.to_string()))
}

/// `_to_internal`.
pub fn to_internal(qid: &str, qdef: &Value) -> Internal {
    let obj = qdef.as_object().expect("validated");
    let t = match obj["type"].as_str().expect("validated") {
        "choice" => QType::Choice,
        "score" => QType::Score,
        _ => QType::Noul,
    };
    let crit_v = obj.get("criteria").cloned().unwrap_or(Value::Null);
    let crit = match t {
        QType::Choice => match crit_v {
            // {c: None for c in crit}: first occurrence keeps its position and object, later equal ones merge
            Value::Array(a) => {
                let mut seen: Vec<String> = Vec::new();
                let mut out: Vec<(Value, Value)> = Vec::new();
                for c in a {
                    let key = py_eq_key(&c).expect("validated scalar");
                    if !seen.contains(&key) {
                        seen.push(key);
                        out.push((c, Value::Null));
                    }
                }
                Crit::Choice(out)
            }
            Value::Object(m) => Crit::Choice(m.into_iter().map(|(k, v)| (Value::String(k), v)).collect()),
            _ => unreachable!("validated"),
        },
        QType::Score => Crit::Score(crit_v.as_array().cloned().expect("validated")),
        QType::Noul => match crit_v {
            // {str(k).lower(): v}: a later key equal after lowering overwrites the value in the first position
            Value::Object(m) => {
                let mut lowered = Map::new();
                for (k, v) in m {
                    lowered.insert(k.to_lowercase(), v);
                }
                Crit::Noul(Some(lowered))
            }
            _ => Crit::Noul(None),
        },
    };
    let ins = match &obj["instructions"] {
        Value::String(s) => s.clone(),
        other => dumps(other),
    };
    Internal { qid: qid.to_string(), t, ins, crit, labels: obj.get("labels").cloned() }
}
