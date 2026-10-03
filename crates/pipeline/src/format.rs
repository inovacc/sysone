//! s02: text rendering — `serialize_state`, `render_criterion`, `render_options` (laya/common.py:33-91),
//! the head text of `build_sequence` (laya/common.py:112-113) and the `[MASK]` scrub.

use serde_json::Value;

use crate::error::LayaError;
use crate::normalize::{resolve_noul_labels, Crit, Internal};
use crate::pyjson::{dumps, py_str};

pub const MASK_TOKEN: &str = "[MASK]";

/// `serialize_state`: a str passes through unchanged; anything else is `json.dumps(ensure_ascii=False)`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => dumps(other),
    }
}

/// `render_criterion`: str as is, else `json.dumps(v, ensure_ascii=False, separators=(", ", ": "))`.
pub fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => dumps(other),
    }
}

fn is_none_or_empty(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null)) || matches!(v, Some(Value::String(s)) if s.is_empty())
}

/// `render_options`: option texts in label-index order; noul is always [false, true].
pub fn render_options(q: &Internal) -> Result<Vec<String>, LayaError> {
    if !matches!(q.crit, Crit::Noul(_)) && q.labels.is_some() {
        return Err(LayaError::value("s01", "labels is only supported for noul questions"));
    }
    Ok(match &q.crit {
        Crit::Choice(items) => items
            .iter()
            .map(|(k, v)| {
                if is_none_or_empty(Some(v)) {
                    py_str(k)
                } else {
                    format!("{}: {}", py_str(k), render_criterion(v))
                }
            })
            .collect(),
        Crit::Score(levels) => levels.iter().enumerate().map(|(i, c)| format!("level {i}: {}", render_criterion(c))).collect(),
        Crit::Noul(crit) => {
            let (fl, tl) = resolve_noul_labels(q.labels.as_ref().unwrap_or(&Value::Null))?;
            let get = |k: &str| crit.as_ref().and_then(|m| m.get(k));
            let f = get("false");
            let t = get("true");
            vec![
                format!("{fl}: {}", if is_none_or_empty(f) { "no, the statement does not hold".to_string() } else { render_criterion(f.unwrap()) }),
                format!("{tl}: {}", if is_none_or_empty(t) { "yes, the statement holds".to_string() } else { render_criterion(t.unwrap()) }),
            ]
        }
    })
}

/// `"%s question: %s" % (t, str(ins).replace(mask, " "))`.
pub fn head_text(q: &Internal) -> String {
    format!("{} question: {}", q.t.name(), q.ins.replace(MASK_TOKEN, " "))
}

/// `" " + opt.replace(mask, " ")`.
pub fn option_texts(q: &Internal) -> Result<Vec<String>, LayaError> {
    Ok(render_options(q)?.into_iter().map(|o| format!(" {}", o.replace(MASK_TOKEN, " "))).collect())
}

/// `serialize_state(state).replace(mask, " ")`.
pub fn state_text(state: &Value) -> String {
    serialize_state(state).replace(MASK_TOKEN, " ")
}
