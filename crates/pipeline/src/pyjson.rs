//! Python-compatible JSON semantics: what `json.loads` produces and what `json.dumps(ensure_ascii=False)`,
//! `str()` and dict-key equality do with it. The reference serializes states and criteria with
//! Python's `json.dumps` (`laya/common.py:33-48`), so every byte here reaches the tokenizer.
//!
//! Values are `serde_json::Value` parsed with `preserve_order` (insertion order, and a duplicate key
//! keeps its first position with the last value, like `json.loads`) and `arbitrary_precision` (each
//! number keeps its literal, so int vs float and big ints survive).
//!
//! Named deviations (flag, don't fake):
//! - D-01: `NaN`/`Infinity` literals are rejected by serde_json; Python accepts them.
//! - D-04: lone UTF-16 surrogates (`"\ud800"`) are rejected by serde_json; Python keeps them.

use serde_json::{Number, Value};

/// A JSON number as Python sees it after `json.loads`.
pub enum PyNum {
    /// Python `int`, as its normalized decimal text (exact, arbitrary size).
    Int(String),
    /// Python `float`.
    Float(f64),
}

/// Classifies a number literal the way `json.loads` does: no `.`/`e`/`E` means `int`.
pub fn py_num(n: &Number) -> PyNum {
    let lit = n.to_string();
    if lit.contains(['.', 'e', 'E']) {
        PyNum::Float(lit.parse::<f64>().unwrap_or(f64::NAN))
    } else if lit.trim_start_matches('-').bytes().all(|b| b == b'0') {
        PyNum::Int("0".to_string()) // json.loads("-0") -> 0
    } else {
        PyNum::Int(lit)
    }
}

/// `repr(float)` for finite values (Python's shortest round-trip repr with its notation switch).
/// `json_style` selects `NaN`/`Infinity` (json.dumps) instead of `nan`/`inf` (repr/str).
pub fn float_repr(x: f64, json_style: bool) -> String {
    if x.is_nan() {
        return if json_style { "NaN".into() } else { "nan".into() };
    }
    if x.is_infinite() {
        let s = if json_style { "Infinity" } else { "inf" };
        return if x < 0.0 { format!("-{s}") } else { s.into() };
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    // CPython's repr is dtoa mode 0: the shortest digits that round-trip and, among those, the closest
    // to the exact value. Ryu guarantees the same; Rust's `{:e}` does not always pick the closest
    // (42 of 151,249 random doubles differed in the V-13 corpus, .scripts/21-B_pycompat_check.out.txt).
    let (digits, exp) = shortest_digits(x.abs());
    let body = if (-4..16).contains(&exp) {
        let decpt = exp + 1; // digits before the decimal point
        if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if decpt as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(decpt as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
        }
    } else {
        let frac = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{}{}e{}{:02}", &digits[..1], frac, esign, exp.abs())
    };
    format!("{sign}{body}")
}

/// Shortest round-trip digits of a finite positive double and its decimal exponent (value = d.ddd × 10^exp).
fn shortest_digits(x: f64) -> (String, i32) {
    let mut buf = ryu::Buffer::new();
    let s = buf.format_finite(x); // forms: "123.45", "1e16", "1.5e-7", "0.0001"
    let (mant, exp10) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().expect("ryu exponent")),
        None => (s, 0),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let all: String = format!("{int_part}{frac_part}");
    let lead = all.len() - all.trim_start_matches('0').len();
    let digits = all.trim_start_matches('0').trim_end_matches('0').to_string();
    // position of the first significant digit relative to the decimal point
    let exp = exp10 + int_part.len() as i32 - 1 - lead as i32;
    (if digits.is_empty() { "0".into() } else { digits }, exp)
}

/// Python `json.dumps(v, ensure_ascii=False)` with the default separators `", "` and `": "`.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, (", ", ": "));
    out
}

/// `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`: the bytes FastAPI's `JSONResponse`
/// puts on the wire, so a sysone response can be compared byte for byte with laya-serve's.
pub fn dumps_compact(v: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, v, (",", ":"));
    out
}

fn write_value(out: &mut String, v: &Value, sep: (&str, &str)) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match py_num(n) {
            PyNum::Int(s) => out.push_str(&s),
            PyNum::Float(f) => out.push_str(&float_repr(f, true)),
        },
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep.0);
                }
                write_value(out, x, sep);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep.0);
                }
                write_str(out, k);
                out.push_str(sep.1);
                write_value(out, x, sep);
            }
            out.push('}');
        }
    }
}

/// `py_encode_basestring`: escape `"`, `\`, and U+0000..U+001F only (lowercase `\u00xx`).
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `str(x)` of a JSON scalar (labels are rendered with `str()` / `%s`).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::Number(n) => match py_num(n) {
            PyNum::Int(s) => s,
            PyNum::Float(f) => float_repr(f, false),
        },
        Value::String(s) => s.clone(),
        other => dumps(other), // unreachable for validated labels
    }
}

/// Python `repr(x)` of a `json.loads` value, as it appears in laya's error messages (`%r`): `'str'` with
/// Python's quote choice and escapes, `None`/`True`/`False`, ints, float repr, `[a, b]`, `{'k': v}`.
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => py_str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{}: {}", py_str_repr(k), py_repr(x)))
            .collect::<Vec<_>>().join(", ")),
        scalar => py_str(scalar),
    }
}

/// `repr(str)`: single quotes unless the text has a `'` and no `"`; `\\`, `\n`, `\r`, `\t`, the chosen
/// quote and other control characters (`\xNN`) escaped; printable non-ASCII kept.
pub fn py_str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// `type(x).__name__` of a `json.loads` value.
pub fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => match py_num(n) {
            PyNum::Int(_) => "int",
            PyNum::Float(_) => "float",
        },
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The key `json.dumps` writes for a Python dict key (labels become answer keys):
/// str as is, int as decimal, float via repr, True/False/None as true/false/null.
pub fn json_key(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::Number(n) => match py_num(n) {
            PyNum::Int(s) => s,
            PyNum::Float(f) => float_repr(f, true),
        },
        Value::String(s) => s.clone(),
        other => dumps(other),
    }
}

/// Identity of a hashable JSON scalar under Python `==`/`hash`: `1 == True == 1.0`, `0 == False == -0.0`,
/// `"1" != 1`, None only equals None. `None` for unhashable values (arrays, objects).
pub fn py_eq_key(v: &Value) -> Option<String> {
    Some(match v {
        Value::Null => "N".into(),
        Value::Bool(b) => format!("I{}", u8::from(*b)),
        Value::Number(n) => match py_num(n) {
            PyNum::Int(s) => format!("I{s}"),
            PyNum::Float(f) if f.is_finite() && f.fract() == 0.0 => {
                let s = format!("{:.0}", f); // exact integer digits
                format!("I{}", if s == "-0" { "0" } else { &s })
            }
            PyNum::Float(f) => format!("F{:016x}", f.to_bits()),
        },
        Value::String(s) => format!("S{s}"),
        Value::Array(_) | Value::Object(_) => return None,
    })
}

/// Python `str.strip()` (its whitespace set differs from Rust's `trim`: U+001C..U+001F are included).
pub fn py_strip(s: &str) -> &str {
    let ws = |c: char| {
        matches!(c, '\t' | '\n' | '\u{0b}' | '\u{0c}' | '\r' | '\u{1c}'..='\u{1f}' | ' ' | '\u{85}' | '\u{a0}'
            | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
    };
    s.trim_matches(ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn float_repr_matches_cpython() {
        // expected strings are CPython repr() outputs (probe recorded in docs/laya-model-analysis.md §3.1)
        for (x, want) in [(1e16, "1e+16"), (1e15, "1000000000000000.0"), (100.0, "100.0"), (1e-4, "0.0001"),
            (1e-5, "1e-05"), (5e-324, "5e-324"), (0.30000000000000004, "0.30000000000000004"), (-0.0, "-0.0"),
            (1.5, "1.5"), (123456789.123, "123456789.123"), (1.5e300, "1.5e+300"), (2.5e-7, "2.5e-07")] {
            assert_eq!(float_repr(x, true), want, "{x}");
        }
    }

    #[test]
    fn dumps_matches_json_dumps() {
        let src = r#"{"s": "é \u0001\u001f\"\\/\n\t\r\b\f", "f": [1.0, 100.0, 1e16, 1e15, 0.0001, 1e-05, -0.0, 1.50],
            "i": [12345678901234567890123, -0, 0], "a": 1e2, "a": "dup", "t": true, "n": null}"#;
        let want = "{\"s\": \"é\u{2028}\\u0001\\u001f\\\"\\\\/\\n\\t\\r\\b\\f\", \"f\": [1.0, 100.0, 1e+16, \
                    1000000000000000.0, 0.0001, 1e-05, -0.0, 1.5], \"i\": [12345678901234567890123, 0, 0], \
                    \"a\": \"dup\", \"t\": true, \"n\": null}";
        assert_eq!(dumps(&v(src)), want);
    }

    #[test]
    fn python_equality_keys() {
        assert_eq!(py_eq_key(&v("1")), py_eq_key(&v("true")));
        assert_eq!(py_eq_key(&v("1")), py_eq_key(&v("1.0")));
        assert_ne!(py_eq_key(&v("1")), py_eq_key(&v("\"1\"")));
        assert_eq!(py_eq_key(&v("0")), py_eq_key(&v("-0.0")));
        assert_eq!(py_str(&v("true")), "True");
        assert_eq!(py_str(&v("null")), "None");
        assert_eq!(json_key(&v("true")), "true");
    }
}
