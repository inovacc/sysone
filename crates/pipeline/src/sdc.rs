//! SDC/1 writers (docs/parity-definition.md §2): exact floats as {"repr","bits"}, JSON stage files,
//! and a minimal safetensors writer (little-endian, names sorted).

use std::path::Path;

use anyhow::Result;
use serde_json::{json, Value};

pub fn f64v(x: f64) -> Value {
    json!({"repr": format!("{x:?}"), "bits": format!("0x{:016x}", x.to_bits())})
}

pub fn f32v(x: f32) -> Value {
    json!({"repr": format!("{:?}", x as f64), "bits": format!("0x{:08x}", x.to_bits())})
}

pub fn write_json(path: &Path, v: &Value) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut s = serde_json::to_string_pretty(v)?;
    s.push('\n');
    std::fs::write(path, s)?;
    Ok(())
}

pub enum TData<'a> {
    F32(&'a [f32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
}

pub fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, TData)]) -> Result<()> {
    let mut sorted: Vec<&(&str, Vec<usize>, TData)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({"sdc": "1"}));
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, t) in sorted {
        let start = data.len();
        let dtype = match t {
            TData::F32(v) => {
                v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes()));
                "F32"
            }
            TData::I64(v) => {
                v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes()));
                "I64"
            }
            TData::U8(v) => {
                data.extend_from_slice(v);
                "U8"
            }
        };
        header.insert((*name).to_string(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + h.len() + data.len());
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, out)?;
    Ok(())
}
