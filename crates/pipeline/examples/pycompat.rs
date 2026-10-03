//! V-13 / V-14: check pyjson + round4 against a CPython-generated corpus (.scripts/20-B_pycompat_cases.py).
//!   cargo run --release --example pycompat -- <cases.json>

use pipeline::decide::round4;
use pipeline::pyjson::{dumps, float_repr, json_key, py_str};
use serde_json::Value;

fn f(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits, 16).expect("hex bits"))
}

fn main() {
    let path = std::env::args().nth(1).expect("cases.json");
    let cases: Value = serde_json::from_slice(&std::fs::read(path).expect("read")).expect("parse");
    let mut report = Vec::new();

    let (mut n, mut bad) = (0, Vec::new());
    for c in cases["round4"].as_array().unwrap() {
        let (x, want) = (f(c[0].as_str().unwrap()), f(c[1].as_str().unwrap()));
        n += 1;
        if round4(x).to_bits() != want.to_bits() {
            bad.push(format!("round4({x:?}) = {:?}, CPython {want:?}", round4(x)));
        }
    }
    report.push(("round4", n, bad));

    let (mut n, mut bad) = (0, Vec::new());
    for c in cases["repr"].as_array().unwrap() {
        let (x, want) = (f(c[0].as_str().unwrap()), c[1].as_str().unwrap());
        n += 1;
        let got = float_repr(x, false);
        if got != want {
            bad.push(format!("repr bits {} = {got}, CPython {want}", c[0]));
        }
    }
    report.push(("float_repr", n, bad));

    let (mut n, mut bad) = (0, Vec::new());
    for c in cases["dumps"].as_array().unwrap() {
        let (doc, want) = (c[0].as_str().unwrap(), c[1].as_str().unwrap());
        n += 1;
        match serde_json::from_str::<Value>(doc) {
            Ok(v) if dumps(&v) == want => {}
            Ok(v) => bad.push(format!("dumps({doc}) = {}, CPython {want}", dumps(&v))),
            Err(e) => bad.push(format!("parse({doc}): {e}")),
        }
    }
    report.push(("json_dumps", n, bad));

    let (mut n, mut bad) = (0, Vec::new());
    for c in cases["labels"].as_array().unwrap() {
        let v: Value = serde_json::from_str(c[0].as_str().unwrap()).unwrap();
        n += 1;
        if py_str(&v) != c[1].as_str().unwrap() || json_key(&v) != c[2].as_str().unwrap() {
            bad.push(format!("label {}: str={} key={} want str={} key={}", c[0], py_str(&v), json_key(&v), c[1], c[2]));
        }
    }
    report.push(("labels", n, bad));

    println!("CPython reference: {}", cases["python"]);
    let mut total_bad = 0;
    for (name, n, bad) in &report {
        println!("{name:<11} {n:>7} cases  {:>5} mismatches", bad.len());
        for b in bad.iter().take(5) {
            println!("    {b}");
        }
        total_bad += bad.len();
    }
    std::process::exit(if total_bad == 0 { 0 } else { 1 });
}
