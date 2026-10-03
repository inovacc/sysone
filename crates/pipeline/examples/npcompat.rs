//! V-15: check `npmath` + the decode math bit-for-bit against a NumPy-generated corpus (.scripts/31-B_numpy_corpus.py).
//!   cargo run --release --example npcompat -- <corpus.json>

use pipeline::decide::{confidence_from_probs, expected_score, tempered_softmax};
use pipeline::npmath::{expf, logf};
use serde_json::Value;

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|h| f32::from_bits(u32::from_str_radix(h.as_str().unwrap(), 16).unwrap())).collect()
}

fn f64h(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap())
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn main() {
    let path = std::env::args().nth(1).expect("corpus.json");
    let c: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    println!("NumPy reference {}", c["numpy"]);
    let mut fails = 0;

    for (name, f, inp, out) in [("exp", expf as fn(f32) -> f32, "exp_in", "exp_out"), ("log", logf, "log_in", "log_out")] {
        let (xs, ys) = (f32s(&c[inp]), f32s(&c[out]));
        let bad: Vec<_> = xs.iter().zip(&ys).filter(|(x, y)| !same(f(**x), **y)).collect();
        println!("{name:<6} {:>7} inputs  {:>6} mismatches", xs.len(), bad.len());
        for (x, y) in bad.iter().take(5) {
            println!("    {name}({x:e}) = {:e}, numpy {y:e}", f(**x));
        }
        fails += bad.len();
    }

    let (mut n, mut bad_p, mut bad_c, mut bad_s) = (0, 0, 0, 0);
    for d in c["decode"].as_array().unwrap() {
        n += 1;
        let (logits, t) = (f32s(&d["logits"]), f64h(&d["t"]));
        let (_, p) = tempered_softmax(&logits, t);
        if !p.iter().zip(f32s(&d["p"])).all(|(a, b)| same(*a, b)) {
            bad_p += 1;
        }
        if confidence_from_probs(&p).to_bits() != f64h(&d["conf"]).to_bits() {
            bad_c += 1;
        }
        if expected_score(&p).to_bits() != f64h(&d["score"]).to_bits() {
            bad_s += 1;
        }
    }
    println!("decode {n:>7} vectors  p mismatches {bad_p}, confidence {bad_c}, score {bad_s}");
    fails += bad_p + bad_c + bad_s;
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
