//! `sysone`: the Rust candidate CLI (docs/architecture.md §5).
//!
//!   sysone batch --model <snapshot> --bundle <onnx-dir> --scenarios <dir> --out <sdc-root>
//!                 [--threads N] [--ort-lib <libonnxruntime.so>] [--opt all|extended|basic|disable] [--no-deterministic]
//!   sysone tokenize --model <snapshot> --in <corpus.jsonl> --out <ids.jsonl>
//!   sysone serve --model <snapshot> --bundle <onnx-dir> [--addr 127.0.0.1:8000] [--threads N] [--ort-lib PATH]
//!                 (bearer auth when LAYA_API_KEY is set, as laya-serve)
//!   sysone version

mod serve;

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use pipeline::infer::{init_runtime, Engine, SessionOptions};
use pipeline::sdc::write_json;
use pipeline::tokenize::Tok;
use pipeline::{run_scenario, Model};
use serde_json::json;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("batch") => batch(&args[1..]),
        Some("tokenize") => tokenize(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("version") => {
            println!("{}", serde_json::to_string_pretty(&version())?);
            Ok(())
        }
        _ => bail!("usage: sysone batch --model DIR --bundle DIR --scenarios DIR --out DIR [--threads N] [--ort-lib PATH] \
                    | serve --model DIR --bundle DIR [--addr HOST:PORT] [--threads N] [--ort-lib PATH] \
                    | tokenize --model DIR --in FILE --out FILE | version"),
    }
}

fn serve(args: &[String]) -> Result<()> {
    let model_dir = path_or_beside(args, "--model", Some("SYSONE_MODEL"), "model")?;
    let bundle = path_or_beside(args, "--bundle", Some("SYSONE_BUNDLE"), "bundle")?;
    let threads: usize = flag(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let ort_lib = path_or_beside(args, "--ort-lib", Some("ORT_DYLIB_PATH"), ORT_LIB_NAME)?;
    let opts = SessionOptions { intra_threads: threads, optimization: opt_level(args), deterministic: true };
    let cfg = serve::Config {
        addr: flag(args, "--addr").unwrap_or_else(|| "127.0.0.1:8000".into()),
        api_key: std::env::var("LAYA_API_KEY").ok().filter(|k| !k.is_empty()),
        model_name: "typed-decisions".into(),
    };
    init_runtime(&ort_lib)?;
    let model = Model::load(&model_dir)?;
    let mut engine = Engine::load(&bundle, &opts)?;
    eprintln!("sysone serve: model {} bundle {} ort {} threads {}", model_dir.display(), bundle.display(), ort_lib.display(), threads);
    serve::run(&model, &mut engine, &cfg)
}

/// Tokenizer differential for V-11: one JSON line per corpus line, `{"i", "ids", "ids_sp"}` where `ids` encodes
/// the text and `ids_sp` encodes `" " + text` (the option form of `build_sequence`), both without special tokens.
fn tokenize(args: &[String]) -> Result<()> {
    let need = |n: &str| flag(args, n).with_context(|| format!("missing {n}"));
    let tok = Tok::load(&PathBuf::from(need("--model")?))?;
    let input = std::io::BufReader::new(std::fs::File::open(need("--in")?)?);
    let mut out = std::io::BufWriter::new(std::fs::File::create(need("--out")?)?);
    let mut n = 0usize;
    for (i, line) in input.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(&line?).with_context(|| format!("corpus line {i}"))?;
        let text = v["text"].as_str().with_context(|| format!("corpus line {i}: text"))?;
        let rec = json!({"i": i, "ids": tok.encode(text)?, "ids_sp": tok.encode(&format!(" {text}"))?});
        writeln!(out, "{rec}")?;
        n += 1;
    }
    out.flush()?;
    eprintln!("tokenized {n} lines");
    Ok(())
}

fn version() -> serde_json::Value {
    json!({
        "sysone": env!("CARGO_PKG_VERSION"),
        "git_commit": option_env!("LAB_GIT_COMMIT").unwrap_or("unknown"),
        "tokenizers": "0.23.2",
        "ort": "2.0.0-rc.13",
        "ort_api": 27,
    })
}

/// A path flag with a fallback next to the executable, so a self-contained distribution
/// (`sysone.exe`, `onnxruntime.dll`, `model/`, `bundle/` in one directory) runs with no flags.
fn path_or_beside(args: &[String], name: &str, env: Option<&str>, beside: &str) -> Result<PathBuf> {
    if let Some(p) = flag(args, name) {
        return Ok(PathBuf::from(p));
    }
    if let Some(v) = env.and_then(|e| std::env::var(e).ok()).filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(v));
    }
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    match exe_dir.map(|d| d.join(beside)).filter(|p| p.exists()) {
        Some(p) => Ok(p),
        None => bail!("missing {name} (no {beside} next to the executable{})",
            env.map(|e| format!(", {e} unset")).unwrap_or_default()),
    }
}

#[cfg(windows)]
const ORT_LIB_NAME: &str = "onnxruntime.dll";
#[cfg(not(windows))]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

/// `--opt all|extended|basic|disable` (default `all`, the refonnx leg's setting).
fn opt_level(args: &[String]) -> String {
    match flag(args, "--opt").as_deref() {
        Some(l @ ("disable" | "basic" | "extended" | "all")) => l.to_string(),
        Some(other) => {
            eprintln!("sysone: unknown --opt {other:?}, using all");
            "all".into()
        }
        None => "all".into(),
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn batch(args: &[String]) -> Result<()> {
    let need = |n: &str| flag(args, n).with_context(|| format!("missing {n}"));
    let model_dir = path_or_beside(args, "--model", Some("SYSONE_MODEL"), "model")?;
    let bundle = path_or_beside(args, "--bundle", Some("SYSONE_BUNDLE"), "bundle")?;
    let scen = PathBuf::from(need("--scenarios")?);
    let out = PathBuf::from(need("--out")?);
    let threads: usize = flag(args, "--threads").map(|s| s.parse()).transpose()?.unwrap_or(1);
    let ort_lib = path_or_beside(args, "--ort-lib", Some("ORT_DYLIB_PATH"), ORT_LIB_NAME)?;
    let opts = SessionOptions {
        intra_threads: threads,
        optimization: opt_level(args),
        deterministic: !args.iter().any(|a| a == "--no-deterministic"),
    };

    let started = Instant::now();
    init_runtime(&ort_lib)?;
    let model = Model::load(&model_dir)?;
    let mut engine = Engine::load(&bundle, &opts)?;
    let mut ids: Vec<_> = std::fs::read_dir(&scen)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("request.json").is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    ids.sort();
    let mut summaries = Vec::new();
    for id in &ids {
        let t0 = Instant::now();
        let req = std::fs::read(scen.join(id).join("request.json"))?;
        let s = run_scenario(&model, &mut engine, &req, &out.join("scenarios").join(id))?;
        println!("{id} {:.1}s {}", t0.elapsed().as_secs_f64(), s);
        summaries.push(json!({"id": id, "seconds": t0.elapsed().as_secs_f64()}));
    }
    write_json(&out.join("run.json"), &json!({
        "sdc": "1", "producer": "candidate", "version": version(), "model_dir": model_dir, "bundle": bundle,
        "ort_lib": ort_lib, "session": {"intra_threads": threads, "inter_threads": 1, "sequential": true,
            "optimization": opts.optimization, "deterministic_compute": opts.deterministic,
            "cpu_mem_arena": false},
        "scenarios": summaries, "seconds": started.elapsed().as_secs_f64(),
    }))?;
    Ok(())
}
