//! Laya typed-decisions inference in Rust, one pure stage per module, writing SDC/1 dumps
//! (docs/architecture.md §5). The reference is the pinned Python laya; see docs/laya-model-analysis.md.

pub mod calibrate;
pub mod decide;
pub mod error;
pub mod format;
pub mod infer;
pub mod normalize;
pub mod npmath;
pub mod pyjson;
pub mod sdc;
pub mod sequence;
pub mod tokenize;

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::calibrate::Temperatures;
use crate::decide::{act_softmax, answer_confidence, argmax_first, confidence_from_probs, expected_score, ranking, round4,
    tempered_softmax};
use crate::error::LayaError;
use crate::format::{head_text, option_texts, render_options, state_text};
use crate::infer::Engine;
use crate::normalize::{check_question, to_internal, Crit, Internal, QType};
use crate::pyjson::{json_key, py_str, py_str_repr};
use crate::sdc::{f32v, f64v, write_json, write_safetensors, TData};
use crate::sequence::{build_sequence, collate, Row, Specials};
use crate::tokenize::Tok;

/// Model-side configuration read from the pinned snapshot.
pub struct Model {
    pub tok: Tok,
    pub temps: Temperatures,
    pub max_len: usize,
    pub head_max_len: usize,
}

impl Model {
    pub fn load(snapshot: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_slice(&std::fs::read(snapshot.join("rl_agent_config.json"))?)
            .context("rl_agent_config.json")?;
        let max_len = cfg.get("max_len").and_then(Value::as_u64).unwrap_or(512) as usize;
        let head_max_len = cfg.get("head_max_len").and_then(Value::as_u64).unwrap_or(192) as usize;
        Ok(Self { tok: Tok::load(snapshot)?, temps: Temperatures::from_config(&cfg), max_len, head_max_len })
    }
}

fn label_obj(v: &Value) -> Value {
    json!({"str": py_str(v), "json": v})
}

fn internal_entry(q: &Internal) -> Value {
    let crit = match &q.crit {
        Crit::Choice(items) => Value::Array(items.iter().map(|(k, d)| json!({"label": label_obj(k), "desc": d})).collect()),
        Crit::Score(levels) => Value::Array(levels.clone()),
        Crit::Noul(m) => m.clone().map(Value::Object).unwrap_or(Value::Null),
    };
    json!({"qid": q.qid, "t": q.t.name(), "ins": q.ins, "labels": q.labels.clone().unwrap_or(Value::Null), "crit": crit})
}

/// Runs one scenario (request bytes) and writes its SDC/1 directory. A request the reference would
/// reject is recorded in `status.json` (error parity), not returned as an error.
pub fn run_scenario(model: &Model, engine: &mut Engine, request: &[u8], out: &Path) -> Result<Value> {
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("request.json"), request)?;
    let req: Value = serde_json::from_slice(request).context("request.json (D-01/D-04: NaN/Infinity/lone surrogates unsupported)")?;
    match stages(model, engine, &req, Some(out))? {
        Ok(output) => {
            write_json(&out.join("status.json"), &json!({"ok": true}))?;
            Ok(json!({"input_tokens": output["usage"]["input_tokens"], "answers": output["answers"]}))
        }
        Err(e) => {
            let st = json!({"ok": false, "stage": e.stage, "error_kind": e.kind, "message": e.message});
            write_json(&out.join("status.json"), &st)?;
            Ok(st)
        }
    }
}

/// Answers one request the way `Agent.system_one(state, questions)` does, with no stage dumps:
/// `Ok(Ok(output))` is the reference-shaped `{model, answers, usage}`; `Ok(Err(e))` is the exception
/// the reference raises for this request (laya-serve maps a `ValueError` to HTTP 422).
pub fn predict(model: &Model, engine: &mut Engine, req: &Value) -> Result<Result<Value, LayaError>> {
    stages(model, engine, req, None)
}

/// Writes the SDC/1 stage files of one scenario, or nothing when `out` is `None` (predict).
struct Dump<'a>(Option<&'a Path>);

impl Dump<'_> {
    fn json(&self, name: &str, v: &Value) -> Result<()> {
        match self.0 {
            Some(out) => write_json(&out.join(name), v),
            None => Ok(()),
        }
    }

    fn safetensors(&self, name: &str, tensors: &[(&str, Vec<usize>, TData)]) -> Result<()> {
        match self.0 {
            Some(out) => write_safetensors(&out.join(name), tensors),
            None => Ok(()),
        }
    }
}

fn stages(model: &Model, engine: &mut Engine, req: &Value, out: Option<&Path>) -> Result<Result<Value, LayaError>> {
    let dump = Dump(out);
    let state = &req["state"];
    let questions = req["questions"].as_object().context("questions must be an object")?;

    // s01 — validate every question first, then normalize (predict_batch order, laya/agent.py:888-890)
    for (qid, qdef) in questions {
        if let Err(e) = check_question(qid, qdef) {
            return Ok(Err(e));
        }
    }
    let internal: Vec<Internal> = questions.iter().map(|(qid, qdef)| to_internal(qid, qdef)).collect();
    dump.json("s01_internal.json", &json!({"questions": internal.iter().map(internal_entry).collect::<Vec<_>>()}))?;
    if internal.is_empty() {
        // laya/agent.py predict_batch: no questions, no forward pass — an empty answer set with zero usage.
        let output = json!({"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}});
        dump.json("s10_output.json", &output)?;
        return Ok(Ok(output));
    }

    // s02 — texts
    let stext = state_text(state);
    let mut texts = Vec::new();
    for q in &internal {
        match option_texts(q) {
            Ok(opts) => texts.push((head_text(q), opts)),
            Err(e) => return Ok(Err(LayaError { stage: "s02", ..e })),
        }
    }
    dump.json("s02_texts.json", &json!({
        "state_text": stext,
        "state_sha256": format!("{:x}", Sha256::digest(stext.as_bytes())),
        "questions": internal.iter().zip(&texts).map(|(q, (h, o))| json!({"qid": q.qid, "head_text": h, "option_texts": o}))
            .collect::<Vec<_>>()}))?;

    // s03 — tokens and sequences
    let tok = &model.tok;
    let sp = Specials { cls: tok.cls, sep: tok.sep, mask: tok.mask };
    let truncate_left = state.is_array();
    let state_ids = tok.encode(&stext)?;
    let mut rows: Vec<Row> = Vec::new();
    let mut s03_rows = Vec::new();
    for (q, (head, opts)) in internal.iter().zip(&texts) {
        let head_full = tok.encode(head)?;
        let opt_full: Vec<Vec<i64>> = opts.iter().map(|o| tok.encode(o)).collect::<Result<_>>()?;
        let (ids, markers) = build_sequence(&sp, &head_full, &opt_full, &state_ids, model.max_len, model.head_max_len, truncate_left);
        let (prefix, _) = build_sequence(&sp, &head_full, &opt_full, &[], model.max_len, model.head_max_len, truncate_left);
        let n_opts = render_options(q).map(|o| o.len()).unwrap_or(0);
        if markers.len() != n_opts {
            return Ok(Err(LayaError::value("s03", format!("question {} options exceed head_max_len={}", py_str_repr(&q.qid), model.head_max_len))));
        }
        s03_rows.push(json!({"qid": q.qid, "head_ids_full": head_full, "option_ids_full": opt_full, "ids": ids,
            "markers": markers, "len": ids.len(), "prefix_len": prefix.len() - 1, "qtype": q.t.code()}));
        rows.push(Row { prefix_len: prefix.len() - 1, ids, markers, qtype: q.t });
    }

    // s04 — batch
    let b = collate(&rows, tok.pad);
    let input_tokens: i64 = b.attention_mask.iter().sum();
    dump.json("s03_tokens.json", &json!({"max_len": model.max_len, "head_max_len": model.head_max_len,
        "truncate_left": truncate_left, "state_ids": state_ids, "questions": s03_rows, "input_tokens": input_tokens}))?;
    let mm: Vec<u8> = b.marker_mask.iter().map(|&m| u8::from(m)).collect();
    dump.safetensors("s04_batch.safetensors", &[
        ("input_ids", vec![b.n, b.l], TData::I64(&b.input_ids)),
        ("attention_mask", vec![b.n, b.l], TData::I64(&b.attention_mask)),
        ("marker_pos", vec![b.n, b.kmax], TData::I64(&b.marker_pos)),
        ("marker_mask", vec![b.n, b.kmax], TData::U8(&mm)),
        ("qtype", vec![b.n], TData::I64(&b.qtype)),
    ])?;

    // s05/s06 — ONNX Runtime
    let o = engine.run(&b)?;
    dump.safetensors("s05_encoder.safetensors",
        &[("last_hidden_state", vec![b.n, b.l, o.hidden_dim], TData::F32(&o.hidden))])?;
    dump.safetensors("s06_head.safetensors", &[
        ("logits", vec![b.n, b.kmax], TData::F32(&o.logits)),
        ("act_logits", vec![b.n, 2], TData::F32(&o.act_logits)),
    ])?;

    // s07..s10 — calibration, probabilities, decisions, output
    let mut calib = Vec::new();
    let mut decisions = Vec::new();
    let mut probs: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut act_p_all: Vec<f32> = Vec::with_capacity(b.n * 2);
    let mut answers = Map::new();
    for (r, q) in internal.iter().enumerate() {
        let k = rows[r].markers.len();
        let lk = model.temps.lookup(q.t, k);
        calib.push(json!({"qid": q.qid, "qtype": q.t.code(), "k": k, "bucket": lk.bucket, "t_source": lk.source,
            "t_raw": f64v(lk.t_raw), "t": f64v(lk.t)}));

        let row_logits = &o.logits[r * b.kmax..r * b.kmax + k];
        let (z, p) = tempered_softmax(row_logits, lk.t);
        let act_p = act_softmax(&o.act_logits[r * 2..r * 2 + 2]);
        act_p_all.extend_from_slice(&act_p);
        let order = ranking(&p);
        let am = argmax_first(&p);
        let ans_conf = answer_confidence(&p);
        let mut d = Map::new();
        d.insert("qid".into(), json!(q.qid));
        d.insert("type".into(), json!(q.t.name()));
        d.insert("k".into(), json!(k));
        d.insert("argmax".into(), json!(am));
        d.insert("ranking".into(), json!(order));
        d.insert("p_dtype".into(), json!("float32"));
        d.insert("z_dtype".into(), json!("float32"));
        d.insert("margin_top2".into(), if k >= 2 { f64v(p[order[0]] as f64 - p[order[1]] as f64) } else { Value::Null });
        d.insert("answer_confidence_raw".into(), f64v(ans_conf));
        d.insert("act_p0_raw".into(), f32v(act_p[0]));

        let action = json!({"act_probability": round4(act_p[0] as f64)});
        let answer = match &q.crit {
            Crit::Choice(items) => {
                let conf = confidence_from_probs(&p);
                d.insert("selected".into(), label_obj(&items[am].0));
                d.insert("confidence_raw".into(), f64v(conf));
                let mut pm = Map::new();
                for ((label, _), &pv) in items.iter().zip(&p) {
                    pm.insert(json_key(label), json!(round4(pv as f64)));
                }
                json!({"type": "choice", "choice": items[am].0, "probabilities": pm, "confidence": round4(conf),
                    "answer_confidence": round4(ans_conf), "action": action})
            }
            Crit::Score(levels) => {
                let conf = confidence_from_probs(&p);
                let score = expected_score(&p);
                d.insert("score_raw".into(), f64v(score));
                d.insert("confidence_raw".into(), f64v(conf));
                let legend: Map<String, Value> = levels.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect();
                let pm: Map<String, Value> = p.iter().enumerate().map(|(i, &pv)| (i.to_string(), json!(round4(pv as f64)))).collect();
                json!({"type": "score", "score": round4(score), "legend": legend, "probabilities": pm,
                    "confidence": round4(conf), "answer_confidence": round4(ans_conf), "action": action})
            }
            Crit::Noul(_) => {
                let p1 = p[1] as f64;
                let conf = p1.max(1.0 - p1);
                d.insert("noul_raw".into(), f64v(p1));
                d.insert("confidence_raw".into(), f64v(conf));
                json!({"type": "noul", "noul": round4(p1), "confidence": round4(conf),
                    "answer_confidence": round4(ans_conf), "action": action})
            }
        };
        answers.insert(q.qid.clone(), answer);
        decisions.push(Value::Object(d));
        probs.push((format!("r{r}.z"), vec![k], z));
        probs.push((format!("r{r}.p"), vec![k], p));
    }
    dump.json("s07_calibration.json", &json!({"questions": calib}))?;
    let mut tensors: Vec<(&str, Vec<usize>, TData)> =
        probs.iter().map(|(n, s, v)| (n.as_str(), s.clone(), TData::F32(v))).collect();
    tensors.push(("act_p", vec![b.n, 2], TData::F32(&act_p_all)));
    dump.safetensors("s08_probs.safetensors", &tensors)?;
    dump.json("s09_decision.json", &json!({"questions": decisions}))?;
    let output = json!({"model": "laya-rl-agent", "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0}});
    dump.json("s10_output.json", &output)?;
    let _ = QType::Choice;
    Ok(Ok(output))
}
