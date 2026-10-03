//! s05/s06: the lab's split ONNX contract (docs/architecture.md §4) on ONNX Runtime 1.28 via `ort`
//! with `load-dynamic`, CPU EP only, pinned session options.

use std::path::Path;

use anyhow::{anyhow, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::sequence::Batch;

pub struct SessionOptions {
    pub intra_threads: usize,
    /// ORT graph optimization level: `disable`, `basic` (constant folding, redundant nodes), `extended` (ORT's
    /// fused contrib ops: attention, layer norm, GELU), `all` (plus layout optimizations). The fusions change
    /// which kernels run and so the float32 rounding; the level is measured against the torch reference.
    pub optimization: String,
    pub deterministic: bool,
}

impl SessionOptions {
    pub fn optimization_all(&self) -> bool {
        self.optimization == "all"
    }
}

pub struct Engine {
    encoder: Session,
    head: Session,
}

pub struct Outputs {
    /// [N, L, H]
    pub hidden: Vec<f32>,
    pub hidden_dim: usize,
    /// [N, K]
    pub logits: Vec<f32>,
    /// [N, 2]
    pub act_logits: Vec<f32>,
}

/// Loads the ORT shared library once per process (`ORT_DYLIB_PATH` or an explicit path).
pub fn init_runtime(lib: &Path) -> Result<()> {
    ort::init_from(lib).map_err(|e| anyhow!("load {}: {e}", lib.display()))?.commit();
    Ok(())
}

/// Session options shared with the refonnx producer (oracle/harness/run.py), so both legs run the same graph
/// the same way. The CPU memory arena is off: with it, a B=5 × L=1024 batch (v003 T09) needs more than the
/// 4.5 GB container, and turning it off changes no output bit (measured, `.scripts/53-B_ort_memory.out.txt`).
/// Weight prepacking stays on because turning it off does change the outputs.
fn session(path: &Path, o: &SessionOptions) -> ort::Result<Session> {
    let level = match o.optimization.as_str() {
        "disable" => GraphOptimizationLevel::Disable,
        "basic" => GraphOptimizationLevel::Level1,
        "extended" => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::All,
    };
    Ok(Session::builder()?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])?
        .with_intra_threads(o.intra_threads)?
        .with_inter_threads(1)?
        .with_parallel_execution(false)?
        .with_optimization_level(level)?
        .with_deterministic_compute(o.deterministic)?
        .commit_from_file(path)?)
}

impl Engine {
    pub fn load(bundle: &Path, o: &SessionOptions) -> Result<Self> {
        let enc = session(&bundle.join("encoder.onnx"), o).map_err(|e| anyhow!("encoder.onnx: {e}"))?;
        let head = session(&bundle.join("head.onnx"), o).map_err(|e| anyhow!("head.onnx: {e}"))?;
        Ok(Self { encoder: enc, head })
    }

    pub fn run(&mut self, b: &Batch) -> Result<Outputs> {
        self.run_inner(b).map_err(|e| anyhow!("onnxruntime: {e}"))
    }

    fn run_inner(&mut self, b: &Batch) -> ort::Result<Outputs> {
        let (n, l, k) = (b.n as i64, b.l as i64, b.kmax as i64);
        let enc_out = self.encoder.run(ort::inputs![
            "input_ids" => Tensor::from_array(([n, l], b.input_ids.clone()))?,
            "attention_mask" => Tensor::from_array(([n, l], b.attention_mask.clone()))?,
        ])?;
        let (shape, hidden) = enc_out["last_hidden_state"].try_extract_tensor::<f32>()?;
        let hidden_dim = shape[2] as usize;
        let hidden = hidden.to_vec();
        drop(enc_out);

        let head_out = self.head.run(ort::inputs![
            "hidden_states" => Tensor::from_array(([n, l, hidden_dim as i64], hidden.clone()))?,
            "attention_mask" => Tensor::from_array(([n, l], b.attention_mask.clone()))?,
            "marker_pos" => Tensor::from_array(([n, k], b.marker_pos.clone()))?,
            "marker_mask" => Tensor::from_array(([n, k], b.marker_mask.clone()))?,
            "qtype" => Tensor::from_array(([n], b.qtype.clone()))?,
        ])?;
        let (_, logits) = head_out["logits"].try_extract_tensor::<f32>()?;
        let (_, act) = head_out["act_logits"].try_extract_tensor::<f32>()?;
        Ok(Outputs { hidden, hidden_dim, logits: logits.to_vec(), act_logits: act.to_vec() })
    }
}
