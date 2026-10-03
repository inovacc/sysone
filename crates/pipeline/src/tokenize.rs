//! s03: the pinned `tokenizer.json` through the `tokenizers` crate — never a re-implemented BPE
//! (added-vocabulary extraction of space runs and placeholders must match the Python wheel).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use tokenizers::Tokenizer;

pub struct Tok {
    inner: Tokenizer,
    pub cls: i64,
    pub sep: i64,
    pub pad: i64,
    pub mask: i64,
}

impl Tok {
    /// Loads `<snapshot>/tokenizer/tokenizer.json`; special ids come from the token names in
    /// `tokenizer_config.json` (what `tok.cls_token_id` etc. resolve on the Python side).
    pub fn load(snapshot: &Path) -> Result<Self> {
        let dir = snapshot.join("tokenizer");
        let inner = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json"))?)
            .context("tokenizer_config.json")?;
        let id = |key: &str| -> Result<i64> {
            let name = cfg[key].as_str().ok_or_else(|| anyhow!("tokenizer_config.json: {key} missing"))?;
            inner.token_to_id(name).map(i64::from).ok_or_else(|| anyhow!("token {name} not in vocab"))
        };
        Ok(Self { cls: id("cls_token")?, sep: id("sep_token")?, pad: id("pad_token")?, mask: id("mask_token")?, inner })
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i64::from(i)).collect())
    }
}
