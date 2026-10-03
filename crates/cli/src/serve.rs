//! `sysone serve`: the HTTP surface of laya-serve (`laya/serve.py`) over the Rust pipeline — TypeSafe Jev's
//! `POST /v1/systemone` wire protocol plus `GET /health` and an optional bearer check (`LAYA_API_KEY`).
//!
//! Status codes, error bodies (`{"detail": ...}`) and request limits follow `laya/serve.py` so a client written
//! against Jev or laya-serve sees the same answers. Inference is sequential, as laya-serve's single worker is.
//!
//! The HTTP/1.1 layer is the standard library's `TcpListener`: one connection at a time, `Connection: close`,
//! `Content-Length` or chunked request bodies, and header values kept as bytes — uvicorn accepts any byte in a
//! header value and answers 401 to a bad bearer; a server that drops the connection on a non-ASCII byte would
//! differ from the reference (observed with tiny_http, `.scripts/08-C_http_parity.out.txt`).
//!
//! Known deviations: no `routing` key (that is laya's Router, not the typed-decisions Agent); the `model` field is
//! accepted and ignored (this process serves one checkpoint); the state-size limit counts `json.dumps` characters
//! where Python counts `str(state)` for non-string states.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use pipeline::infer::Engine;
use pipeline::pyjson::{dumps, dumps_compact};
use pipeline::Model;
use serde_json::{json, Value};

// laya/serve.py:59-65 — guardrails for unauthenticated remote input.
const MAX_QUESTIONS: usize = 64;
const MAX_STATE_CHARS: usize = 50_000;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
const MAX_CHOICE_OPTIONS: usize = 100;
const MAX_SCORE_LEVELS: usize = 32;
const MAX_TOTAL_OPTIONS: usize = 512;
/// Request head (request line + headers) cap; h11's default is 16 KiB per line, 64 KiB here is generous.
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// How much of an over-limit body is read and discarded so the 413 reaches the client.
const MAX_DRAIN_BYTES: usize = 64 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(60);

pub struct Config {
    pub addr: String,
    pub api_key: Option<String>,
    pub model_name: String,
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, Vec<u8>)>, // name lower-cased ASCII, value raw bytes trimmed
    /// Set once the body was read (or drained): a reply written while request bytes are still unread makes
    /// the close a TCP reset, and the client may never see the status (observed on Windows, `.scripts/11-C_*`).
    body_read: std::cell::Cell<bool>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&[u8]> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
    }

    fn content_length(&self) -> Option<usize> {
        self.header("content-length").and_then(|v| std::str::from_utf8(v).ok()).and_then(|s| s.trim().parse().ok())
    }
}

struct Reply {
    status: u16,
    body: String,
    extra: Vec<(String, String)>,
}

impl Reply {
    fn json(status: u16, v: &Value) -> Self {
        Self { status, body: dumps_compact(v), extra: Vec::new() }
    }

    /// FastAPI's `HTTPException(status_code, detail)` body.
    fn detail(status: u16, detail: impl Into<String>) -> Self {
        Self::json(status, &json!({"detail": detail.into()}))
    }
}

/// What reading the body produced: the bytes, or the reply the reference gives instead (413).
enum Body {
    Bytes(Vec<u8>),
    Reject(Reply),
}

pub fn run(model: &Model, engine: &mut Engine, cfg: &Config) -> Result<()> {
    let listener = TcpListener::bind(&cfg.addr).map_err(|e| anyhow!("bind {}: {e}", cfg.addr))?;
    eprintln!("sysone serve: listening on http://{} (POST /v1/systemone, GET /health; bearer {})", cfg.addr,
        if cfg.api_key.is_some() { "required" } else { "off" });
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("sysone serve: accept: {e}");
                continue;
            }
        };
        if let Err(e) = handle(model, engine, cfg, stream) {
            eprintln!("sysone serve: connection: {e:#}");
        }
    }
    Ok(())
}

fn handle(model: &Model, engine: &mut Engine, cfg: &Config, stream: TcpStream) -> Result<()> {
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let reply = match read_head(&mut reader) {
        Ok(Some(rq)) => {
            let reply = route(model, engine, cfg, &rq, &mut reader);
            if !rq.body_read.get() {
                drain(&rq, &mut reader);
            }
            reply
        }
        Ok(None) => return Ok(()), // client closed before sending anything
        Err(e) => {
            eprintln!("sysone serve: bad request head: {e}");
            Reply { status: 400, body: "Invalid HTTP request received.".into(), extra: vec![("Content-Type".into(), "text/plain; charset=utf-8".into())] }
        }
    };
    write_reply(stream, reply)
}

/// Parses the request line and headers. Header values stay bytes (latin-1 on the wire, as Starlette reads them).
fn read_head(reader: &mut BufReader<TcpStream>) -> Result<Option<Request>> {
    let mut line = Vec::new();
    let mut total = 0usize;
    let mut read_line = |reader: &mut BufReader<TcpStream>, line: &mut Vec<u8>| -> Result<usize> {
        line.clear();
        let n = reader.read_until(b'\n', line).context("read")?;
        total += n;
        if total > MAX_HEAD_BYTES {
            anyhow::bail!("request head over {MAX_HEAD_BYTES} bytes");
        }
        while line.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
            line.pop();
        }
        Ok(n)
    };
    // Tolerate leading empty lines (RFC 9112 §2.2), stop on a closed socket.
    loop {
        if read_line(reader, &mut line)? == 0 {
            return Ok(None);
        }
        if !line.is_empty() {
            break;
        }
    }
    let request_line = String::from_utf8(line.clone()).context("request line is not UTF-8")?;
    let mut parts = request_line.split(' ');
    let (method, target, version) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if method.is_empty() || target.is_empty() || !version.starts_with("HTTP/1.") {
        anyhow::bail!("malformed request line {request_line:?}");
    }
    let path = target.split('?').next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    loop {
        if read_line(reader, &mut line)? == 0 {
            anyhow::bail!("connection closed inside the headers");
        }
        if line.is_empty() {
            break;
        }
        let Some(colon) = line.iter().position(|b| *b == b':') else { anyhow::bail!("header line without ':'") };
        let name = std::str::from_utf8(&line[..colon]).context("header name")?.trim().to_ascii_lowercase();
        let mut value = line[colon + 1..].to_vec();
        while value.first().is_some_and(|b| *b == b' ' || *b == b'\t') {
            value.remove(0);
        }
        while value.last().is_some_and(|b| *b == b' ' || *b == b'\t') {
            value.pop();
        }
        headers.push((name, value));
    }
    Ok(Some(Request { method: method.to_string(), path, headers, body_read: std::cell::Cell::new(false) }))
}

fn is_chunked(rq: &Request) -> bool {
    rq.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().windows(7).any(|w| w == b"chunked"))
}

/// Reads and discards an unread body (up to `MAX_DRAIN_BYTES`; past that the connection is simply closed).
fn drain(rq: &Request, reader: &mut BufReader<TcpStream>) {
    rq.body_read.set(true);
    if let Some(n) = rq.content_length() {
        let _ = std::io::copy(&mut reader.take(n.min(MAX_DRAIN_BYTES) as u64), &mut std::io::sink());
    } else if is_chunked(rq) {
        let _ = std::io::copy(&mut reader.take(MAX_DRAIN_BYTES as u64), &mut std::io::sink());
    }
}

/// Reads the body with laya-serve's cap (`_read_body_capped`): the declared length is the client's claim; the
/// stream itself is bounded, for `Content-Length` and chunked framing alike.
fn read_body(rq: &Request, reader: &mut BufReader<TcpStream>) -> Result<Body> {
    rq.body_read.set(true);
    if let Some(len) = rq.header("content-length") {
        let declared = std::str::from_utf8(len).ok().and_then(|s| s.trim().parse::<usize>().ok());
        match declared {
            Some(n) if n > MAX_BODY_BYTES => {
                drain(rq, reader);
                return Ok(Body::Reject(Reply::detail(413, "request body too large")));
            }
            Some(n) => {
                let mut buf = vec![0u8; n];
                reader.read_exact(&mut buf).context("read body")?;
                return Ok(Body::Bytes(buf));
            }
            None => {} // an unparsable length is ignored, as serve.py's `except ValueError: pass`
        }
    }
    if !is_chunked(rq) {
        return Ok(Body::Bytes(Vec::new()));
    }
    let mut body = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        reader.read_until(b'\n', &mut line).context("chunk size")?;
        let size_text = std::str::from_utf8(&line).unwrap_or("").trim();
        let size = usize::from_str_radix(size_text.split(';').next().unwrap_or(""), 16).context("chunk size")?;
        if size == 0 {
            // trailers until the empty line
            loop {
                line.clear();
                if reader.read_until(b'\n', &mut line)? == 0 || line == b"\r\n" || line == b"\n" {
                    break;
                }
            }
            return Ok(Body::Bytes(body));
        }
        if body.len() + size > MAX_BODY_BYTES {
            let _ = std::io::copy(&mut reader.take(MAX_DRAIN_BYTES as u64), &mut std::io::sink());
            return Ok(Body::Reject(Reply::detail(413, "request body too large")));
        }
        let mut chunk = vec![0u8; size + 2];
        reader.read_exact(&mut chunk).context("chunk")?;
        chunk.truncate(size);
        body.extend_from_slice(&chunk);
    }
}

fn write_reply(mut stream: TcpStream, reply: Reply) -> Result<()> {
    let reason = match reply.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Content Too Large",
        422 => "Unprocessable Content",
        500 => "Internal Server Error",
        _ => "",
    };
    let mut head = format!("HTTP/1.1 {} {}\r\n", reply.status, reason);
    if !reply.extra.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        head.push_str("Content-Type: application/json\r\n");
    }
    for (k, v) in &reply.extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", reply.body.len()));
    stream.write_all(head.as_bytes())?;
    stream.write_all(reply.body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn route(model: &Model, engine: &mut Engine, cfg: &Config, rq: &Request, reader: &mut BufReader<TcpStream>) -> Reply {
    match (rq.method.as_str(), rq.path.as_str()) {
        ("GET", "/health") => Reply::json(200, &json!({
            "status": "ok", "loaded": [cfg.model_name], "revisions": {}, "device": "cpu"})),
        ("POST", "/v1/systemone") => match systemone(model, engine, cfg, rq, reader) {
            Ok(r) => r,
            Err(e) => {
                // laya/serve.py:382-388 — the client learns nothing, the operator gets the cause.
                eprintln!("sysone serve: inference failed: {e:#}");
                Reply::detail(500, "inference failed")
            }
        },
        (_, "/health") | (_, "/v1/systemone") => Reply::detail(405, "Method Not Allowed"),
        _ => Reply::detail(404, "Not Found"),
    }
}

fn systemone(model: &Model, engine: &mut Engine, cfg: &Config, rq: &Request, reader: &mut BufReader<TcpStream>) -> Result<Reply> {
    if let Some(key) = &cfg.api_key {
        // laya/serve.py:298-305 — the whole header value against "Bearer <key>", constant time, bytes.
        let expected = format!("Bearer {key}");
        if !constant_time_eq(rq.header("authorization").unwrap_or(b""), expected.as_bytes()) {
            return Ok(Reply::detail(401, "invalid or missing bearer token"));
        }
    }
    let raw = match read_body(rq, reader)? {
        Body::Bytes(b) => b,
        Body::Reject(r) => return Ok(r),
    };
    let body: Value = match serde_json::from_slice(&raw) {
        Ok(v) => v,
        Err(_) => return Ok(Reply::detail(400, "request body must be valid JSON")),
    };
    let Some(obj) = body.as_object().filter(|o| o.contains_key("questions")) else {
        return Ok(Reply::detail(400, "request body must be an object with a 'questions' field"));
    };
    let state = obj.get("state").cloned().unwrap_or(Value::Null);
    let questions = &obj["questions"];
    if let Err(r) = check_request_limits(&state, questions) {
        return Ok(r);
    }

    let t0 = Instant::now();
    let req = json!({"state": state, "questions": questions});
    let result = pipeline::predict(model, engine, &req)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(out) => {
            let mut r = Reply::json(200, &out);
            r.extra.push(("Server-Timing".into(), format!("inference;dur={ms:.2}")));
            r.extra.push(("X-Inference-Time-Ms".into(), format!("{ms:.2}")));
            Ok(r)
        }
        // laya/serve.py:379-381 — a ValueError names the question and what to fix; anything else is a 500.
        Err(e) if e.kind == "ValueError" => Ok(Reply::detail(422, e.message)),
        Err(e) => Err(anyhow!("{e}")),
    }
}

/// `_check_request_limits` (laya/serve.py:130-182): 400 for an absent state or non-object questions, 413 for sizes.
fn check_request_limits(state: &Value, questions: &Value) -> Result<(), Reply> {
    if state.is_null() {
        return Err(Reply::detail(400, "'state' is required"));
    }
    let Some(qs) = questions.as_object() else {
        return Err(Reply::detail(400, "'questions' must be an object"));
    };
    if qs.len() > MAX_QUESTIONS {
        return Err(Reply::detail(413, format!("too many questions ({} > {MAX_QUESTIONS})", qs.len())));
    }
    let mut total = 0usize;
    for (qid, q) in qs {
        let Some(q) = q.as_object() else { continue };
        let crit = q.get("criteria");
        match q.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let count = match crit {
                    Some(Value::Object(m)) => m.len(),
                    Some(Value::Array(a)) => a.len(),
                    _ => continue,
                };
                total += count;
                if count > MAX_CHOICE_OPTIONS {
                    return Err(Reply::detail(413, format!("too many choice options for {} ({count} > {MAX_CHOICE_OPTIONS})",
                        pipeline::pyjson::py_str_repr(qid))));
                }
            }
            Some("score") => {
                let Some(Value::Array(a)) = crit else { continue };
                total += a.len();
                if a.len() > MAX_SCORE_LEVELS {
                    return Err(Reply::detail(413, format!("too many score levels for {} ({} > {MAX_SCORE_LEVELS})",
                        pipeline::pyjson::py_str_repr(qid), a.len())));
                }
            }
            _ => {}
        }
    }
    if total > MAX_TOTAL_OPTIONS {
        return Err(Reply::detail(413, format!("too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})")));
    }
    let state_len = match state {
        Value::String(s) => s.chars().count(),
        other => dumps(other).chars().count(),
    };
    if state_len > MAX_STATE_CHARS {
        return Err(Reply::detail(413, format!("state too large ({state_len} > {MAX_STATE_CHARS} chars)")));
    }
    Ok(())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}
