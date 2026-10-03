# sysone
<!-- rev:001 (RFC 3339) 2026-10-03T17:30:00Z -->

**Typed decisions from a local binary. No Python, no GPU, no cloud.**

sysone answers typed questions about a piece of text — *which team?* (choice), *how urgent?* (score),
*does the customer want a refund?* (yes/no) — with calibrated probabilities, in one forward pass, on your own
machine. It is a Rust implementation of the inference path of **[Laya](https://github.com/NandhaKishorM/laya)**,
running Laya's published `laya-typed-decisions` model through ONNX Runtime, and it serves the
**System One HTTP protocol** (`POST /v1/systemone`), so a client written for TypeSafe's Jev API or for Laya's own
server works by changing the base URL.

*Powered by Laya (ConvAI Innovations, Apache-2.0). Not affiliated with or endorsed by TypeSafe AI or ConvAI Innovations.*

## Install

The installer brings everything sysone needs — the binary, ONNX Runtime and the model (~1.7 GB) — verifies every
file's sha256, and puts `sysone` on your PATH. No Python, no administrator rights; re-running resumes.

**Windows x64** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/inovacc/sysone/main/install.ps1 | iex
```

**Linux x86_64**:

```sh
curl -fsSL https://raw.githubusercontent.com/inovacc/sysone/main/install.sh | sh
```

Installs to `%LOCALAPPDATA%\Programs\sysone` (Windows) or `~/.local/share/sysone` with a link in `~/.local/bin`
(Linux). Pin a version with `-Version v0.1.0` / `SYSONE_VERSION=v0.1.0`, choose the directory with `-Dir` /
`SYSONE_DIR`.

**Manual install**: download `sysone-<version>-<platform>` from [Releases](https://github.com/inovacc/sysone/releases)
and check it against `SHA256SUMS`, unpack it, then download the `model/` and `bundle/` folders from
[huggingface.co/Dyam/sysone-laya-typed-decisions-onnx](https://huggingface.co/Dyam/sysone-laya-typed-decisions-onnx)
next to the executable. sysone looks for `model/`, `bundle/` and the ONNX Runtime library beside itself
(`--model`, `--bundle`, `--ort-lib` or `SYSONE_MODEL`, `SYSONE_BUNDLE`, `ORT_DYLIB_PATH` override that).

## Run

```sh
sysone serve                                  # http://127.0.0.1:8000
sysone serve --addr 0.0.0.0:8000 --threads 4  # listen on the network, 4 ONNX Runtime threads
LAYA_API_KEY=secret sysone serve              # require "Authorization: Bearer secret"
```

```sh
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice this month and want my money back.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team handles this?",
                   "criteria": {"billing": "payments, refunds", "support": "product help", "sales": "upgrades"}},
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"}}}'
```

Use `--threads` = your number of physical cores. Loading takes ~15–40 s; a request takes from under a second to a
couple of minutes on CPU depending on state length and question count (one 1024-token state × five questions is
the slow end). One request is served at a time.

`sysone batch --scenarios <dir> --out <dir>` runs `<dir>/<id>/request.json` files and writes every intermediate
stage per scenario, which is what parity checking reads. `sysone version` prints the build.

## API

`POST /v1/systemone`, `GET /health`; optional bearer auth; JSON errors `{"detail": ...}`. Full reference, with
every field, status code and limit, and the differences from Jev: **[docs/API.md](docs/API.md)**.

## What it is (and is not)

- **A decision model, not a chatbot.** The model never generates text, so it cannot produce an answer outside
  your options. It is a ModernBERT-large encoder (28 layers, 421M parameters) with a small transformer head that
  scores every option of every question at once, then calibrates the scores with per-type temperatures.
- **The same answers as Laya in Python.** Validation, tokenization, sequence building, calibration, ranking,
  rounding and JSON output are ported line for line from laya 0.3.20 and are bit-identical to it. The neural
  network runs in ONNX Runtime instead of PyTorch, so internal floats differ by ~1e-6 (the same order as PyTorch
  differs from itself between thread counts); on every test case the HTTP response is byte-identical to the Python
  server's. How this was measured: **[docs/PARITY.md](docs/PARITY.md)**.
- **Laya's limits are its limits.** It reads up to 1024 tokens of state (longer text is truncated), and accuracy
  drops with many options (past ~20, the labels share a 256-token budget). The model is fine-tuned on four
  workflows — invoice processing, security incidents, customer service and agent-trace observability — and is
  near chance zero-shot on unrelated tasks. See the
  [model card](https://huggingface.co/convaiinnovations/laya-typed-decisions).

## Where it comes from

| Piece | Source | License |
|---|---|---|
| The model | [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) @ `1a793eb5` | Apache-2.0 |
| The reference implementation | [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) 0.3.20 @ `4066d5d5` | Apache-2.0 |
| The ONNX export of the model | exported from the checkpoint above as `encoder.onnx` + `head.onnx`, hosted at [`Dyam/sysone-laya-typed-decisions-onnx`](https://huggingface.co/Dyam/sysone-laya-typed-decisions-onnx) | Apache-2.0 |
| The inference runtime | [ONNX Runtime](https://github.com/microsoft/onnxruntime) 1.28.0, official release | MIT |
| This code | this repository | BSD-3-Clause |

The protocol is TypeSafe's System One API as documented publicly and as implemented by laya-serve; sysone
implements a compatible server and contains no TypeSafe code.

## Build from source

```sh
cargo build --release        # Rust 1.98 (pinned in rust-toolchain.toml); a C compiler for the tokenizer's regex engine
```

The binary loads ONNX Runtime 1.28.0 at run time (`onnxruntime.dll` / `libonnxruntime.so`), so put the official
release library next to it, or point `--ort-lib` at it.

## License

BSD-3-Clause for the code in this repository ([LICENSE](LICENSE)). The model and ONNX Runtime keep their own
licenses ([NOTICE](NOTICE)).
