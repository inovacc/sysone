# Parity with Laya (Python)
<!-- rev:001 (RFC 3339) 2026-10-03T17:30:00Z -->

sysone is accepted only if it gives the answers the Python reference gives. This file says how that was checked
and what was found. The harness that produced it (a private porting lab) runs the Python reference and sysone on the
same requests and compares them stage by stage.

## The reference

laya 0.3.20 (`NandhaKishorM/laya` @ `4066d5d5`) with `convaiinnovations/laya-typed-decisions` @ `1a793eb5`, on CPU
in float32 (torch 2.14.0+cpu, transformers 5.17.0, tokenizers 0.23.2), calling `Agent.system_one` unchanged.

## What is compared

Each request goes through 13 stages; both sides dump every one.

| Stages | What they are | Result |
|---|---|---|
| question normalization, texts, tokens, masks, shapes | everything Laya's code does before the network | **bit-identical** |
| encoder hidden states, logits, probabilities | the network (PyTorch vs ONNX Runtime) | **within tolerance**: worst encoder 0.0065, logits 2.3e-05, probabilities 3.8e-06 |
| calibration, ranking, decision, output | everything after the network | **bit-identical** |
| HTTP response | the bytes a client receives | **byte-identical** to the Python server's |

Tolerances are 4 × the reference's own variation (PyTorch against itself when only the thread count, the attention
backend, batching or the fused-kernel path change), measured on a calibration set and then checked on a held-out
set nobody had looked at. The same ONNX graphs run by Python's ONNX Runtime give exactly sysone's bits: the
remaining difference is PyTorch vs ONNX Runtime arithmetic, not the port.

## Coverage

36 requests in three sets (12 calibration, 12 + 12 held-out): choice / score / yes-no questions, 1 to 5 questions,
states from one line to 1024 tokens, string, object and conversation states, English and French, list and object
criteria, edge cases of label equality and float formatting. HTTP behaviour was also compared against laya-serve on
32 edge cases (malformed bodies, every size limit, unknown types, bearer auth including non-ASCII bytes, wrong method
and path): same status, same body.

Result for v0.1.0 on Windows x64: 36/36 within tolerance stage by stage, 36/36 HTTP responses byte-identical,
32/32 edge cases identical, with the packaged binary run from a process with no Python on its PATH.

## What is not covered

- Linux builds are compiled and smoke-tested in CI (three requests, response bytes compared with the Windows
  answers) but have not been through the full stage-by-stage comparison.
- A near-tie between two options (a gap under ~4e-06) could flip the top choice between PyTorch and sysone. It did
  not happen in any test; it would be a choice between options the model itself cannot tell apart.
- Accuracy is the model's, not sysone's; see the model card.
