# sysone HTTP API
<!-- rev:001 (RFC 3339) 2026-10-03T17:30:00Z -->

sysone speaks the **System One wire protocol**: the one TypeSafe's Jev API uses (`POST /v1/systemone`) and the one
laya-serve, Laya's own Python server, exposes. Its answers, status codes, error bodies and limits are those of
laya-serve, byte for byte (see [PARITY.md](PARITY.md)). A client written for Jev or laya-serve needs only a new
base URL.

Base URL: `http://127.0.0.1:8000` by default (`sysone serve --addr HOST:PORT`).

## Authentication

Off by default. Start the server with `LAYA_API_KEY` set to require `Authorization: Bearer <key>` on
`POST /v1/systemone`. A missing or different header gets `401 {"detail":"invalid or missing bearer token"}`;
the comparison is constant-time over the raw header bytes. `GET /health` is never authenticated.

## `GET /health`

```json
{"status":"ok","loaded":["typed-decisions"],"revisions":{},"device":"cpu"}
```

## `POST /v1/systemone`

### Request

```json
{
  "model": "jev-latest",
  "state": "I was charged twice for my subscription this month.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments, refunds", "support": "product help", "sales": "upgrades"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is this?",
                   "criteria": ["not urgent", "soon", "today", "right now"]},
    "refund":     {"type": "noul",   "instructions": "Does the customer want a refund?"}
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `state` | yes | The text the questions are about: a string, a JSON object (serialized with Python's `json.dumps` rules), or a conversation list (`[{"role": ..., "content": ...}]`, truncated from the left when long) |
| `questions` | yes | An object `{question_id: question}`; ids become the keys of `answers`, in the same order |
| `model` | no | Accepted and ignored: this server serves one checkpoint |

Question types:

| `type` | `criteria` | Answer |
|---|---|---|
| `choice` | an object `{label: description}` (description may be `null`) or a list of labels | the most probable label |
| `score` | a list of level descriptions, index 0 first (at least one level) | the probability-weighted level, 0-based |
| `noul` | omitted, or an object with only `true` / `false` descriptions | P(yes) |

Every question needs `instructions`. A `noul` question may carry `labels: {"false": "...", "true": "..."}`.

### Response `200`

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.9512, "support": 0.0391, "sales": 0.0097},
                   "confidence": 0.8104, "answer_confidence": 0.9512, "action": {"act_probability": 1.0}},
    "urgency":    {"type": "score", "score": 1.4021,
                   "legend": {"0": "not urgent", "1": "soon", "2": "today", "3": "right now"},
                   "probabilities": {"0": 0.12, "1": 0.41, "2": 0.33, "3": 0.14},
                   "confidence": 0.0866, "answer_confidence": 0.41, "action": {"act_probability": 1.0}},
    "refund":     {"type": "noul", "noul": 0.7731, "confidence": 0.7731, "answer_confidence": 0.7731,
                   "action": {"act_probability": 1.0}}
  },
  "usage": {"input_tokens": 182, "output_tokens": 0}
}
```

(Values above are illustrative.)

| Key | Meaning |
|---|---|
| `probabilities` | Calibrated distribution over the options, rounded to 4 decimals |
| `choice` / `score` / `noul` | The answer, as described above |
| `confidence` | `choice`/`score`: 1 − normalized entropy of the distribution. `noul`: max(p, 1 − p) |
| `answer_confidence` | The probability of the top option |
| `action.act_probability` | Laya's act head. The model card reports it carries no usable signal; prefer `confidence` |
| `usage.input_tokens` | Tokens the model read (state + question, per question, summed) |

Response headers: `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms: <ms>`. The body is compact JSON
(`,` and `:` separators), as FastAPI writes it.

### Errors

Every error body is `{"detail": "<message>"}`.

| Status | When |
|---|---|
| 400 | Body is not JSON; not an object with `questions`; `state` missing or `null`; `questions` not an object |
| 401 | Bearer check on and the header is missing or wrong |
| 404 / 405 | Unknown path / wrong method |
| 413 | Body over 2 MiB; more than 64 questions; a `choice` with more than 100 options; a `score` with more than 32 levels; more than 512 options in total; `state` over 50,000 characters |
| 422 | A question the model rejects; the message names it and says what to fix, e.g. `question 'a': unknown type 'ranking'; use one of ['choice', 'noul', 'score']` |
| 500 | `{"detail":"inference failed"}`; the cause is logged on the server's stderr, never sent |

An empty `questions` object returns `200` with empty `answers` and zero usage, without running the model.

## Differences from TypeSafe Jev

| | Jev | sysone |
|---|---|---|
| `model` in the response | `jev-1.13.0` | `laya-rl-agent` |
| Extra answer keys | — | `answer_confidence`, `action`; `confidence` also on `noul` |
| Probability rounding | 2 decimals | 4 decimals |
| `usage.output_tokens` | non-zero | always 0 |
| State budget | ~32k tokens | 1024 tokens (longer states are truncated) |
| Many options | strong | weaker past ~20 options (they share a 256-token budget) |
| `choice` criteria as a list | rejected (422) | accepted |
| Confidence formula | (n·p_max − 1)/(n − 1) | 1 − normalized entropy: thresholds do not transfer |

Every key Jev returns, sysone returns with the same meaning and the same `probabilities` keys; the `score`
scale is the same (probability-weighted 0-based level).

## Concurrency

One request is processed at a time, like laya-serve's single worker; others wait on the socket. `--threads`
sets ONNX Runtime's intra-op threads (use the number of physical cores); it does not change the answers.

## Examples

```sh
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The invoice total does not match the purchase order.",
  "questions": {"mismatch": {"type": "noul", "instructions": "Is there a price discrepancy?"}}}'
```

```powershell
$body = @{ state = 'Order arrived damaged.'; questions = @{ action = @{ type = 'choice';
  instructions = 'What should we do?'; criteria = @('refund', 'replace', 'ask for photos') } } } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/systemone -Method Post -ContentType 'application/json' -Body $body
```
