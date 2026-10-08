# STT API

Whisper speech-to-text over HTTP. Default `http://127.0.0.1:8771`
(`python -m voice_services.stt`). This contract is the wire format the `voice`
project's `stt_server.py` already speaks; it is kept as is so existing callers
keep working.

No authentication. Bind to localhost, or to a Tailscale interface only. Never
expose it on a public interface.

## `GET /`

Health check. Available once the model has loaded (the server does not accept
connections before that).

```json
{"status": "ok", "model": "turbo", "device": "cuda"}
```

## `POST /transcribe`

**Request body:** raw little-endian float32 PCM, mono, 16 kHz, no header
(`Content-Type: application/octet-stream`). The caller resamples and mixes
down; the server does not. Length must be a multiple of 4 bytes.

**Query parameters**

| name | meaning |
|---|---|
| `language` | Pin the transcript language (e.g. `en`, `ja`). Skips detection; fastest. |
| `candidates` | Comma-separated languages to detect among, used only when `language` is absent. Whisper's own detector wanders on short accented clips, so detection is restricted to this set. |

Neither given: detects among the server's configured languages
(`--languages`, default `en,ja`).

**Response 200**

```json
{
  "text": "こんにちは",
  "language": "ja",
  "max_no_speech_prob": 0.01,
  "avg_logprob": -0.21,
  "duration_s": 1.6
}
```

`text` is `""` when the server judges the audio to be silence or a Whisper
hallucination (empty or non-word output, all segments silence-like, or a known
filler phrase such as "Thank you."). Treat `""` as "nothing was said", not as an
error. The filter runs server side; callers should not re-implement it.

**Errors** (JSON `{"detail": "..."}`)

| status | when |
|---|---|
| 400 | body length is not a multiple of 4 bytes |
| 422 | malformed query parameters |

## Concurrency

One model instance. Requests are processed one at a time; concurrent callers
queue. Expect latency to grow with the queue, not errors.
