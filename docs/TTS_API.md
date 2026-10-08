# TTS API

GPT-SoVITS fine-tuned character voices over HTTP. Default
`http://127.0.0.1:8772` (`python -m voice_services.tts`).

**This contract is new.** Nothing in the `voice` project speaks it yet: that
project talks to upstream GPT-SoVITS's own `api_v2.py` directly. This server
sits in front of one or more `api_v2.py` processes, supervises them, and hides
their quirks (voice manifests, reference audio, the degenerate-output retry).

CUDA only. No authentication: bind to localhost or a Tailscale interface only.

## Voices and the pool

A voice is a directory under `VOICE_WEIGHTS_ROOT` containing a `voice.json`
(`gpt_weights`, `sovits_weights`, `ref_audio_path`, `ref_text`, optional
`version`, `ref_lang`). Callers never see paths; they name voices.

The server runs a **pool of upstream `api_v2.py` processes**, each holding one
voice at a time. Pool size defaults to **2** (one English voice and one
Japanese voice warm) and is configurable; the intended next step is 3
(en + ja + a second ja). The caller never addresses an instance:

- A request for a voice that is loaded in some instance goes to that instance
  with no delay.
- A request for a voice that is loaded nowhere is assigned to a free instance,
  else the least recently used one (never a pinned one), which hot-swaps its
  weights first (seconds). This delays only requests for that instance.
- Different instances synthesize in parallel. Requests to the same instance are
  serialized in arrival order.

Startup configuration (flags, with `.env` equivalents; `.env.example` lists
them):

| setting | meaning |
|---|---|
| `--pool-size N` | Number of upstream processes. Default 2. |
| `--pin a,b` | Voices that keep their instance. A pinned voice is loaded at startup (whether or not it is in `--preload`), always served by its own instance, and never chosen to swap: other voices share the remaining instances. Must be fewer than the pool size; an unknown name stops startup. Use it for the voice that is almost always in use (e.g. `--pin rosamund`), so asking for another voice cannot evict it. |
| `--preload a,b,c` | Voices loaded at startup, one per instance, in order. Voices beyond `N` are ignored with a warning; fewer than `N` leaves the rest idle until first use. Preloaded voices are also warmed up with a synthesis. |

## `GET /`

```json
{
  "status": "ok",
  "ready": true,
  "instances": [
    {"voice": "ayaka", "ready": true, "busy": false},
    {"voice": "rosamund", "ready": true, "busy": true, "pinned": true},
    {"voice": null, "ready": true, "busy": false}
  ]
}
```

`ready` is the server as a whole: `true` once the `--preload` voices have all
finished loading (immediately if there are none). The server accepts connections
during startup, so clients should poll this rather than rely on connect errors.
An instance with `voice: null` is idle. `pinned: true` (present only when true) marks
an instance holding a `--pin` voice.

`status` is `"degraded"` when any instance failed to start or swap; that
instance then carries an `error` string, and it is retried on the next request
that needs it. A failed preload still sets `ready` to `true`, so check `status`.

## `GET /voices`

Voices available on disk. Optional `?language=ja` keeps only voices whose
reference clip is in that language. `loaded` lists the voices currently held by
an instance (the ones that will answer without a swap). `pinned` lists the `--pin`
voices, which are never swapped out (empty when none).

```json
{
  "loaded": ["ayaka", "rosamund"],
  "pinned": ["rosamund"],
  "voices": [{"name": "ayaka", "ref_lang": "ja", "version": "v2"}]
}
```

## `POST /voice`

Ensure a voice is loaded in some instance, swapping the least recently used
unpinned one if needed. Use it to warm a voice ahead of the first `/tts`.

Request: `{"name": "hutao"}`. Response 200: `{"voice": "hutao", "swapped": true}`
(`swapped` is `false` if it was already loaded). 404 if the name is not an
available voice. Takes seconds when `swapped` is true.

## `POST /tts`

Request (JSON):

| field | required | meaning |
|---|---|---|
| `text` | yes | Text to speak. |
| `text_lang` | yes | Language of `text` (`ja`, `en`). Not the reference clip's language, which is fixed per voice. |
| `voice` | yes | Voice name. Routed as described under "Voices and the pool". |

`voice` is required: with a pool there is no meaningful "current voice".

Query: `stream=true` selects streaming (below). Default is `false`.

**Non-streaming response 200:** `Content-Type: audio/wav`, mono, 16-bit PCM,
plus `X-Sample-Rate: <hz>`. The body is the complete utterance.

**Streaming response 200:** `Content-Type: audio/L16`, raw little-endian
16-bit mono PCM with no header, chunked transfer, plus `X-Sample-Rate: <hz>`
sent before the first chunk. Chunks arrive as generated and always contain whole
samples. The sample rate is read from the WAV header the upstream sends first; the
header itself is not forwarded. Measured on an RTX 5090: first byte in about
0.3 s, versus about 1.4 s for the complete non-streaming response to the same text.

**Server-side behavior callers can rely on**

- Output shorter than a plausible floor for the text (a rare upstream
  sampling failure) is regenerated, up to 3 attempts, before being returned
  anyway with a warning in the server log. This applies to non-streaming
  requests; a streamed response cannot be retried once bytes have been sent.
- A swap in progress on one instance never blocks requests for voices held by
  another.

**Errors** (JSON `{"detail": "..."}`; for streaming, only before the first
byte is sent):

| status | when |
|---|---|
| 400 | empty `text`, or unsupported `text_lang` |
| 404 | unknown `voice` |
| 422 | malformed body |
| 503 | no instance could serve the request (not ready, crashed, or failed to start); `detail` carries the reason |

If an upstream process dies mid-stream, the connection is closed without a
clean end; clients should treat a truncated stream as a failure. A crashed
instance is restarted on the next request that needs it.
