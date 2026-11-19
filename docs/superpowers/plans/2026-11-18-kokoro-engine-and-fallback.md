# Kokoro engine and TTS fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve French and Italian (and English and Japanese as a safety net) from Kokoro-82M beside the GPT-SoVITS pool, behind the same `/tts` and `/voices` contract, and fall back to Kokoro when a GPT-SoVITS voice cannot answer.

**Architecture:** A second engine, `KokoroEngine`, runs in-process (one small model, no subprocess, no pool slot). Its voices are a fixed catalogue in code, listed by `GET /voices` next to the voice directories, each with an `engine` field. `POST /tts` routes by voice name: a catalogue name goes to Kokoro, anything else to the pool as today. When a pool voice fails with a 503 before any audio is sent, the server retries once with Kokoro's voice for `text_lang` and marks the response with headers.

**Tech Stack:** Python 3.12, FastAPI, pytest, `kokoro` (Kokoro-82M) with `misaki`, existing torch (CUDA) install.

**Spec:** `/Users/curisu/dev/voice-js/docs/french-italian-prd.md` (decisions D1, D4–D8; AC2, AC8–AC10). This plan covers the `voice-services` half only.

## Global Constraints

- Kokoro voices: French `ff_siwis`, Italian `if_sara` (decided). English and Japanese fallback voices are chosen in Task 1's listening check; until then `af_heart` (en) and `jf_alpha` (ja) are placeholders.
- Kokoro output is 24 kHz mono; the `X-Sample-Rate` header carries it, as for existing voices.
- Kokoro honours `speed_factor` only. `temperature`, `top_p`, `top_k` and `fragment_interval` are accepted and ignored for Kokoro voices (never a 400).
- A voice name that exists nowhere is still 404. Fallback never masks 404, 400 or 422.
- GPT-SoVITS pool, pinning and swap behaviour (`pool.py`) do not change.
- Kokoro is optional: without the `kokoro` package the server starts, lists no Kokoro voices, and has no fallback.
- `kokoro` needs Python 3.12 or older: keep `requires-python` compatible, and say so in the README.

## Review Focus

- `kokoro` not installed, or its weights missing: server starts, `/voices` and `/tts` for GPT-SoVITS voices are unchanged, a Kokoro voice name is a 404 with a message naming the cause.
- A fallback for a `text_lang` with no Kokoro voice (e.g. `de`): the original 503 is returned, not a 500.
- Fallback must not trigger on a 404 (unknown voice) or on a request that already named a Kokoro voice (no retry loop).
- Streaming: fallback only before the first byte; a mid-stream failure still just ends the connection, as documented.
- Concurrency: two simultaneous requests to Kokoro (one model, shared state) must not interleave or crash; and a Kokoro request must not hold a pool slot or evict a pinned voice.

---

### Task 1: Spike: install Kokoro and pick the voices

**Files:**
- Create: `scripts/kokoro_samples.py` (writes WAVs for a phrase file; not imported by the server)
- Create: `tests/fixtures/phrases.json` (about 20 phrases each for `fr`, `it`, plus 5 each for `en`, `ja`)
- Modify: `docs/TTS_API.md` (record the outcome in a "Default engine" section)

**Interfaces:**
- Produces: the chosen voice name per language, recorded in `docs/TTS_API.md`, which Task 2 copies into `KOKORO_VOICES`.

- [ ] **Step 1: Install on a dev machine** with `pip install kokoro misaki[ja]` plus `espeak-ng` (the French and Italian G2P needs it); note the exact versions in the doc. Confirm it imports under the repo's Python and runs on `clotho` (CPU or CUDA beside GPT-SoVITS).
- [ ] **Step 2: Write `scripts/kokoro_samples.py`**: for each language in the fixtures, synthesize every phrase with the language's candidate voices to `samples/<lang>/<voice>/<n>.wav`. Phrases cover digits, dates, accents, liaisons, double consonants, loanwords.
- [ ] **Step 3: Listening check** by a fluent listener for `ff_siwis` and `if_sara`; if either fails, try the other Kokoro voices for that language. Pick en and ja fallback voices the same way.
- [ ] **Step 4: Record the outcome** (versions, voices, phrase file, any phrase that failed) in `docs/TTS_API.md`.
- [ ] **Step 5: Commit** `git add scripts docs tests/fixtures && git commit -m "docs: Kokoro engine spike and voice choice"`

### Task 2: `KokoroEngine` and its voice catalogue

**Files:**
- Create: `voice_services/tts/kokoro.py`
- Modify: `pyproject.toml` (optional extra `kokoro = ["kokoro", "misaki[ja]"]`)
- Test: `tests/test_kokoro.py`, `tests/fakes.py` (a `FakePipeline`)

**Interfaces:**
- Produces:
  - `KOKORO_VOICES: dict[str, str]`, voice name to `ref_lang` (`{"ff_siwis": "fr", "if_sara": "it", <en>: "en", <ja>: "ja"}`, from Task 1).
  - `class KokoroEngine(pipeline_factory=None)`:
    - `available: bool`, false when the package or weights cannot be loaded (the reason in `.error`).
    - `synthesize(text: str, voice: str, sampling: dict) -> tuple[np.ndarray, int]`, mono float32 and 24000.
    - `stream_tts(text: str, voice: str, sampling: dict) -> tuple[int, Iterator[bytes]]`, whole-sample little-endian 16-bit PCM, like `UpstreamInstance.stream_tts`.
    - `voice_for(language: str) -> str | None`, the catalogue's voice for a language (the fallback voice).
  - Only `speed_factor` is read from `sampling`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests** in `tests/test_kokoro.py`: `synthesize` returns `(float32 array, 24000)` and passes the voice, a lang code (`f` for `fr`, `i` for `it`, `a` for `en`, `j` for `ja`) and `speed` to the pipeline; `stream_tts` yields only even-length byte chunks; `sampling` with `temperature`/`top_k` does not raise; `voice_for("fr") == "ff_siwis"` and `voice_for("de") is None`; a factory that raises makes `available` false with `.error` set; two threads calling `synthesize` together both succeed (the engine holds a lock).
- [ ] **Step 2: Run** `pytest tests/test_kokoro.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement `voice_services/tts/kokoro.py`.** One `KPipeline` per lang code, created lazily under a lock and reused; the real factory is the default, the fake is injected in tests. Concatenate the pipeline's yielded chunks for `synthesize`; convert to int16 for `stream_tts`. Import `kokoro` inside the factory so the module imports without it.
- [ ] **Step 4: Run** `pytest tests/test_kokoro.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `git add voice_services/tts/kokoro.py tests pyproject.toml && git commit -m "feat(tts): Kokoro engine and voice catalogue"`

### Task 3: Route `/tts`, list voices, new languages

**Files:**
- Modify: `voice_services/tts/server.py` (`TEXT_LANGUAGES`, `voices()`, `set_voice()`, `tts()`), `voice_services/tts/pauses.py` (`_COMMA`), `voice_services/tts/__main__.py` or `main()` (create the engine into `app.state.kokoro`)
- Create: `pronunciations/fr.json`, `pronunciations/it.json` (both `{}`)
- Test: `tests/test_tts_server.py`, `tests/test_pauses.py`

**Interfaces:**
- Consumes: `KokoroEngine`, `KOKORO_VOICES` (Task 2).
- Produces:
  - `GET /voices` entries gain `engine`: `"gpt-sovits"` for existing voices, `"kokoro"` for catalogue voices (with `ref_lang` from the catalogue, `version` null). `?language=` filters both. Kokoro voices appear only when `available`.
  - `POST /tts` with a catalogue voice is served by Kokoro, never touching the pool. `POST /voice` for a catalogue voice answers `{"voice": name, "swapped": false}`.
  - `TEXT_LANGUAGES = ("ja", "en", "fr", "it")`. `pauses._COMMA` gains `fr: ", "`, `it: ", "`, and `apply` treats the French space before `: ; ? !` and `«  »` without leaving a stray gap or breaking a time like `10:30`.

- [ ] **Step 1: Write the failing tests** (use a fake `KokoroEngine` on `app.state`):
  - `test_voices_lists_kokoro_voices_with_engine` (also: `?language=fr` returns only `ff_siwis`);
  - `test_tts_routes_a_kokoro_voice_to_kokoro_not_the_pool` (the fake pool sees no `acquire`);
  - `test_tts_accepts_fr_and_it_and_still_400s_for_de`;
  - `test_tts_kokoro_ignores_sampling_except_speed`;
  - `test_set_voice_for_a_kokoro_voice_is_not_swapped`;
  - `test_a_kokoro_voice_is_404_when_kokoro_is_unavailable` (message names the cause);
  - in `test_pauses.py`: `apply("Il est 10:30 : voilà !", "fr")` keeps `10:30` and pauses at the second colon.
- [ ] **Step 2: Run** `pytest tests/test_tts_server.py tests/test_pauses.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `tts()`, resolve the engine from the voice name first (catalogue and `available` means Kokoro, else the existing path), keeping the respell and pause steps for both. Reuse the existing response building (WAV/stream) for both engines; do not duplicate it. `GET /voices` merges the two lists, directory voices first.
- [ ] **Step 4: Run** the whole suite: `pytest`. Expected: all pass.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(tts): serve fr/it and Kokoro voices behind the same contract"`

### Task 4: Fallback to Kokoro

**Files:**
- Modify: `voice_services/tts/server.py` (`tts()`)
- Test: `tests/test_tts_server.py`

**Interfaces:**
- Consumes: `KokoroEngine.voice_for`, the routing from Task 3.
- Produces: when a pool voice's `_acquire` or non-streaming synthesis fails with a 503 (or a streaming start fails before the first byte), the server answers with Kokoro's `voice_for(req.text_lang)` and adds headers `X-Fallback-Voice: <kokoro voice>` and `X-Fallback-From: <requested voice>`. If there is no Kokoro voice for the language, or Kokoro is unavailable, the original 503 stands. It never retries a Kokoro voice, a 404, a 400 or a 422. A line goes to the server log for every fallback.

- [ ] **Step 1: Write the failing tests:**
  - `test_pool_503_falls_back_to_kokoro_with_headers` (non-streaming);
  - `test_streaming_start_failure_falls_back`;
  - `test_no_fallback_for_a_language_kokoro_lacks` (503 kept);
  - `test_no_fallback_on_unknown_voice` (404 kept);
  - `test_a_kokoro_failure_is_not_retried` (503 kept, one attempt);
  - `test_pinned_voice_stays_loaded_during_a_fallback` (the pool state is unchanged);
  - `test_normal_responses_have_no_fallback_headers`.
- [ ] **Step 2: Run** `pytest tests/test_tts_server.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** by wrapping the existing pool path in `tts()` in a helper that returns the engine-specific result, and calling the Kokoro path once on `HTTPException(503)`. The fallback decision lives in one place.
- [ ] **Step 4: Run** `pytest`. Expected: all pass.
- [ ] **Step 5: Commit** `git add -A && git commit -m "feat(tts): fall back to Kokoro when a GPT-SoVITS voice cannot answer"`

### Task 5: STT defaults, docs, setup checks

**Files:**
- Modify: `voice_services/config.py` (`WhisperConfig.languages` stays `en,ja`; document `VOICE_STT_LANGUAGES` for `fr,it`), `docs/TTS_API.md`, `docs/STT_API.md`, `README.md`, `.env.example`, `scripts/check_setup.py`, `scripts/fetch_weights.py`
- Test: `tests/test_stt_server.py` (a `languages=("en","ja","fr","it")` config is accepted and `?language=fr` passes through), `tests/test_env.py` if env parsing changes

**Interfaces:**
- Consumes: the whole contract from Tasks 2 to 4.
- Produces: the documented contract that voice-js reads: `engine` on `/voices`, `X-Fallback-Voice` and `X-Fallback-From` on `/tts`, `text_lang` in `ja,en,fr,it`, `GET/PATCH /pronunciations/{lang}` for the four, Kokoro's sampling limits.

- [ ] **Step 1: Write the failing test** for the STT config in `tests/test_stt_server.py`.
- [ ] **Step 2: Run** it. Expected: FAIL only if parsing needs a change; if it already passes, record that and move on.
- [ ] **Step 3: Update docs and scripts:** `TTS_API.md` (voices `engine`, fallback headers and rules, new languages, sampling table per engine), `STT_API.md`, `README` and `.env.example` (`kokoro` extra, `espeak-ng`, Python 3.12 note, Kokoro weights download), `check_setup.py` (reports Kokoro availability, a warning not an error), `fetch_weights.py` (Kokoro weights).
- [ ] **Step 4: Run** `pytest`. Expected: all pass.
- [ ] **Step 5: Commit** `git add -A && git commit -m "docs: Kokoro engine, fallback headers, fr/it languages"`

### Task 6: Smoke test on clotho (manual)

**Files:** none.

- [ ] **Step 1:** Deploy to `clotho` and restart the TTS service with `--pin rosamund` as today.
- [ ] **Step 2:** `curl` `/voices` and check the Kokoro voices and `engine` appear.
- [ ] **Step 3:** `curl` `/tts` once each with `ff_siwis` (`fr`) and `if_sara` (`it`); play the WAVs.
- [ ] **Step 4:** Stop the unpinned GPT-SoVITS instance (or ask for it while mid-swap) and request a Japanese voice; confirm the Kokoro response and both headers, and that `rosamund` stayed loaded in `GET /`.
- [ ] **Step 5:** Record results in the PR description.
