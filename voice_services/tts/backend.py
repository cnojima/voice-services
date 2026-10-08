"""One upstream GPT-SoVITS process (its own `api_v2.py`), supervised and called over HTTP.

Moved from the `voice` project's voice/gpt_sovits.py (commit 650dd4a) and reshaped for a
service: it no longer swallows startup errors (the pool decides what a failure means) and
no longer knows about Config, languages or Kokoro fallback.

The upstream stack (torch + CUDA, its own venv under `<gpt_sovits_root>/runtime`) is a
different environment from this project's, so it is never imported in-process: it runs as a
subprocess and is talked to over HTTP. Each instance holds one voice at a time;
`switch_voice()` hot-swaps weights on the running process.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
import time
from io import BytesIO

import httpx
import numpy as np

from voice_services.config import TTSConfig
from voice_services.devices import is_windows
from voice_services.tts.voices import load_voice, min_plausible_duration_s

# Always request cuda: this project is CUDA-first, and a CPU TTS service is not a supported
# configuration (upstream would silently degrade to cpu + fp32).
_DEVICE = "cuda"
_BERT_BASE_PATH = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
_CNHUBERT_BASE_PATH = "GPT_SoVITS/pretrained_models/chinese-hubert-base"

_WARMUP_TEXT = "これはウォームアップです。"
_STARTUP_POLL_S = 1.0
SYNTHESIZE_RETRIES = 3

_CUSTOM_CONFIG_TEMPLATE = """\
custom:
  bert_base_path: '{bert_base_path}'
  cnhuhbert_base_path: '{cnhuhbert_base_path}'
  device: {device}
  is_half: true
  t2s_weights_path: '{t2s_weights_path}'
  version: {version}
  vits_weights_path: '{vits_weights_path}'
"""


def upstream_python(gpt_sovits_root: str) -> str:
    sub = ("runtime", "Scripts", "python.exe") if is_windows() else ("runtime", "bin", "python")
    return os.path.join(gpt_sovits_root, *sub)


class UpstreamInstance:
    """One `api_v2.py` on 127.0.0.1:<port>. Not thread-safe: the pool serializes calls per instance."""

    def __init__(self, cfg: TTSConfig, port: int):
        self.cfg = cfg
        self.port = port
        self.voice_name: str | None = None
        self.ready = False
        self._url = f"http://127.0.0.1:{port}"
        self._voice: dict | None = None
        self._proc: subprocess.Popen | None = None
        self._log_path: str | None = None
        self._client = httpx.Client(timeout=30.0)

    def start(self, voice_name: str) -> None:
        """Bring the instance up serving `voice_name`, warmed. Raises on any failure (and
        cleans up), including VoiceNotFoundError for a bad voice. If something already answers
        on the port it is attached to, and told to load the voice."""
        try:
            voice = load_voice(self.cfg.weights_root, voice_name)
            if self._already_running():
                self._set_weights(voice)
            else:
                self._launch(voice)
            self._voice = voice
            self.synthesize(_WARMUP_TEXT, voice["ref_lang"])  # pay cold-start cost now, not on a user's turn
            self.voice_name = voice_name
            self.ready = True
        except BaseException:
            self.close()
            raise

    def switch_voice(self, name: str) -> None:
        """Hot-swap to a different voice's weights on the running process. Raises on failure;
        on success, `.voice_name` reflects the new voice."""
        voice = load_voice(self.cfg.weights_root, name)
        self._set_weights(voice)
        self._voice = voice
        self.voice_name = name

    def _set_weights(self, voice: dict) -> None:
        for endpoint, path in (("set_gpt_weights", voice["gpt_weights"]), ("set_sovits_weights", voice["sovits_weights"])):
            resp = self._client.get(f"{self._url}/{endpoint}", params={"weights_path": path}, timeout=60.0)
            if resp.status_code != 200:
                raise RuntimeError(f"{endpoint} {resp.status_code}: {resp.text[:300]}")

    def _already_running(self) -> bool:
        try:
            self._client.get(self._url, timeout=1.0)
            return True
        except httpx.TransportError:
            return False

    def _write_custom_config(self, voice: dict) -> str:
        text = _CUSTOM_CONFIG_TEMPLATE.format(
            bert_base_path=_BERT_BASE_PATH,
            cnhuhbert_base_path=_CNHUBERT_BASE_PATH,
            device=_DEVICE,
            t2s_weights_path=voice["gpt_weights"],
            vits_weights_path=voice["sovits_weights"],
            version=voice["version"],
        )
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="gpt_sovits_cfg_", delete=False, encoding="utf-8"
        )
        f.write(text)
        f.close()
        return f.name

    def _launch(self, voice: dict) -> None:
        root = self.cfg.gpt_sovits_root
        if not root:
            raise RuntimeError("VOICE_GPT_SOVITS_ROOT is not set (the upstream GPT-SoVITS checkout)")
        python = upstream_python(root)
        if not os.path.isfile(python):
            raise RuntimeError(f"GPT-SoVITS venv python not found: {python}")

        config_path = self._write_custom_config(voice)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([root, os.path.join(root, "GPT_SoVITS")])
        if self.cfg.ffmpeg_bin:
            env["PATH"] = os.pathsep.join([self.cfg.ffmpeg_bin, env.get("PATH", "")])
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        log = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", prefix="gpt_sovits_api_", delete=False, encoding="utf-8"
        )
        self._log_path = log.name
        self._proc = subprocess.Popen(
            [python, "-X", "utf8", "api_v2.py", "-a", "127.0.0.1", "-p", str(self.port), "-c", config_path],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.close)

        timeout_s = self.cfg.startup_timeout_s
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"api_v2.py exited early (code {self._proc.returncode}); log: {self._log_path}"
                )
            if self._already_running():
                return
            time.sleep(_STARTUP_POLL_S)
        raise TimeoutError(f"api_v2.py did not come up within {timeout_s:.0f}s; log: {self._log_path}")

    def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Synthesize `text` (language `language`) in the current voice. Raises on any
        request/decode failure. Retries on an implausibly short result (see
        min_plausible_duration_s) before giving up and returning it anyway."""
        floor_s = min_plausible_duration_s(text)
        last_result = None
        for _attempt in range(SYNTHESIZE_RETRIES):
            wav, sr = self._request_tts(text, language)
            last_result = wav, sr
            if len(wav) / sr >= floor_s:
                return wav, sr
        wav, sr = last_result
        print(
            f"GPT-SoVITS decode stayed implausibly short after {SYNTHESIZE_RETRIES} attempts "
            f"(voice={self.voice_name!r}, got {len(wav) / sr:.2f}s, floor={floor_s:.2f}s); using it anyway",
            file=sys.stderr,
        )
        return last_result

    def _request_tts(self, text: str, language: str) -> tuple[np.ndarray, int]:
        # prompt_lang is the reference clip's own language (fixed per voice); text_lang is the
        # language of the text being spoken.
        resp = self._client.post(
            f"{self._url}/tts",
            json={
                "text": text,
                "text_lang": language,
                "ref_audio_path": self._voice["ref_audio_path"],
                "prompt_text": self._voice["ref_text"],
                "prompt_lang": self._voice["ref_lang"],
                "top_p": 1,
                "temperature": 1,
                "media_type": "wav",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"GPT-SoVITS API {resp.status_code}: {resp.text[:300]}")
        import soundfile as sf

        data, sr = sf.read(BytesIO(resp.content), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), sr

    @property
    def closed(self) -> bool:
        """True once close() has run: the subprocess is gone and the client can't send."""
        return self._client.is_closed

    def close(self) -> None:
        """Terminate the api_v2.py subprocess, if we started one. Idempotent (an explicit close
        and the atexit hook can both call this) and never raises -- this can run during
        interpreter shutdown, where a raised exception is an unactionable traceback."""
        self.ready = False
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:  # noqa: BLE001 -- shutdown must not raise
            pass
        self._proc = None
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
