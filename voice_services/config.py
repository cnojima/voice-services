"""Settings for the services. Plain dataclasses; env vars only supply CLI defaults
(see .env.example), so tests and library callers never depend on the environment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WhisperConfig:
    model: str = "turbo"
    device: str = "auto"  # auto: cuda if available, else cpu (voice_services/devices.py)
    # Languages the server detects among when a request names neither `language`
    # nor `candidates`.
    languages: tuple[str, ...] = ("en", "ja")


@dataclass
class TTSConfig:
    # Directory of trained voices (<root>/<voice>/voice.json); see tts/voices.py.
    weights_root: str = ""
    # The upstream GPT-SoVITS checkout, with its own CUDA venv under runtime/. No default:
    # the install location differs per machine, so a blank value fails with a clear error.
    gpt_sovits_root: str = ""
    # ffmpeg bin directory for the upstream process (bundled on Windows; usually on PATH elsewhere).
    ffmpeg_bin: str = ""
    startup_timeout_s: float = 60.0
