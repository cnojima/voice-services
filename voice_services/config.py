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
