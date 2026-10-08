"""Voice manifests. Moved from the `voice` project's voice/gpt_sovits.py (commit 650dd4a).

Voices live outside this repo in the `voice-weights` repo (Git LFS), one directory per voice:

    <weights_root>/<voice>/
        GPT_weights_v2/<name>-eN.ckpt
        SoVITS_weights_v2/<name>_eN_sM.pth
        voice.json   -- {"gpt_weights": "...", "sovits_weights": "...",
                          "ref_audio_path": "...", "ref_text": "...",
                          "version": "v2", "ref_lang": "ja"}
                          (version and ref_lang optional: "v2Pro" for v2Pro-trained
                          voices; ref_lang is the reference clip's own language, "ja"
                          if absent. All three paths are relative to this voice's own
                          directory.)

Adding a voice is just adding a directory; no code changes.
"""
from __future__ import annotations

import json
import os

from voice_services import env


def default_weights_root() -> str:
    """VOICE_WEIGHTS_ROOT if set, else a `voice-weights` dir beside this repo."""
    return os.environ.get("VOICE_WEIGHTS_ROOT") or str(env.REPO_ROOT.parent / "voice-weights")


class VoiceNotFoundError(RuntimeError):
    pass


def min_plausible_duration_s(text: str) -> float:
    """A conservative floor on synthesized duration for `text` -- catches the AR decode's
    rare near-instant-EOS degenerate case (the model's stochastic top-k sampling
    occasionally samples an end-of-sequence token almost immediately by chance) without
    false-triggering on legitimately short lines. Deliberately generous -- real speech is
    much slower than this floor -- so only truly degenerate output trips it."""
    chars = len(text.strip())
    return max(0.3, chars * 0.05)


def list_voices(weights_root: str, language: str | None = None) -> list[str]:
    """Voice names available under weights_root (dirs with a voice.json), sorted.
    With `language`, only voices whose reference clip is in that language."""
    root = os.fspath(weights_root)
    if not os.path.isdir(root):
        return []
    names = sorted(
        name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name, "voice.json"))
    )
    if language is None:
        return names
    found = []
    for name in names:
        try:
            with open(os.path.join(root, name, "voice.json"), encoding="utf-8") as f:
                ref_lang = json.load(f).get("ref_lang", "ja")
        except (OSError, ValueError):
            continue
        if ref_lang == language:
            found.append(name)
    return found


def load_voice(weights_root: str, voice: str) -> dict:
    """The voice's manifest with its three paths resolved relative to the voice directory."""
    voice_dir = os.path.join(weights_root, voice)
    manifest_path = os.path.join(voice_dir, "voice.json")
    if not os.path.isfile(manifest_path):
        raise VoiceNotFoundError(f"no voice.json for {voice!r} at {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("gpt_weights", "sovits_weights", "ref_audio_path", "ref_text"):
        if key not in manifest:
            raise VoiceNotFoundError(f"{manifest_path} missing required key {key!r}")
    manifest.setdefault("version", "v2")
    manifest.setdefault("ref_lang", "ja")
    for key in ("gpt_weights", "sovits_weights", "ref_audio_path"):
        manifest[key] = os.path.normpath(os.path.join(voice_dir, manifest[key]))
    return manifest
