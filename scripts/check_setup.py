"""Check this machine can run the TTS service, and say what is missing.

    python scripts/check_setup.py

Verifies the upstream GPT-SoVITS checkout (venv python, api_v2.py, pretrained base models),
ffmpeg, CUDA, and the voice weights. It changes nothing; installing the checkout and its CUDA
venv is a manual step (see docs/VOICE_TRAINING.md in the voice project).
"""
import os
import shutil
import sys

from voice_services import env
from voice_services.devices import describe, has_cuda
from voice_services.tts.backend import _BERT_BASE_PATH, _CNHUBERT_BASE_PATH, upstream_python
from voice_services.tts.voices import default_weights_root, list_voices

problems: list[str] = []


def check(ok: bool, good: str, bad: str) -> None:
    print(("ok      " if ok else "MISSING ") + (good if ok else bad))
    if not ok:
        problems.append(bad)


def main() -> int:
    env.load_env()
    print(describe())
    check(has_cuda(), "CUDA available to this python (Whisper STT)", "CUDA not available to this python: STT will run on cpu")

    root = os.environ.get("VOICE_GPT_SOVITS_ROOT", "")
    check(bool(root) and os.path.isdir(root), f"VOICE_GPT_SOVITS_ROOT = {root}", "VOICE_GPT_SOVITS_ROOT is unset or not a directory")
    if root and os.path.isdir(root):
        python = upstream_python(root)
        check(os.path.isfile(python), f"upstream venv python: {python}", f"upstream venv python not found: {python}")
        check(os.path.isfile(os.path.join(root, "api_v2.py")), "api_v2.py", f"api_v2.py not found in {root}")
        for rel in (_BERT_BASE_PATH, _CNHUBERT_BASE_PATH):
            check(os.path.isdir(os.path.join(root, rel)), rel, f"pretrained model missing: {os.path.join(root, rel)}")

    ffmpeg_bin = os.environ.get("VOICE_FFMPEG_BIN", "")
    check(
        bool(shutil.which("ffmpeg")) or (bool(ffmpeg_bin) and os.path.isdir(ffmpeg_bin)),
        "ffmpeg", "ffmpeg not on PATH and VOICE_FFMPEG_BIN not set",
    )

    weights = default_weights_root()
    voices = list_voices(weights)
    check(bool(voices), f"{len(voices)} voice(s) in {weights}: {', '.join(voices)}", f"no voices (voice.json) under {weights}; run scripts/fetch_weights.py")

    print("\nAll good." if not problems else f"\n{len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
