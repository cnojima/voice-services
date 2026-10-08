"""Project settings from a `.env` file.

Per-machine paths (GPT-SoVITS install, voice weights, ffmpeg) live in a gitignored
`.env` at the repo root instead of in committed scripts. `load_env()` copies the
file's values into `os.environ`, so everything that already reads the environment
(CLI defaults, `huggingface_hub`) picks them up unchanged. Moved from the `voice`
project's voice/env.py (commit 650dd4a).

* The real environment wins: a variable that is already set is never overwritten,
  so `set BRAVE_API_KEY=...` in a terminal still overrides the file for that run.
* It has to run before anything reads those variables: each server's `main()`
  calls it first.
* `VOICE_ENV_FILE` points at a different file; set to an empty string it disables
  loading altogether (tests/conftest.py does this, so a developer's real `.env`
  can't leak into a test run).
* A blank value (`NAME=`) counts as unset, so a copied `.env.example` is harmless.
* Values are never printed, only the names.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_result: tuple[Path, list[str], list[str]] | None = None
_done = False


def env_file() -> Path | None:
    """The file to load: $VOICE_ENV_FILE if set ("" = none), else `.env` in the
    repo root, else `.env` in the current directory."""
    override = os.environ.get("VOICE_ENV_FILE")
    if override is not None:
        return Path(override) if override else None
    for candidate in (REPO_ROOT / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            return candidate
    return None


def load_env(force: bool = False) -> tuple[Path, list[str], list[str]] | None:
    """Load the `.env` into os.environ once. Returns (path, names set, names that
    were already set in the environment and so kept), or None if there is no file."""
    global _result, _done
    if _done and not force:
        return _result
    _done = True
    _result = None

    path = env_file()
    if path is None or not path.is_file():
        return None
    try:
        from dotenv import dotenv_values
    except ImportError:
        print(
            f"Found {path} but python-dotenv isn't installed, so it was ignored. "
            "Run: pip install -e .",
            file=sys.stderr,
        )
        return None

    # utf-8-sig: Windows editors (Notepad) like to prepend a BOM, which would
    # otherwise become part of the first variable's name.
    values = dotenv_values(path, encoding="utf-8-sig")
    set_now: list[str] = []
    kept: list[str] = []
    for name, value in values.items():
        # A bare `NAME`, or `NAME=` left blank (as in .env.example), means "not set":
        # an empty string would otherwise override code defaults that
        # read the variable with os.environ.get().
        if not value:
            continue
        if name in os.environ:
            kept.append(name)
        else:
            os.environ[name] = value
            set_now.append(name)
    _result = (path, set_now, kept)
    return _result


def report() -> str | None:
    """One line for the startup log (names only, never values)."""
    if _result is None:
        return None
    path, set_now, kept = _result
    line = f"Loaded {len(set_now)} setting(s) from {path}"
    if kept:
        line += f" ({len(kept)} already set in the environment, kept: {', '.join(kept)})"
    return line
