"""voice_services/env.py loading, and the guard that every env var the code reads is
documented in .env.example."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from voice_services import env

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "VOICE_ENV_FILE"})
    monkeypatch.setattr(env, "_done", False)
    monkeypatch.setattr(env, "_result", None)
    return os.environ


def test_loads_values_and_real_environment_wins(clean_env, tmp_path):
    f = tmp_path / ".env"
    f.write_text("VOICE_STT_PORT=9000\nVOICE_STT_MODEL=small\n", encoding="utf-8")
    clean_env["VOICE_ENV_FILE"] = str(f)
    clean_env["VOICE_STT_MODEL"] = "base"

    _, set_now, kept = env.load_env(force=True)

    assert clean_env["VOICE_STT_PORT"] == "9000"
    assert clean_env["VOICE_STT_MODEL"] == "base"
    assert set_now == ["VOICE_STT_PORT"] and kept == ["VOICE_STT_MODEL"]


def test_blank_value_counts_as_unset(clean_env, tmp_path):
    f = tmp_path / ".env"
    f.write_text("VOICE_STT_PORT=\n", encoding="utf-8")
    clean_env["VOICE_ENV_FILE"] = str(f)

    env.load_env(force=True)

    assert "VOICE_STT_PORT" not in clean_env


def test_empty_override_disables_loading(clean_env):
    clean_env["VOICE_ENV_FILE"] = ""
    assert env.load_env(force=True) is None


def test_every_environment_read_is_in_env_example():
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (REPO / ".env.example").read_text(), re.M))
    read = set()
    for path in (REPO / "voice_services").rglob("*.py"):
        read |= set(re.findall(r"environ(?:\.get)?[\[(]\s*[\"']([A-Z][A-Z0-9_]+)[\"']", path.read_text(encoding="utf-8")))
    assert read - documented == set(), "add to .env.example"
