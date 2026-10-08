"""voice.json manifest loading and the implausible-duration floor (voice_services/tts/voices.py).

Model-free: only path resolution, error handling and arithmetic.
"""
from __future__ import annotations

import json

import pytest

from voice_services.tts.voices import VoiceNotFoundError, list_voices, load_voice, min_plausible_duration_s

BASE = {"gpt_weights": "g.ckpt", "sovits_weights": "s.pth", "ref_audio_path": "r.wav", "ref_text": "x"}


def _write_manifest(tmp_path, voice: str, data: dict) -> None:
    voice_dir = tmp_path / voice
    voice_dir.mkdir()
    (voice_dir / "voice.json").write_text(json.dumps(data), encoding="utf-8")


def test_load_voice_resolves_weight_paths_relative_to_its_own_directory(tmp_path):
    _write_manifest(
        tmp_path,
        "raidenshogun",
        {
            "gpt_weights": "GPT_weights_v2/raidenshogun-e20.ckpt",
            "sovits_weights": "SoVITS_weights_v2/raidenshogun_e20_s420.pth",
            "ref_audio_path": "ref.wav",
            "ref_text": "テスト",
        },
    )

    manifest = load_voice(str(tmp_path), "raidenshogun")

    assert manifest["gpt_weights"] == str(tmp_path / "raidenshogun" / "GPT_weights_v2" / "raidenshogun-e20.ckpt")
    assert manifest["sovits_weights"] == str(tmp_path / "raidenshogun" / "SoVITS_weights_v2" / "raidenshogun_e20_s420.pth")
    assert manifest["ref_audio_path"] == str(tmp_path / "raidenshogun" / "ref.wav")
    assert manifest["ref_text"] == "テスト"


def test_load_voice_raises_when_voice_directory_or_manifest_is_missing(tmp_path):
    with pytest.raises(VoiceNotFoundError):
        load_voice(str(tmp_path), "nonexistent_voice")


def test_load_voice_raises_when_manifest_missing_a_required_key(tmp_path):
    _write_manifest(tmp_path, "incomplete", {"gpt_weights": "a.ckpt", "sovits_weights": "b.pth"})

    with pytest.raises(VoiceNotFoundError):
        load_voice(str(tmp_path), "incomplete")


def test_min_plausible_duration_has_a_floor_for_very_short_text():
    assert min_plausible_duration_s("") == 0.3
    assert min_plausible_duration_s("a") == 0.3


def test_min_plausible_duration_scales_with_text_length():
    long_text = "a much, much longer line of text than the short one above"
    assert min_plausible_duration_s(long_text) > min_plausible_duration_s("short line")
    assert min_plausible_duration_s(long_text) == pytest.approx(len(long_text) * 0.05)


def test_load_voice_version_defaults_to_v2_and_honors_override(tmp_path):
    _write_manifest(tmp_path, "old", BASE)
    _write_manifest(tmp_path, "pro", {**BASE, "version": "v2Pro"})
    assert load_voice(str(tmp_path), "old")["version"] == "v2"
    assert load_voice(str(tmp_path), "pro")["version"] == "v2Pro"


def test_load_voice_ref_lang_defaults_to_ja_and_honors_override(tmp_path):
    _write_manifest(tmp_path, "kafka", BASE)
    _write_manifest(tmp_path, "juliet", {**BASE, "ref_lang": "en"})
    assert load_voice(str(tmp_path), "kafka")["ref_lang"] == "ja"
    assert load_voice(str(tmp_path), "juliet")["ref_lang"] == "en"


def test_list_voices_can_filter_by_reference_language(tmp_path):
    _write_manifest(tmp_path, "kafka", BASE)
    _write_manifest(tmp_path, "ayaka", {**BASE, "ref_lang": "ja"})
    _write_manifest(tmp_path, "juliet", {**BASE, "ref_lang": "en"})
    assert list_voices(str(tmp_path)) == ["ayaka", "juliet", "kafka"]
    assert list_voices(str(tmp_path), "ja") == ["ayaka", "kafka"]
    assert list_voices(str(tmp_path), "en") == ["juliet"]


def test_list_voices_of_a_missing_root_is_empty(tmp_path):
    assert list_voices(str(tmp_path / "nope")) == []
