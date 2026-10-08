"""A stand-in for UpstreamInstance: no subprocess, no GPU, records what the pool asked of it."""
from __future__ import annotations

import json
import threading

import numpy as np

SR = 32000
MANIFEST = {"gpt_weights": "g.ckpt", "sovits_weights": "s.pth", "ref_audio_path": "r.wav", "ref_text": "hi"}


def make_voices(root, *names, **per_voice):
    """Create <root>/<name>/voice.json for each name; per_voice maps name -> extra manifest keys."""
    for name in names:
        d = root / name
        d.mkdir()
        (d / "voice.json").write_text(json.dumps({**MANIFEST, **per_voice.get(name, {})}), encoding="utf-8")


class FakeInstance:
    created: list["FakeInstance"] = []
    fail_start: set[str] = set()

    def __init__(self, cfg, port):
        self.cfg, self.port = cfg, port
        self.voice_name = None
        self.ready = False
        self.dead = False
        self.closed_ = False
        self.calls: list[tuple] = []
        self.gate: threading.Event | None = None  # when set, synthesize blocks until it is
        FakeInstance.created.append(self)

    @property
    def alive(self):
        return self.ready and not self.dead

    def start(self, voice):
        if voice in FakeInstance.fail_start:
            self.closed_ = True
            raise RuntimeError(f"cannot start {voice}")
        self.calls.append(("start", voice))
        self.voice_name, self.ready = voice, True

    def switch_voice(self, name):
        self.calls.append(("switch", name))
        self.voice_name = name

    def synthesize(self, text, language):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append(("synth", self.voice_name, text, language))
        return np.zeros(SR, dtype=np.float32), SR

    def stream_tts(self, text, language):
        self.calls.append(("stream", self.voice_name, text, language))

        def chunks():
            yield b"\x01\x00" * 100
            yield b"\x02\x00" * 100

        return SR, chunks()

    def close(self):
        self.ready = False
        self.closed_ = True

    @classmethod
    def reset(cls):
        cls.created, cls.fail_start = [], set()
