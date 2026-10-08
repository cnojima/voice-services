"""A pool of upstream GPT-SoVITS processes, one voice per instance (contract: docs/TTS_API.md).

Routing: a voice already held by an instance goes there; otherwise a never-used slot is
started with it; otherwise the least recently used slot hot-swaps. Slots work in parallel;
calls to one slot are serialized by its lock. Instances are started lazily (or by `preload`)
and replaced if their process dies.
"""
from __future__ import annotations

import itertools
import threading
from typing import Callable

import httpx

from voice_services.config import TTSConfig
from voice_services.tts.backend import UpstreamInstance
from voice_services.tts.voices import VoiceNotFoundError, load_voice


class TTSUnavailable(RuntimeError):
    """No instance could serve the request (maps to HTTP 503); the message says why."""


class _Slot:
    def __init__(self, port: int):
        self.port = port
        self.lock = threading.Lock()
        self.inst: UpstreamInstance | None = None
        self.target: str | None = None  # voice this slot is (being) assigned; set under the pool lock
        self.last_used = 0
        self.error: str | None = None


class Lease:
    """Exclusive use of one instance, already holding the requested voice. Release it (or use
    it as a context manager) when done; an instance that failed with a transport error or
    whose process died is discarded so the next request starts a fresh one."""

    def __init__(self, pool: "TTSPool", slot: _Slot, swapped: bool):
        self._pool, self._slot, self.swapped = pool, slot, swapped
        self.inst: UpstreamInstance = slot.inst
        self._released = False

    def release(self, failed: BaseException | None = None) -> None:
        if self._released:
            return
        self._released = True
        slot = self._slot
        if failed is not None and (isinstance(failed, httpx.TransportError) or not self.inst.alive):
            self._pool._discard(slot)
        slot.last_used = next(self._pool._ticks)
        slot.lock.release()

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release(exc)


class TTSPool:
    def __init__(
        self,
        cfg: TTSConfig,
        size: int = 2,
        base_port: int = 9890,
        factory: Callable[[TTSConfig, int], UpstreamInstance] = UpstreamInstance,
    ):
        if size < 1:
            raise ValueError("pool size must be at least 1")
        self.cfg = cfg
        self._factory = factory
        self._slots = [_Slot(base_port + i) for i in range(size)]
        self._lock = threading.Lock()
        self._ticks = itertools.count(1)
        self._preload_done = threading.Event()
        self._preload_done.set()  # until preload() says otherwise

    @property
    def size(self) -> int:
        return len(self._slots)

    # -- state ---------------------------------------------------------------------------

    def loaded(self) -> list[str]:
        return [s.inst.voice_name for s in self._slots if s.inst is not None and s.inst.ready]

    def status(self) -> dict:
        instances = [
            {
                "voice": s.inst.voice_name if s.inst is not None and s.inst.ready else None,
                "ready": s.inst is not None and s.inst.ready,
                "busy": s.lock.locked(),
                **({"error": s.error} if s.error else {}),
            }
            for s in self._slots
        ]
        return {
            "status": "degraded" if any(s.error for s in self._slots) else "ok",
            "ready": self._preload_done.is_set(),
            "instances": instances,
        }

    # -- routing -------------------------------------------------------------------------

    def _choose(self, voice: str) -> _Slot:  # call with self._lock held
        for s in self._slots:
            if s.target == voice:
                return s
        for s in self._slots:
            if s.target is None:
                return s
        return min(self._slots, key=lambda s: s.last_used)

    def acquire(self, voice: str) -> Lease:
        """Block until an instance holds `voice` and is exclusively ours. Raises
        VoiceNotFoundError for an unknown voice, TTSUnavailable if no instance could be made to serve."""
        load_voice(self.cfg.weights_root, voice)  # fail fast on an unknown voice, before any swap
        with self._lock:
            slot = self._choose(voice)
            slot.target = voice
            slot.last_used = next(self._ticks)
        slot.lock.acquire()
        try:
            swapped = self._ensure(slot, voice)
        except BaseException:
            slot.lock.release()
            raise
        return Lease(self, slot, swapped)

    def _ensure(self, slot: _Slot, voice: str) -> bool:
        inst = slot.inst
        if inst is not None and not inst.alive:
            self._discard(slot)
            inst = None
        try:
            if inst is None:
                new = self._factory(self.cfg, slot.port)
                new.start(voice)  # closes itself on failure
                slot.inst, slot.error = new, None
                return True
            if inst.voice_name != voice:
                inst.switch_voice(voice)
                slot.error = None
                return True
            return False
        except VoiceNotFoundError:
            self._restore_target(slot)
            raise
        except Exception as e:  # noqa: BLE001 -- surfaced to the client as 503
            slot.error = f"{type(e).__name__}: {e}"
            self._restore_target(slot)
            raise TTSUnavailable(slot.error) from e

    def _restore_target(self, slot: _Slot) -> None:
        with self._lock:
            slot.target = slot.inst.voice_name if slot.inst is not None and slot.inst.ready else None

    def _discard(self, slot: _Slot) -> None:
        if slot.inst is not None:
            slot.inst.close()
        slot.inst = None
        with self._lock:
            slot.target = None

    # -- lifecycle -----------------------------------------------------------------------

    def preload(self, voices: list[str]) -> list[threading.Thread]:
        """Start loading `voices` (one per slot, in order) in the background. `status()["ready"]`
        turns true when every one has finished, successfully or not (failures show as `error`)."""
        if len(voices) > self.size:
            print(f"--preload lists {len(voices)} voices but the pool has {self.size} instances; ignoring "
                  f"{', '.join(voices[self.size:])}", flush=True)
            voices = voices[: self.size]
        if not voices:
            return []
        self._preload_done.clear()
        remaining = [len(voices)]
        count_lock = threading.Lock()

        def load(voice: str) -> None:
            try:
                self.acquire(voice).release()
                print(f"Preloaded {voice}.", flush=True)
            except Exception as e:  # noqa: BLE001 -- recorded on the slot by _ensure; log the rest
                print(f"Preload of {voice!r} failed: {type(e).__name__}: {e}", flush=True)
            finally:
                with count_lock:
                    remaining[0] -= 1
                    if remaining[0] == 0:
                        self._preload_done.set()

        threads = [threading.Thread(target=load, args=(v,), daemon=True, name=f"preload-{v}") for v in voices]
        for t in threads:
            t.start()
        return threads

    def close(self) -> None:
        for slot in self._slots:
            if slot.inst is not None:
                slot.inst.close()
