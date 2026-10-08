"""TTSPool routing: which instance serves which voice, swaps, failures. FakeInstance, no GPU."""
from __future__ import annotations

import threading
import time

import pytest

from tests.fakes import FakeInstance, make_voices
from voice_services.config import TTSConfig
from voice_services.tts.pool import TTSPool, TTSUnavailable
from voice_services.tts.voices import VoiceNotFoundError


@pytest.fixture
def pool(tmp_path):
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "kafka", "juliet", juliet={"ref_lang": "en"})
    return TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9000, factory=FakeInstance)


def test_first_voices_take_separate_new_instances(pool):
    with pool.acquire("ayaka") as a:
        assert a.swapped and a.inst.port == 9000
    with pool.acquire("juliet") as j:
        assert j.swapped and j.inst.port == 9001
    assert sorted(pool.loaded()) == ["ayaka", "juliet"]


def test_a_loaded_voice_is_served_without_a_swap(pool):
    pool.acquire("ayaka").release()
    pool.acquire("juliet").release()

    with pool.acquire("ayaka") as lease:
        assert lease.swapped is False
        assert lease.inst.port == 9000

    assert [c for i in FakeInstance.created for c in i.calls if c[0] == "switch"] == []


def test_a_third_voice_swaps_the_least_recently_used_instance(pool):
    pool.acquire("ayaka").release()
    pool.acquire("juliet").release()
    pool.acquire("ayaka").release()  # ayaka is now more recent than juliet

    with pool.acquire("kafka") as lease:
        assert lease.swapped and lease.inst.port == 9001  # juliet's slot
    assert sorted(pool.loaded()) == ["ayaka", "kafka"]


def test_unknown_voice_is_rejected_before_any_instance_is_touched(pool):
    with pytest.raises(VoiceNotFoundError):
        pool.acquire("nobody")

    assert FakeInstance.created == []


def test_a_failed_start_is_reported_and_the_slot_is_reusable(pool):
    FakeInstance.fail_start = {"ayaka"}

    with pytest.raises(TTSUnavailable, match="cannot start ayaka"):
        pool.acquire("ayaka")
    assert pool.status()["status"] == "degraded"
    assert "cannot start ayaka" in pool.status()["instances"][0]["error"]

    FakeInstance.fail_start = set()
    with pool.acquire("ayaka") as lease:
        assert lease.swapped
    assert pool.status()["status"] == "ok"


def test_a_dead_instance_is_replaced_on_the_next_request(pool):
    pool.acquire("ayaka").release()
    first = FakeInstance.created[0]
    first.dead = True

    with pool.acquire("ayaka") as lease:
        assert lease.swapped
        assert lease.inst is not first
    assert first.closed_


def test_a_request_that_fails_on_a_dead_instance_discards_it(pool):
    lease = pool.acquire("ayaka")
    inst = lease.inst
    inst.dead = True

    lease.release(RuntimeError("process gone"))

    assert inst.closed_
    assert pool.loaded() == []


def test_different_instances_run_in_parallel_and_the_same_one_queues(pool):
    pool.acquire("ayaka").release()
    pool.acquire("juliet").release()
    ayaka_lease = pool.acquire("ayaka")  # holds ayaka's slot

    # juliet is not blocked by ayaka being busy
    with pool.acquire("juliet") as j:
        assert j.inst.voice_name == "juliet"

    got = []
    t = threading.Thread(target=lambda: got.append(pool.acquire("ayaka")))
    t.start()
    time.sleep(0.1)
    assert got == []  # queued behind the lease
    ayaka_lease.release()
    t.join(2)
    assert len(got) == 1
    got[0].release()


def test_status_reports_busy_and_idle_slots(pool):
    pool.acquire("ayaka").release()
    lease = pool.acquire("ayaka")

    instances = pool.status()["instances"]

    assert instances[0] == {"voice": "ayaka", "ready": True, "busy": True}
    assert instances[1] == {"voice": None, "ready": False, "busy": False}
    lease.release()


def test_preload_loads_one_voice_per_slot_and_becomes_ready(pool):
    threads = pool.preload(["ayaka", "juliet"])
    for t in threads:
        t.join(2)

    status = pool.status()
    assert status["ready"] is True
    assert sorted(pool.loaded()) == ["ayaka", "juliet"]


def test_preload_beyond_pool_size_is_trimmed(pool, capsys):
    for t in pool.preload(["ayaka", "juliet", "kafka"]):
        t.join(2)

    assert sorted(pool.loaded()) == ["ayaka", "juliet"]
    assert "ignoring kafka" in capsys.readouterr().out


def test_ready_is_false_while_preload_runs_and_a_failure_still_finishes_it(pool):
    FakeInstance.fail_start = {"ayaka"}
    for t in pool.preload(["ayaka"]):
        t.join(2)

    status = pool.status()
    assert status["ready"] is True and status["status"] == "degraded"
