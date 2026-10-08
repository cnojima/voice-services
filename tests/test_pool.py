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


# -- pinned voices ---------------------------------------------------------------------------------


@pytest.fixture
def pinned_pool(tmp_path):
    """juliet (en) pinned in a pool of 2, so one instance is left to swap."""
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "kafka", "hutao", "juliet", juliet={"ref_lang": "en"})
    return TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9000, factory=FakeInstance, pinned=["juliet"])


def test_a_pinned_voice_is_never_swapped_out(pinned_pool):
    pool = pinned_pool
    for voice in ("ayaka", "kafka", "hutao", "ayaka"):
        pool.acquire(voice).release()
    pool.acquire("juliet").release()  # juliet is touched least recently of all of them

    # every other voice shared the one free instance, swapping each time; juliet's never moved
    by_port = {i.port: i for i in FakeInstance.created}
    assert "juliet" in pool.loaded()
    assert [c for c in by_port[9000].calls if c[0] == "switch"] == []  # the pinned instance (port 9000) never swapped
    assert [c[1] for c in by_port[9001].calls if c[0] == "switch"] == ["kafka", "hutao", "ayaka"]  # the free one did
    assert pool.loaded().count("juliet") == 1 and len(pool.loaded()) == 2


def test_a_pinned_voice_wins_even_when_it_is_the_oldest(pinned_pool):
    pool = pinned_pool
    pool.acquire("juliet").release()
    pool.acquire("ayaka").release()  # juliet is now the least recently used...

    with pool.acquire("kafka") as lease:  # ...but it is pinned, so ayaka's slot is the one that swaps
        assert lease.swapped and lease.inst.port == 9001
    assert sorted(pool.loaded()) == ["juliet", "kafka"]


def test_requests_for_the_pinned_voice_go_to_its_own_instance(pinned_pool):
    pool = pinned_pool
    pool.acquire("ayaka").release()  # ayaka must not take the pinned slot, even though it was asked for first

    with pool.acquire("ayaka") as a:
        assert a.inst.port == 9001
    with pool.acquire("juliet") as j:
        assert j.swapped and j.inst.port == 9000  # first start of the pinned slot
    with pool.acquire("juliet") as j:
        assert j.swapped is False and j.inst.port == 9000


def test_preload_always_loads_the_pinned_voices_first(pinned_pool):
    for t in pinned_pool.preload(["ayaka"]):
        t.join(2)
    assert sorted(pinned_pool.loaded()) == ["ayaka", "juliet"]

    for t in pinned_pool.preload([]):
        t.join(2)
    assert pinned_pool.status()["ready"] is True


def test_preload_does_not_let_other_voices_push_out_a_pin(pinned_pool, capsys):
    for t in pinned_pool.preload(["ayaka", "kafka"]):
        t.join(2)

    assert sorted(pinned_pool.loaded()) == ["ayaka", "juliet"]
    assert "ignoring kafka" in capsys.readouterr().out


def test_a_pinned_slot_that_failed_to_start_stays_spoken_for(pinned_pool):
    pool = pinned_pool
    FakeInstance.fail_start = {"juliet"}
    with pytest.raises(TTSUnavailable):
        pool.acquire("juliet")

    # other voices must not move into the pinned slot while it is down
    with pool.acquire("ayaka") as a:
        assert a.inst.port == 9001
    with pool.acquire("kafka") as k:
        assert k.inst.port == 9001

    FakeInstance.fail_start = set()
    with pool.acquire("juliet") as j:
        assert j.swapped and j.inst.port == 9000
    assert pool.status()["status"] == "ok"


def test_a_dead_pinned_instance_is_restarted_in_place(pinned_pool):
    pool = pinned_pool
    pool.acquire("juliet").release()
    first = FakeInstance.created[0]
    first.dead = True
    pool.acquire("ayaka").release()  # a request for something else does not claim the dead pinned slot

    assert pool.status()["instances"][0].get("pinned") is True
    with pool.acquire("juliet") as lease:
        assert lease.swapped and lease.inst is not first and lease.inst.port == 9000


def test_status_and_pinned_report_the_pin(pinned_pool):
    pinned_pool.acquire("ayaka").release()

    assert pinned_pool.pinned() == ["juliet"]
    instances = pinned_pool.status()["instances"]
    assert instances[0] == {"voice": None, "ready": False, "busy": False, "pinned": True}
    assert "pinned" not in instances[1]


def test_pinning_the_whole_pool_is_refused(tmp_path):
    FakeInstance.reset()
    with pytest.raises(ValueError, match="at least one instance must stay free"):
        TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, factory=FakeInstance, pinned=["a", "b"])
    with pytest.raises(ValueError, match="at least one instance must stay free"):
        TTSPool(TTSConfig(weights_root=str(tmp_path)), size=1, factory=FakeInstance, pinned=["a"])


def test_the_same_voice_pinned_twice_counts_once(tmp_path):
    FakeInstance.reset()
    pool = TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, factory=FakeInstance, pinned=["a", "a"])
    assert pool.pinned() == ["a"]
