import json
from pathlib import Path

import pytest

from bladerunner import chunking, state as state_mod, taxonomy as tax_mod
from bladerunner.metadata import parse_metadata_row
from bladerunner.rate_limit import RateLimiter


# -- chunking --------------------------------------------------------------
def test_chunking_respects_count(project_root):
    images = [project_root / "Defects" / f"img{i}.png" for i in range(6)]
    for i in images:
        i.touch()
    chunks = chunking.chunk_images(images, images_per_chunk=2, max_chunk_tokens=10**9)
    assert [len(c) for c in chunks] == [2, 2, 2]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunking_empty(project_root):
    assert chunking.chunk_images([]) == []


# -- metadata --------------------------------------------------------------
def test_metadata_alias_mapping():
    rec = parse_metadata_row(
        {
            "image_name": "IMG_1.jpg",
            "Span_M": "42.5",
            "Surface": "suction_side",
            "Capture_Time": "2026-10-12T14:22:10Z",
            "random_extra": "keep",
        }
    )
    assert rec.image_name == "IMG_1.jpg"
    assert rec.distance_from_root_m == 42.5
    assert rec.surface_face == "suction_side"
    assert rec.timestamp == "2026-10-12T14:22:10Z"
    assert rec.extra == {"random_extra": "keep"}


# -- state / idempotency ---------------------------------------------------
def test_checkpoint_idempotency(project_root):
    img = project_root / "Defects" / "erosion-sample.png"
    cp = state_mod.CheckpointState(project_root / "Logs" / "cp.json")
    cp.reset_run("run-1", "v0.1.0")
    assert not cp.has_processed(img)
    cp.mark_processed(img)
    assert cp.has_processed(img)
    cp.save()
    loaded = state_mod.CheckpointState(project_root / "Logs" / "cp.json").load()
    assert loaded.has_processed(img)
    assert loaded.run_id == "run-1"


# -- rate limiter ----------------------------------------------------------
def test_rate_limiter_retries_then_succeeds():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return "ok"

    rl = RateLimiter(max_retries=3, base_delay_s=0.0, max_delay_s=0.0,
                     circuit_breaker_threshold=10, sleep=sleeps.append)
    assert rl.call(flaky) == "ok"
    assert calls["n"] == 3


def test_rate_limiter_raises_after_retries():
    def always_fail():
        raise RuntimeError("nope")

    rl = RateLimiter(max_retries=2, base_delay_s=0.0, max_delay_s=0.0,
                     circuit_breaker_threshold=10, sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        rl.call(always_fail)


# -- taxonomy --------------------------------------------------------------
def test_taxonomy_add_and_bump(project_root):
    tp = project_root / "defect_taxonomy.json"
    tp.write_text(json.dumps({"version": "v1.2.0", "classes": [], "samples": {}}), encoding="utf-8")
    tm = tax_mod.TaxonomyManager(tp)
    t = tm.load_or_init()
    t = tm.add_class("micro_pitting", default_severity=2, taxonomy=t)
    assert t["version"] == "v1.2.1"
    assert "micro_pitting" in tm.class_names(t)
    assert tm.default_severity("micro_pitting", t) == 2


def test_bump_patch():
    assert tax_mod.bump_patch("v1.2.0") == "v1.2.1"
    assert tax_mod.bump_patch("bad") == "v0.1.0"
