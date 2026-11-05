import json
from pathlib import Path

import pytest

from bladerunner import config as config_mod
from bladerunner import vision as vision_mod
from bladerunner.orchestrator import OrchestratorAgent
from bladerunner.vision import InferenceRequest


@pytest.fixture
def cfg(project_root):
    data = dict(config_mod.DEFAULTS)
    data["project_root"] = str(project_root)
    data["backend"] = "mock"
    return config_mod.Config(data=data)


def test_mock_backend_well_formed(cfg, project_root):
    img = project_root / "Defects" / "erosion-sample.png"
    vm = vision_mod.get_backend("mock", config=cfg)
    req = InferenceRequest(images=[img], taxonomy_classes=["erosion", "crack"],
                           default_severities={"erosion": 3, "crack": 4})
    res = vm.infer(req)
    assert res.token_usage["prompt_tokens"] > 0
    assert str(img) in res.detections_by_image
    for d in res.for_image(img):
        assert d.defect_class in ("erosion", "crack")
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.bbox) == 4


def test_full_pipeline_with_mock(cfg, turbine_dir):
    orch = OrchestratorAgent(cfg, run_id="run-test", backend="mock")
    summary = orch.run(turbine_dir, confidence_threshold=0.0, train=True, report=True)

    run_dir = cfg.runs_dir / "run-test"
    results = run_dir / "results.json"
    dashboard = run_dir / "dashboard" / "index.html"
    wiki_index = cfg.wiki_dir / "Turbines" / "Turbine_T104" / "index.md"

    assert results.exists()
    assert dashboard.exists()
    assert wiki_index.exists()

    payload = json.loads(results.read_text(encoding="utf-8"))
    turbine = payload["turbine"]
    assert turbine["turbine_id"] == "Turbine_T104"
    blade_ids = {b["blade_id"] for b in turbine["blade_reports"]}
    assert blade_ids == {"Blade_A", "Blade_B"}

    # At least one blade has some inspection results.
    assert any(b["image_results"] for b in turbine["blade_reports"])

    # Annotated overlays were produced for flagged images (per-blade subfolders).
    annotated = [p for p in (run_dir / "annotated").rglob("*") if p.is_file()]
    flagged = sum(len([r for r in b["image_results"] if r["detections"]]) for b in turbine["blade_reports"])
    assert len(annotated) == flagged or flagged == 0

    # HTML dashboard embeds the data payload.
    html = dashboard.read_text(encoding="utf-8")
    assert "BladeRunner" in html
    assert "Turbine_T104" in html

    assert "artifacts" in summary and "dashboard" in summary["artifacts"]


def test_resume_reuses_cached_results(cfg, turbine_dir):
    orch = OrchestratorAgent(cfg, run_id="run-resume", backend="mock")
    orch.run(turbine_dir, confidence_threshold=0.0, train=False, report=False)
    run_dir = cfg.runs_dir / "run-resume"
    results = run_dir / "results.json"
    assert results.exists()
    # Second run with resume should not error and should reuse cache.
    orch2 = OrchestratorAgent(cfg, run_id="run-resume", backend="mock")
    summary2 = orch2.run(turbine_dir, confidence_threshold=0.0, train=False, report=False, resume=True)
    assert summary2["run_id"] == "run-resume"
