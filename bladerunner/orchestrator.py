"""OrchestratorAgent (spec Section 2.1).

Primary coordinator / scheduler / state manager. Wires the event bus, manages
the vision backend, and drives the train -> inspect -> report pipeline with
checkpointing and best-effort resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from . import logging_setup
from .agents import (
    BladeInspectorAgent,
    DefectTrainingAgent,
    ReportingAgent,
    ReexamineAgent,
)
from .config import Config
from .events import EventBus
from .logging_setup import BladeLogger
from .state import CheckpointState
from .vision import VisionModel, get_backend


class OrchestratorAgent:
    name = "OrchestratorAgent"

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
        run_id: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.config = config
        self.events = event_bus or EventBus()
        self.run_id = run_id
        logging_setup.BladeLogger.setup(config.log_file)
        self.log = BladeLogger()
        self.backend_name = backend or config.backend
        self.state = CheckpointState(config.checkpoint_file)
        self.vision: VisionModel = get_backend(self.backend_name, config=config)
        self.training = DefectTrainingAgent(config, event_bus=self.events, run_id=self.run_id, logger=self.log)
        self.inspector = BladeInspectorAgent(
            config, vision=self.vision, event_bus=self.events, run_id=self.run_id,
            state=self.state, logger=self.log,
        )
        self.reexamine = ReexamineAgent(
            config, vision=self.vision, event_bus=self.events, run_id=self.run_id,
            state=self.state, logger=self.log,
        )
        self.reporting = ReportingAgent(config, event_bus=self.events, run_id=self.run_id, logger=self.log)

    def _cached_results(self, turbine_dir: Path) -> Optional[Path]:
        run_dir = self.config.runs_dir / (self.run_id or Path(turbine_dir).name)
        p = run_dir / "results.json"
        return p if p.exists() else None

    def run(
        self,
        turbine_dir: Path,
        confidence_threshold: Optional[float] = None,
        train: bool = True,
        report: bool = True,
        resume: bool = False,
    ) -> Dict[str, Any]:
        turbine_dir = Path(turbine_dir)
        if self.run_id is None:
            self.run_id = turbine_dir.name
        self.config.project_root  # ensure resolved
        self.log.info(self.name, "ORCHESTRATE_START", "Pipeline start",
                      detail={"turbine": turbine_dir.name, "backend": self.backend_name,
                              "run_id": self.run_id, "train": train, "resume": resume})

        summary: Dict[str, Any] = {"run_id": self.run_id, "turbine": turbine_dir.name,
                                   "backend": self.backend_name}

        # 1. Training (taxonomy build).
        if train:
            summary["taxonomy"] = self.training.run()

        # 2. Inspection (with best-effort resume: reuse cached results).
        cached = self._cached_results(turbine_dir)
        if resume and cached:
            self.log.warning(self.name, "RESUME_CACHED", "Reusing cached results (resume)",
                             detail={"cached": str(cached)})
            with open(cached, "r", encoding="utf-8") as f:
                summary["inspected"] = json.load(f).get("turbine", {}).get("overall_health_index")
            turbine_report = self._load_report(cached)
        else:
            turbine_report = self.inspector.inspect_turbine(
                turbine_dir, confidence_threshold=confidence_threshold, resume=resume
            )

        # 3. Reporting.
        artifacts: Dict[str, Any] = {}
        if report:
            artifacts = self.reporting.generate(turbine_report)
            summary["artifacts"] = artifacts

        summary["health_index"] = turbine_report.overall_health_index
        summary["defect_distribution"] = turbine_report.defect_distribution()
        self.log.info(self.name, "ORCHESTRATE_DONE", "Pipeline complete",
                      detail={"health_index": summary["health_index"],
                              "defect_distribution": summary["defect_distribution"]})
        return summary

    def _load_report(self, path: Path):
        from .models import Detection, MetadataRecord, TurbineReport, BladeReport, ImageResult

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f).get("turbine", {})
        turbine = TurbineReport(turbine_id=raw.get("turbine_id", ""))
        for br_raw in raw.get("blade_reports", []):
            br = BladeReport(blade_id=br_raw["blade_id"], turbine_id=raw.get("turbine_id", ""))
            for ir_raw in br_raw.get("image_results", []):
                ir = ImageResult(
                    image_path=ir_raw["image_path"],
                    image_name=ir_raw["image_name"],
                    chunk_index=ir_raw.get("chunk_index"),
                )
                for d in ir_raw.get("detections", []):
                    ir.detections.append(Detection(**d))
                br.image_results.append(ir)
            turbine.blade_reports.append(br)
        return turbine
