"""ReexamineAgent (spec Section 2.4).

Targeted retrospective evaluation. When the Training Agent publishes a new
defect class or updated classifier version, this agent re-runs inference on the
cached image set and appends *newly identified* defects from the delta classes,
preserving all previously valid detections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Detection, TurbineReport
from ..state import CheckpointState
from ..events import EventBus, REEXAMINE_REQUESTED
from ..vision import VisionModel
from .base import BaseAgent
from .inspector import BladeInspectorAgent


class ReexamineAgent(BaseAgent):
    name = "ReexamineAgent"

    def __init__(
        self,
        config,
        vision: VisionModel,
        event_bus: Optional[EventBus] = None,
        run_id: Optional[str] = None,
        state: Optional[CheckpointState] = None,
        **kwargs: Any,
    ):
        super().__init__(config, event_bus=event_bus, run_id=run_id, **kwargs)
        self.vision = vision
        self.state = state or CheckpointState(config.checkpoint_file)
        self.run_dir = config.runs_dir / (run_id or "reexamine")

    @staticmethod
    def _signature(d: Detection) -> tuple:
        return (d.defect_class, tuple(d.bbox))

    def _existing_signatures(self, prior: Dict[str, Any]) -> set:
        sigs = set()
        for br in prior.get("blade_reports", []):
            for ir in br.get("image_results", []):
                for d in ir.get("detections", []):
                    sigs.add((d["defect_class"], tuple(d["bbox"])))
        return sigs

    def run(
        self,
        turbine_dir: Path,
        prior_results_path: Path,
        delta_classes: List[str],
        confidence_threshold: Optional[float] = None,
    ) -> TurbineReport:
        turbine_dir = Path(turbine_dir)
        threshold = confidence_threshold or float(self.config.inspection.get("confidence_threshold", 0.75))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        prior: Dict[str, Any] = {}
        if prior_results_path and Path(prior_results_path).exists():
            with open(prior_results_path, "r", encoding="utf-8") as f:
                prior = json.load(f).get("turbine", {})
        existing = self._existing_signatures(prior)

        inspector = BladeInspectorAgent(
            config=self.config,
            vision=self.vision,
            event_bus=self.events,
            run_id=self.run_id,
            state=self.state,
        )
        fresh = inspector.inspect_turbine(turbine_dir, confidence_threshold=threshold)

        added: List[Dict[str, Any]] = []
        for br in fresh.blade_reports:
            for ir in br.image_results:
                kept = [d for d in ir.detections if self._signature(d) in existing]
                new: List[Detection] = []
                for d in ir.detections:
                    if d.defect_class not in delta_classes:
                        continue
                    if self._signature(d) in existing:
                        continue
                    existing.add(self._signature(d))
                    new.append(d)
                ir.detections = kept + new
                added.extend(
                    {"blade": br.blade_id, "image": ir.image_name,
                     "class": d.defect_class, "severity": d.severity, "bbox": d.bbox}
                    for d in new
                )

        out = self.run_dir / "results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "reexamined": True,
                       "delta_classes": delta_classes, "turbine": fresh.to_dict()},
                      f, indent=2, default=str)

        self.log_info(
            "REEXAMINE_DONE",
            "Re-examination complete",
            detail={
                "turbine": fresh.turbine_id,
                "delta_classes": delta_classes,
                "newly_added": added,
                "results_file": str(out),
            },
        )
        self.events.publish(REEXAMINE_REQUESTED, {
            "turbine_id": fresh.turbine_id, "added": len(added), "delta_classes": delta_classes,
        })
        return fresh
