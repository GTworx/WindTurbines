"""DefectTrainingAgent (spec Section 2.2).

Visual taxonomy ingestion and few-shot library builder. Scans the ``Defects/``
directory for sample images, infers their defect class (from a subfolder name
when present, else from filename heuristics), updates ``defect_taxonomy.json``,
records sample paths, and emits a ``TAXONOMY_UPDATED`` event with the new
version tag so the Orchestrator / Re-examine Agent can react.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import taxonomy as tax
from ..events import EventBus, TAXONOMY_UPDATED
from .base import BaseAgent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# filename heuristics for flat (no-subfolder) Defects dirs
_NAME_HINTS = [
    ("lightning", "lightning_strike"),
    ("delam", "delamination"),
    ("crack", "crack"),
    ("erosion", "leading_edge_erosion"),
    ("peel", "peel_paint"),
    ("pit", "micro_pitting"),
    ("debris", "dirt_debris"),
]


class DefectTrainingAgent(BaseAgent):
    name = "DefectTrainingAgent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tax = tax.TaxonomyManager(self.config.taxonomy_file)

    @staticmethod
    def _class_from_name(filename: str) -> Optional[str]:
        lower = filename.lower()
        for hint, cls in _NAME_HINTS:
            if hint in lower:
                return cls
        return None

    def scan(self) -> Dict[str, List[str]]:
        """Map defect class -> list of sample image paths found under Defects/."""
        defects_dir = self.config.defects_dir
        found: Dict[str, List[str]] = {}
        if not defects_dir.exists():
            return found
        # 1) subfolders: Defects/{class}/*.png
        for sub in sorted(defects_dir.iterdir()):
            if sub.is_dir():
                cls = re.sub(r"[^a-z0-9]+", "_", sub.name.lower()).strip("_") or "unclassified"
                for img in sorted(sub.iterdir()):
                    if img.suffix.lower() in IMAGE_EXTS:
                        found.setdefault(cls, []).append(str(img))
        # 2) flat files: Defects/*.png
        for img in sorted(defects_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                cls = self._class_from_name(img.name) or "unclassified"
                found.setdefault(cls, []).append(str(img))
        return found

    def run(self) -> Dict[str, Any]:
        """Build/refresh the taxonomy from the Defects/ directory."""
        self.log_info("TRAINING_START", "Scanning Defects/ directory for defect samples")
        found = self.scan()
        t = self.tax.load_or_init()
        new_classes: List[str] = []
        for cls, samples in found.items():
            existing = {c["name"] for c in t["classes"]}
            if cls not in existing and cls != "unclassified":
                t = self.tax.add_class(cls, taxonomy=t)
                new_classes.append(cls)
            for s in samples:
                t = self.tax.record_sample(cls, s, taxonomy=t)
        t["updated_at"] = t.get("updated_at")
        self.tax.save(t)
        version = t.get("version")
        self.log_info(
            "TAXONOMY_BUILD",
            "Taxonomy refreshed",
            detail={"version": version, "new_classes": new_classes,
                    "classes": self.tax.class_names(t),
                    "sample_count": sum(len(v) for v in found.values())},
        )
        self.events.publish(
            TAXONOMY_UPDATED,
            {"version": version, "new_classes": new_classes, "classes": self.tax.class_names(t)},
        )
        return {
            "version": version,
            "new_classes": new_classes,
            "classes": self.tax.class_names(t),
            "samples_by_class": {k: len(v) for k, v in found.items()},
        }
