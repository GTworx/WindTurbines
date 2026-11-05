"""Defect taxonomy management (``defect_taxonomy.json``).

Holds the canonical list of defect classes, their severity mapping, and a
version tag (e.g. ``v1.2.0``) so the Re-examine Agent can reason about deltas.

The severity scale follows the spec (Section 4.1), aligned with common
structural-risk grading (IEC/DNV-style):

  1 = cosmetic
  2 = minor / monitor
  3 = moderate / plan repair
  4 = severe / urgent repair
  5 = catastrophic failure risk
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SEVERITY_SCALE = {
    1: "cosmetic",
    2: "minor - monitor",
    3: "moderate - plan repair",
    4: "severe - urgent repair",
    5: "catastrophic failure risk",
}

# Seed classes from the spec's examples. The Training Agent appends to this.
DEFAULT_TAXONOMY = {
    "version": "v0.1.0",
    "updated_at": None,
    "severity_scale": SEVERITY_SCALE,
    "classes": [
        {"name": "leading_edge_erosion", "default_severity": 3, "aliases": ["erosion", "leading edge"]},
        {"name": "crack", "default_severity": 4, "aliases": ["cracks"]},
        {"name": "lightning_strike", "default_severity": 4, "aliases": ["lightning", "lightning damage"]},
        {"name": "delamination", "default_severity": 4, "aliases": ["delam", "peel"]},
        {"name": "peel_paint", "default_severity": 1, "aliases": ["peeled paint", "paint peel"]},
        {"name": "dirt_debris", "default_severity": 1, "aliases": ["debris", "dirt"]},
        {"name": "micro_pitting", "default_severity": 2, "aliases": ["pitting"]},
    ],
    "samples": {},  # class name -> list of {path, source}
}

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def bump_patch(version: str) -> str:
    m = _VERSION_RE.match(version)
    if not m:
        return "v0.1.0"
    major, minor, patch = (int(g) for g in m.groups())
    return f"v{major}.{minor}.{patch + 1}"


class TaxonomyManager:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load_or_init(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return dict(DEFAULT_TAXONOMY)

    def save(self, taxonomy: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2)

    def class_names(self, taxonomy: Optional[Dict[str, Any]] = None) -> List[str]:
        tax = taxonomy or self.load_or_init()
        return [c["name"] for c in tax.get("classes", [])]

    def default_severity(self, name: str, taxonomy: Optional[Dict[str, Any]] = None) -> int:
        tax = taxonomy or self.load_or_init()
        for c in tax.get("classes", []):
            if c["name"] == name:
                return int(c.get("default_severity", 3))
        return 3

    def add_class(
        self,
        name: str,
        default_severity: int = 3,
        aliases: Optional[List[str]] = None,
        taxonomy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tax = taxonomy or self.load_or_init()
        name = name.lower().replace(" ", "_")
        if not any(c["name"] == name for c in tax["classes"]):
            tax["classes"].append(
                {"name": name, "default_severity": int(default_severity), "aliases": aliases or []}
            )
            tax["version"] = bump_patch(tax.get("version", "v0.1.0"))
            tax["updated_at"] = datetime.now(timezone.utc).isoformat()
        return tax

    def record_sample(self, class_name: str, path: str, taxonomy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tax = taxonomy or self.load_or_init()
        samples = tax.setdefault("samples", {})
        samples.setdefault(class_name, []).append({"path": path, "source": "training"})
        return tax

    def classes_by_version(self, taxonomy: Dict[str, Any]) -> List[str]:
        return self.class_names(taxonomy)
