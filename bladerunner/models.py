"""Core data models shared across the BladeRunner agents.

Plain dataclasses are used (no heavy dependency on Pydantic) so the framework
stays importable and testable with a minimal environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MetadataRecord:
    """A single row of a blade ``metadata.csv``.

    Enforces the strict schema contract recommended in the spec (Section 4.2):
    ``image_name`` is required; all other fields are optional but normalized
    to canonical keys here.
    """

    image_name: str
    timestamp: Optional[str] = None
    distance_from_root_m: Optional[float] = None
    surface_face: Optional[str] = None  # e.g. pressure | suction
    edge: Optional[str] = None  # e.g. leading | trailing
    camera_pitch: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    altitude_m: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("extra"):
            d.pop("extra", None)
        return d


@dataclass
class Detection:
    """A single detected defect on a blade image."""

    defect_class: str
    severity: int  # Category 1..5
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    span_position_m: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageResult:
    """Inspection result for one blade image."""

    image_path: str
    image_name: str
    metadata: Optional[MetadataRecord] = None
    detections: List[Detection] = field(default_factory=list)
    chunk_index: Optional[int] = None
    token_usage: Optional[Dict[str, int]] = None

    @property
    def flagged(self) -> bool:
        return any(d.confidence >= 0.5 for d in self.detections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_name": self.image_name,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "detections": [d.to_dict() for d in self.detections],
            "chunk_index": self.chunk_index,
            "token_usage": self.token_usage,
        }


@dataclass
class BladeReport:
    """Aggregated inspection results for one blade."""

    blade_id: str
    turbine_id: str
    image_results: List[ImageResult] = field(default_factory=list)

    @property
    def detections(self) -> List[Detection]:
        out: List[Detection] = []
        for ir in self.image_results:
            out.extend(ir.detections)
        return out

    def defect_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for d in self.detections:
            dist[d.defect_class] = dist.get(d.defect_class, 0) + 1
        return dist

    @property
    def health_index(self) -> float:
        """Simple 0..100 health index: penalize detections weighted by severity.

        Not a certified metric; a deterministic heuristic for ranking/reporting.
        """
        if not self.detections:
            return 100.0
        penalty = sum(min(d.severity, 5) * d.confidence for d in self.detections)
        return max(0.0, round(100.0 - penalty, 2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blade_id": self.blade_id,
            "turbine_id": self.turbine_id,
            "health_index": self.health_index,
            "defect_distribution": self.defect_distribution(),
            "image_results": [ir.to_dict() for ir in self.image_results],
        }


@dataclass
class TurbineReport:
    """Aggregated results for one turbine (all blades)."""

    turbine_id: str
    blade_reports: List[BladeReport] = field(default_factory=list)

    @property
    def detections(self) -> List[Detection]:
        out: List[Detection] = []
        for br in self.blade_reports:
            out.extend(br.detections)
        return out

    @property
    def overall_health_index(self) -> float:
        if not self.blade_reports:
            return 100.0
        return round(
            sum(br.health_index for br in self.blade_reports) / len(self.blade_reports),
            2,
        )

    def defect_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for d in self.detections:
            dist[d.defect_class] = dist.get(d.defect_class, 0) + 1
        return dist

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turbine_id": self.turbine_id,
            "overall_health_index": self.overall_health_index,
            "defect_distribution": self.defect_distribution(),
            "blade_reports": [br.to_dict() for br in self.blade_reports],
        }
