"""Vision backend interface and request/result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class VisionError(RuntimeError):
    """Raised when a vision backend cannot perform inference."""


class VisionModel(ABC):
    """Interface implemented by all vision backends."""

    name: str = "abstract"

    @abstractmethod
    def infer(self, request: "InferenceRequest") -> "InferenceResult":
        """Run inspection inference over the images in ``request``."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional hook to load models; no-op by default."""
        return None


@dataclass
class InferenceRequest:
    """One inspection request: a chunk of images + the active taxonomy."""

    images: List[Path]
    taxonomy_classes: List[str]
    default_severities: Optional[Dict[str, int]] = None
    confidence_threshold: float = 0.75
    metadata_by_image: Optional[Dict[str, Dict[str, Any]]] = None

    def __len__(self) -> int:
        return len(self.images)


@dataclass
class _ImageDetection:
    defect_class: str
    severity: int
    bbox: List[int]
    confidence: float
    description: Optional[str] = None


@dataclass
class InferenceResult:
    """Result for one request: per-image detections + token accounting."""

    detections_by_image: Dict[str, List[_ImageDetection]] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    model: str = "unknown"

    def for_image(self, image_path: Path) -> List[_ImageDetection]:
        return self.detections_by_image.get(str(image_path), [])
