"""Vision model backends for the Inspector / Re-examine agents.

  * :class:`VisionModel` -- the interface all backends implement.
  * :class:`MockVisionModel` -- deterministic, dependency-light backend used for
    dry runs and tests (no GPU / model weights required).
  * :class:`LocalVisionModel` -- local transformers/PyTorch backend (Qwen-VL
    for classification + optional YOLO detector for bounding boxes).
"""

from .base import VisionModel, InferenceRequest, InferenceResult, VisionError
from .mock import MockVisionModel
from .local import LocalVisionModel

__all__ = [
    "VisionModel",
    "InferenceRequest",
    "InferenceResult",
    "VisionError",
    "MockVisionModel",
    "LocalVisionModel",
]


def get_backend(name: str, config=None, **kwargs) -> "VisionModel":
    """Factory: return a backend by name."""
    name = (name or "mock").lower()
    if name == "mock":
        return MockVisionModel(config=config, **kwargs)
    if name in ("local", "transformers", "pytorch"):
        return LocalVisionModel(config=config, **kwargs)
    raise VisionError(f"Unknown vision backend: {name!r} (expected 'mock' or 'local')")
