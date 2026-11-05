"""Deterministic mock vision backend.

Produces reproducible detections from the image's byte content so the whole
pipeline (chunking, logging, checkpointing, overlay, reporting) is exercisable
without a GPU or model weights. It is NOT a real detector -- use the ``local``
backend for genuine inspection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import VisionModel, InferenceRequest, InferenceResult, _ImageDetection


class MockVisionModel(VisionModel):
    name = "mock"

    def __init__(self, config=None, seed: int = 1, **_: Any):
        self.config = config
        self.seed = seed

    def _image_size(self, path: Path):
        try:
            from PIL import Image

            with Image.open(path) as im:
                return im.size
        except Exception:
            return (1024, 768)

    def _deterministic(self, path: Path) -> bytes:
        h = hashlib.sha256()
        h.update(str(path).encode("utf-8"))
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 64), b""):
                    h.update(block)
        except OSError:
            pass
        return h.digest()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        result = InferenceResult(model="mock")
        classes = request.taxonomy_classes or ["unclassified"]
        defaults = request.default_severities or {}
        prompt_tokens = 0
        completion_tokens = 0

        for image in request.images:
            image = Path(image)
            digest = self._deterministic(image)
            w, h = self._image_size(image)
            # Number of detections is deterministic (0..3) from the digest.
            n_detections = digest[0] % 4
            dets: List[_ImageDetection] = []
            for i in range(n_detections):
                cls = classes[(digest[i] + i) % len(classes)]
                severity = int(defaults.get(cls, 3))
                seed_bytes = digest[(i + 1) % len(digest): (i + 1) % len(digest) + 4]
                rx, ry, rw, rh = (b for b in seed_bytes)
                bw = max(8, min(w, int(20 + (rw % 60) * (w // 100))))
                bh = max(8, min(h, int(20 + (ry % 60) * (h // 100))))
                x = int((rw % 100) * w // 100)
                y = int((ry % 100) * h // 100)
                confidence = round(0.55 + ((digest[0] + i) % 45) / 100.0, 3)
                dets.append(
                    _ImageDetection(
                        defect_class=cls,
                        severity=severity,
                        bbox=[x, y, bw, bh],
                        confidence=confidence,
                        description="mock detection (not a real detector)",
                    )
                )
            result.detections_by_image[str(image)] = dets
            prompt_tokens += 256 + 8 * n_detections
            completion_tokens += 32 * max(1, n_detections)

        result.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return result
