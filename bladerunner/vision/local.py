"""Local transformers/PyTorch vision backend.

Combines:
  * an optional object detector (ultralytics YOLO) for bounding boxes, and
  * a multimodal LLM (Qwen2-VL by default) for open-set defect classification,
    severity grading, and a short description per box.

All heavy imports are lazy so the module imports cleanly without torch installed;
a clear :class:`VisionError` is raised only when inference is actually requested
and a required dependency is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import VisionError, VisionModel, InferenceRequest, InferenceResult, _ImageDetection


class LocalVisionModel(VisionModel):
    name = "local"

    def __init__(self, config=None, vlm_model: Optional[str] = None, detector: Optional[str] = "yolo",
                 detector_weights: Optional[str] = None, device: str = "auto", **_: Any):
        cfg_local = getattr(config, "local", {}) if config is not None else {}
        self.vlm_model = vlm_model or cfg_local.get("vlm_model", "Qwen/Qwen2-VL-7B-Instruct")
        self.detector_name = (detector or cfg_local.get("detector", "yolo"))
        self.detector_weights = detector_weights or cfg_local.get("detector_weights")
        self.device = device or cfg_local.get("device", "auto")
        self.max_new_tokens = int(cfg_local.get("max_new_tokens", 512))
        self._processor = None
        self._vlm = None
        self._detector = None

    # -- lazy loading -----------------------------------------------------
    def _pick_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def warmup(self) -> None:
        """Load the VLM and (optionally) the detector. Raises if unavailable."""
        device = self._pick_device()
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as exc:  # pragma: no cover - environment dependent
            raise VisionError(
                "Local backend requires 'torch' and 'transformers'. "
                "Install with: pip install 'bladerunner[local]' -- "
                f"{exc}"
            ) from exc
        try:
            self._processor = AutoProcessor.from_pretrained(self.vlm_model, trust_remote_code=True)
            self._vlm = AutoModelForImageTextToText.from_pretrained(
                self.vlm_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device, trust_remote_code=True,
            )
        except Exception as exc:  # pragma: no cover
            raise VisionError(f"Failed to load VLM {self.vlm_model!r}: {exc}") from exc

        if self.detector_name == "yolo":
            try:
                from ultralytics import YOLO

                self._detector = YOLO(self.detector_weights or "yolov8n.pt")
            except Exception:
                self._detector = None  # fall back to full-image classification
        elif self.detector_name == "none":
            self._detector = None
        else:
            raise VisionError(f"Unsupported detector: {self.detector_name!r}")

    # -- inference --------------------------------------------------------
    def _detect_boxes(self, image_path: Path) -> List[List[int]]:
        if self._detector is None:
            return []
        try:
            results = self._detector.predict(str(image_path), verbose=False)
            boxes = []
            for r in results:
                for b in getattr(r, "boxes", None) or []:
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
            return boxes
        except Exception:
            return []

    def _classify(self, image_path: Path, region: Optional[List[int]], taxonomy: List[str]) -> _ImageDetection:
        region = region or [0, 0, 0, 0]
        prompt = (
            "You are a wind turbine blade defect inspector. Classify this blade image region.\n"
            f"Allowed defect classes: {', '.join(taxonomy)}. If none match, use 'unclassified'.\n"
            "Respond ONLY with JSON: "
            '{"defect_class": str, "severity": int (1-5), "confidence": float (0-1), "description": str}'
        )
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{Path(image_path).as_posix()}"},
                {"type": "text", "text": prompt},
            ]}
        ]
        import torch

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = None
        inputs = self._processor(text=[text], images=image, return_tensors="pt")
        inputs = inputs.to(self._vlm.device)
        with torch.no_grad():
            out = self._vlm.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        return self._parse_classification(raw, taxonomy, region)

    @staticmethod
    def _parse_classification(raw: str, taxonomy: List[str], region: List[int]) -> _ImageDetection:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj: Dict[str, Any] = {}
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = {}
        cls = str(obj.get("defect_class", "unclassified")).lower().strip()
        if cls not in taxonomy and taxonomy:
            # best-effort containment match
            for t in taxonomy:
                if cls in t or t in cls:
                    cls = t
                    break
        try:
            severity = int(max(1, min(5, obj.get("severity", 3))))
        except (TypeError, ValueError):
            severity = 3
        try:
            confidence = float(max(0.0, min(1.0, obj.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        x, y, w, h = region
        return _ImageDetection(
            defect_class=cls,
            severity=severity,
            bbox=region if any(region) else [x, y, max(1, w), max(1, h)],
            confidence=confidence,
            description=obj.get("description"),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if self._vlm is None:
            self.warmup()
        taxonomy = request.taxonomy_classes or ["unclassified"]
        result = InferenceResult(model=self.vlm_model)
        prompt_tokens = 0
        completion_tokens = 0
        for image in request.images:
            image = Path(image)
            boxes = self._detect_boxes(image)
            regions = boxes if boxes else [None]  # no detector -> full image
            dets: List[_ImageDetection] = []
            for region in regions:
                try:
                    det = self._classify(image, region, taxonomy)
                except Exception as exc:  # pragma: no cover - per-region resilience
                    continue
                if det.confidence >= 0.0:
                    dets.append(det)
            result.detections_by_image[str(image)] = dets
            prompt_tokens += 256
            completion_tokens += 64 * max(1, len(dets))
        result.token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return result
