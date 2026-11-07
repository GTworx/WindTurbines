"""BladeInspectorAgent (spec Section 2.3).

The surface-defect detector and metadata correlator. For each blade it parses
``metadata.csv``, splits images into token-bounded chunks, runs vision
inspection against the active defect taxonomy, filters by confidence, and
publishes detection records to the run datastore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import taxonomy as tax
from ..chunking import chunk_images
from ..metadata import find_metadata_file, load_metadata
from ..models import BladeReport, Detection, ImageResult, TurbineReport
from ..state import CheckpointState
from ..events import EventBus, BLADE_INSPECTED, TURBINE_INSPECTED
from ..vision import VisionModel, InferenceRequest
from .base import BaseAgent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def discover_blades(turbine_dir: Path) -> List[Path]:
    """Return blade directories under a turbine directory.

    A blade dir is any subdirectory that contains images or a metadata CSV.
    Falls back to treating ``turbine_dir`` itself as a single blade if it holds
    images directly.
    """
    turbine_dir = Path(turbine_dir)
    if not turbine_dir.exists():
        return []
    blades: List[Path] = []
    for sub in sorted(turbine_dir.iterdir()):
        if not sub.is_dir():
            continue
        if _has_images(sub) or find_metadata_file(sub) is not None:
            blades.append(sub)
    if not blades and (_has_images(turbine_dir) or find_metadata_file(turbine_dir) is not None):
        blades = [turbine_dir]
    return blades


def _has_images(blade_dir: Path) -> bool:
    images_dir = blade_dir / "images"
    if images_dir.is_dir():
        return any(_is_image(i) for i in images_dir.iterdir())
    return any(_is_image(i) for i in blade_dir.iterdir())


def blade_images(blade_dir: Path) -> List[Path]:
    images_dir = blade_dir / "images"
    pool = images_dir.iterdir() if images_dir.is_dir() else blade_dir.iterdir()
    imgs = [i for i in sorted(pool) if _is_image(i)]
    return imgs


class BladeInspectorAgent(BaseAgent):
    name = "BladeInspectorAgent"

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
        self.tax = tax.TaxonomyManager(config.taxonomy_file)
        self.state = state or CheckpointState(config.checkpoint_file)
        self.run_dir = config.runs_dir / (run_id or "adhoc")

    def _taxonomy_snapshot(self) -> Dict[str, Any]:
        t = self.tax.load_or_init()
        return {
            "version": t.get("version"),
            "classes": [c["name"] for c in t.get("classes", [])],
            "default_severities": {c["name"]: int(c.get("default_severity", 3)) for c in t.get("classes", [])},
        }

    def inspect_blade(
        self,
        blade_dir: Path,
        turbine_id: str,
        confidence_threshold: float,
        run_taxonomy: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> BladeReport:
        blade_id = blade_dir.name
        images = blade_images(blade_dir)
        if limit is not None and limit > 0:
            images = images[: int(limit)]
        meta = load_metadata(blade_dir)
        tax = self._taxonomy_snapshot()
        chunk_cfg = self.config.chunking

        report = BladeReport(blade_id=blade_id, turbine_id=turbine_id)
        if not images:
            self.log_info("BLADE_EMPTY", "No images found", detail={"blade": blade_id})
            return report

        chunks = chunk_images(
            images,
            images_per_chunk=chunk_cfg.get("images_per_chunk", 6),
            max_chunk_tokens=chunk_cfg.get("max_chunk_tokens", 8000),
        )

        for chunk in chunks:
            self.state.set_position(turbine_id, blade_id, chunk.index)
            metadata_by_image = {
                str(img): meta[img.name].to_dict() if img.name in meta else None
                for img in chunk.images
            }
            request = InferenceRequest(
                images=list(chunk.images),
                taxonomy_classes=tax["classes"] or ["unclassified"],
                default_severities=tax["default_severities"],
                confidence_threshold=confidence_threshold,
                metadata_by_image=metadata_by_image,
            )
            result = self.rate.call(self.vision.infer, request)

            for img in chunk.images:
                ir = ImageResult(
                    image_path=str(img),
                    image_name=img.name,
                    metadata=meta.get(img.name),
                    chunk_index=chunk.index,
                    token_usage=result.token_usage,
                )
                for d in result.for_image(img):
                    confidence = d.confidence
                    if confidence < confidence_threshold:
                        continue
                    md = meta.get(img.name)
                    span = md.distance_from_root_m if md else None
                    ir.detections.append(
                        Detection(
                            defect_class=d.defect_class,
                            severity=d.severity,
                            bbox=d.bbox,
                            confidence=confidence,
                            span_position_m=span,
                            description=d.description,
                        )
                    )
                    self.state.mark_processed(img)
                report.image_results.append(ir)
            self.state.save()
            self.log_info(
                "BLADE_CHUNK_INSPECTED",
                f"Blade {blade_id} chunk {chunk.index}: {len(chunk)} images",
                detail={
                    "blade": blade_id,
                    "chunk_index": chunk.index,
                    "n_images": len(chunk),
                    "token_usage": result.token_usage,
                    "detections": sum(len(r.detections) for r in report.image_results if r.chunk_index == chunk.index),
                },
            )

        self.state.save()
        self.log_info(
            "BLADE_INSPECTED",
            f"Blade {blade_id} inspected",
            detail={
                "blade": blade_id,
                "turbine": turbine_id,
                "n_images": len(images),
                "n_detections": len(report.detections),
                "health_index": report.health_index,
                "defect_distribution": report.defect_distribution(),
            },
        )
        self.events.publish(
            BLADE_INSPECTED,
            {"turbine_id": turbine_id, "blade_id": blade_id,
             "n_detections": len(report.detections)},
        )
        return report

    def inspect_turbine(
        self,
        turbine_dir: Path,
        confidence_threshold: Optional[float] = None,
        resume: bool = False,
        blade: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> TurbineReport:
        turbine_dir = Path(turbine_dir)
        turbine_id = turbine_dir.name
        threshold = confidence_threshold or float(self.config.inspection.get("confidence_threshold", 0.75))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        run_taxonomy = self._taxonomy_snapshot()
        if self.run_id is None:
            self.run_id = turbine_id

        self.state.reset_run(self.run_id, run_taxonomy["version"])
        turbine = TurbineReport(turbine_id=turbine_id)
        blades = discover_blades(turbine_dir)
        if blade:
            selected = [b for b in blades if b.name == blade]
            if not selected:
                self.log_error("BLADE_NOT_FOUND", "Requested blade not found",
                               detail={"blade": blade, "available": [b.name for b in blades]})
                raise FileNotFoundError(
                    f"Blade {blade!r} not found under {turbine_dir} "
                    f"(available: {', '.join(b.name for b in blades)})"
                )
            blades = selected
        self.log_info(
            "INSPECT_START",
            "Turbine inspection started",
            detail={"turbine": turbine_id, "blades": [b.name for b in blades],
                    "confidence_threshold": threshold,
                    "backend": getattr(self.vision, "name", "unknown"),
                    "taxonomy_version": run_taxonomy["version"]},
        )
        for blade_dir in blades:
            turbine.blade_reports.append(
                self.inspect_blade(blade_dir, turbine_id, threshold, run_taxonomy, limit=limit)
            )

        self._persist(turbine)
        self.log_info(
            "INSPECT_DONE",
            "Turbine inspection complete",
            detail={
                "turbine": turbine_id,
                "overall_health_index": turbine.overall_health_index,
                "defect_distribution": turbine.defect_distribution(),
                "results_file": str(self.run_dir / "results.json"),
            },
        )
        self.events.publish(
            TURBINE_INSPECTED,
            {"turbine_id": turbine_id,
             "overall_health_index": turbine.overall_health_index,
             "defect_distribution": turbine.defect_distribution()},
        )
        return turbine

    def _persist(self, turbine: TurbineReport) -> None:
        out = self.run_dir / "results.json"
        payload = {
            "run_id": self.run_id,
            "turbine": turbine.to_dict(),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
