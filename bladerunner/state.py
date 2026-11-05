"""State persistence & checkpointing (spec Section 3.2).

Guarantees idempotent, resumable runs:
  * tracks processed image hashes to avoid duplicate inference
  * records the exact interruption point so the Orchestrator can resume
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def image_hash(path: Path) -> str:
    """Stable hash: content hash if the file is small/readable, else mtime+size.

    Using content for small files keeps idempotency robust to path moves; for
    large images hashing the whole file each run is wasteful, so we fall back to
    a cheap (mtime, size) signature.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return "missing"
    if size <= (8 * 1024 * 1024):  # <= 8MB -> hash content
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            return "sha256:" + h.hexdigest()
        except OSError:
            pass
    stat = path.stat()
    return "sig:%d:%d" % (stat.st_mtime, size)


class CheckpointState:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": None,
            "current_turbine": None,
            "current_blade": None,
            "last_processed_image_index": 0,
            "completed_images_hash_list": [],
            "active_taxonomy_version": None,
            "started_at": None,
            "updated_at": None,
        }

    def load(self) -> "CheckpointState":
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                self.data.update(stored)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt -> start fresh
        return self

    def save(self) -> None:
        self.data["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # -- position tracking ------------------------------------------------
    def set_position(
        self,
        turbine: Optional[str],
        blade: Optional[str],
        image_index: int,
    ) -> None:
        self.data["current_turbine"] = turbine
        self.data["current_blade"] = blade
        self.data["last_processed_image_index"] = image_index
        self.save()

    # -- idempotency ------------------------------------------------------
    def has_processed(self, path: Path) -> bool:
        return image_hash(path) in self.completed_hashes

    def mark_processed(self, path: Path) -> None:
        hashes: List[str] = self.data.setdefault("completed_images_hash_list", [])
        h = image_hash(path)
        if h not in hashes:
            hashes.append(h)

    @property
    def completed_hashes(self) -> Set[str]:
        return set(self.data.get("completed_images_hash_list", []))

    def reset_run(self, run_id: str, taxonomy_version: Optional[str] = None) -> None:
        self.data.update(
            {
                "run_id": run_id,
                "current_turbine": None,
                "current_blade": None,
                "last_processed_image_index": 0,
                "completed_images_hash_list": [],
                "active_taxonomy_version": taxonomy_version,
                "started_at": time.time(),
            }
        )
        self.save()

    @property
    def run_id(self) -> Optional[str]:
        return self.data.get("run_id")

    @property
    def active_taxonomy_version(self) -> Optional[str]:
        return self.data.get("active_taxonomy_version")
