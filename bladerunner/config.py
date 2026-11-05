"""Configuration for BladeRunner.

Loads from a JSON config file with sensible defaults. All paths are resolved
relative to the project root (the directory containing the ``Defects/``,
``Logs/`` and ``Wiki/`` folders) unless given as absolute paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULTS: Dict[str, Any] = {
    "project_root": ".",
    "backend": "mock",  # mock | local
    "local": {
        "vlm_model": "Qwen/Qwen2-VL-7B-Instruct",
        "detector": "yolo",  # yolo | none
        "detector_weights": None,
        "device": "auto",  # auto | cuda | cpu
        "max_new_tokens": 512,
    },
    "chunking": {
        "images_per_chunk": 6,
        "max_chunk_tokens": 8000,
    },
    "inspection": {
        "confidence_threshold": 0.75,
        "min_severity_to_report": 1,
    },
    "paths": {
        "defects_dir": "Defects",
        "logs_dir": "Logs",
        "wiki_dir": "Wiki",
        "runs_dir": "Runs",
    },
    "taxonomy_file": "defect_taxonomy.json",
    "rate_limit": {
        "max_retries": 5,
        "base_delay_s": 1.0,
        "max_delay_s": 60.0,
        "circuit_breaker_threshold": 5,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        data = dict(DEFAULTS)
        if path:
            p = Path(path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                data = _deep_merge(DEFAULTS, user_cfg)
            else:
                raise FileNotFoundError(f"Config file not found: {p}")
        return cls(data=data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    # -- typed helpers ---------------------------------------------------
    @property
    def project_root(self) -> Path:
        return Path(self.data["project_root"]).resolve()

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def defects_dir(self) -> Path:
        return self._resolve(self.data["paths"]["defects_dir"])

    @property
    def logs_dir(self) -> Path:
        return self._resolve(self.data["paths"]["logs_dir"])

    @property
    def wiki_dir(self) -> Path:
        return self._resolve(self.data["paths"]["wiki_dir"])

    @property
    def runs_dir(self) -> Path:
        return self._resolve(self.data["paths"]["runs_dir"])

    @property
    def taxonomy_file(self) -> Path:
        return self._resolve(self.data["taxonomy_file"])

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "BladeRunner.log"

    @property
    def checkpoint_file(self) -> Path:
        return self.logs_dir / ".checkpoint_state.json"

    @property
    def backend(self) -> str:
        return self.data["backend"]

    @property
    def local(self) -> Dict[str, Any]:
        return self.data["local"]

    @property
    def chunking(self) -> Dict[str, Any]:
        return self.data["chunking"]

    @property
    def inspection(self) -> Dict[str, Any]:
        return self.data["inspection"]

    @property
    def rate_limit(self) -> Dict[str, Any]:
        return self.data["rate_limit"]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
