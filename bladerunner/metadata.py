"""Blade ``metadata.csv`` parsing and schema contract.

Implements the strict schema contract from the spec (Section 4.2). The canonical
columns are:

  image_name, timestamp, distance_from_root_m, surface_face,
  camera_pitch, gps_lat, gps_lon, altitude_m

The parser tolerates a handful of common header aliases and maps any
``Blade_*_technical_metadata.csv`` file to a canonical :class:`MetadataRecord`.
Unknown columns are preserved in ``MetadataRecord.extra``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .models import MetadataRecord

# canonical key -> list of accepted header aliases (lower-cased, trimmed)
ALIAS_MAP: Dict[str, List[str]] = {
    "image_name": [
        "image_name", "image", "filename", "file_name", "img_name", "file",
        "anonymized_filename", "image_filename", "photo_name",
    ],
    "timestamp": ["timestamp", "capture_time", "datetime", "captured_at", "time", "date"],
    "distance_from_root_m": [
        "distance_from_root_m",
        "span_distance_from_root",
        "distance_from_root",
        "span_m",
        "distance_m",
    ],
    "surface_face": ["surface_face", "surface", "face", "side"],
    "edge": ["edge", "edge_position", "leading_trailing_edge"],
    "camera_pitch": ["camera_pitch", "pitch", "camera_angle", "angle"],
    "gps_lat": ["gps_lat", "lat", "latitude"],
    "gps_lon": ["gps_lon", "lon", "longitude"],
    "altitude_m": ["altitude_m", "altitude", "alt", "relative_altitude_m", "dji_relative_altitude_m"],
}

# header alias -> canonical key
_HEADER_TO_CANONICAL = {
    alias: canonical for canonical, aliases in ALIAS_MAP.items() for alias in aliases
}


def find_metadata_file(blade_dir: Path) -> Optional[Path]:
    """Locate a metadata CSV in a blade directory.

    Matches ``*_metadata.csv`` or any ``*.csv``; the spec notes files such as
    ``Blade_A_technical_metadata.csv``.
    """
    if not blade_dir.exists():
        return None
    candidates = list(blade_dir.glob("*metadata*.csv")) + list(blade_dir.glob("*.csv"))
    if not candidates:
        return None
    return candidates[0]


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(" ", "")
    # Handle European-style decimal commas (e.g. "0,0", "148,58") that appear
    # in semicolon-delimited metadata exports.
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        f = float(text)
        return f if f == f else None  # reject NaN
    except (ValueError, TypeError):
        return None


def parse_metadata_row(raw: Dict[str, object]) -> MetadataRecord:
    """Normalize one CSV row into a :class:`MetadataRecord`."""
    canonical: Dict[str, object] = {}
    extra: Dict[str, object] = {}
    for key, value in raw.items():
        canon = _HEADER_TO_CANONICAL.get(str(key).strip().lstrip("\ufeff").lower())
        if canon is None:
            if value not in (None, ""):
                extra[str(key)] = value
        else:
            if value not in (None, ""):
                canonical[canon] = value

    image_name = str(canonical.get("image_name", "") or "")
    if not image_name:
        # fall back to first non-empty string value (defensive)
        for v in raw.values():
            if isinstance(v, str) and v.strip():
                image_name = v.strip()
                break
    return MetadataRecord(
        image_name=image_name,
        timestamp=str(canonical.get("timestamp", "")) or None,
        distance_from_root_m=_to_float(canonical.get("distance_from_root_m")),
        surface_face=str(canonical.get("surface_face", "")) or None,
        edge=str(canonical.get("edge", "")) or None,
        camera_pitch=_to_float(canonical.get("camera_pitch")),
        gps_lat=_to_float(canonical.get("gps_lat")),
        gps_lon=_to_float(canonical.get("gps_lon")),
        altitude_m=_to_float(canonical.get("altitude_m")),
        extra=extra,
    )


def load_metadata(blade_dir: Path) -> Dict[str, MetadataRecord]:
    """Load a blade's metadata CSV into ``{image_name: MetadataRecord}``."""
    mapping: Dict[str, MetadataRecord] = {}
    md_file = find_metadata_file(blade_dir)
    if md_file is None:
        return mapping
    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is a core dependency
        return mapping
    # Auto-detect the delimiter: real exports are often semicolon-delimited
    # (e.g. DJI) with comma decimals, while our canonical format is comma-delimited.
    try:
        df = pd.read_csv(md_file, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(md_file, sep=";")
        except Exception:
            return mapping
    for row in df.to_dict(orient="records"):
        rec = parse_metadata_row(row)
        if rec.image_name:
            mapping[rec.image_name] = rec
    return mapping


def metadata_for_image(blade_dir: Path, image_name: str) -> Optional[MetadataRecord]:
    """Look up a single image's metadata by exact or stem match."""
    mapping = load_metadata(blade_dir)
    if image_name in mapping:
        return mapping[image_name]
    stem = Path(image_name).stem
    for key, rec in mapping.items():
        if Path(key).stem == stem:
            return rec
    return None
