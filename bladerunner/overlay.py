"""Image coordinate overlay generation (spec Section 4.3).

Produces annotated copies of blade images with bounding boxes drawn, so the
dashboard / wiki can reference an immediately-verifiable image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Detection

# Category 1..5 -> color
_SEVERITY_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (200, 200, 200),  # grey
    2: (255, 200, 0),  # amber
    3: (255, 140, 0),  # orange
    4: (255, 60, 60),  # red
    5: (180, 0, 180),  # magenta
}


def _color(severity: int) -> Tuple[int, int, int]:
    return _SEVERITY_COLORS.get(int(max(1, min(5, severity))), (255, 60, 60))


def annotate_image(
    image_path: Path,
    detections: List[Detection],
    output_path: Optional[Path] = None,
    line_width: int = 3,
    draw_labels: bool = True,
) -> Optional[Path]:
    """Draw ``detections`` onto a copy of ``image_path`` and save it.

    Returns the output path, or None if the image could not be opened.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:  # pragma: no cover
        return None

    image_path = Path(image_path)
    if not image_path.exists():
        return None
    if output_path is None:
        output_path = image_path.with_suffix(".annotated" + image_path.suffix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    draw = ImageDraw.Draw(im)
    for d in detections:
        try:
            x, y, w, h = (int(v) for v in d.bbox)
        except Exception:
            continue
        box = (x, y, x + w, y + h)
        color = _color(d.severity)
        draw.rectangle(box, outline=color, width=line_width)
        if draw_labels:
            label = f"{d.defect_class} S{d.severity} {d.confidence:.2f}"
            ty = max(0, y - 14)
            draw.text((x, ty), label, fill=color)
    try:
        im.save(output_path)
    except Exception:
        return None
    return output_path


def annotate_blade(
    image_path: Path,
    detections: List[Detection],
    annotated_dir: Path,
) -> Optional[str]:
    """Annotate and save under ``annotated_dir``, returning a relative path."""
    annotated_dir = Path(annotated_dir)
    out = annotated_dir / (image_path.stem + ".annotated" + image_path.suffix)
    result = annotate_image(image_path, detections, out)
    if result is None:
        return None
    try:
        return result.relative_to(annotated_dir).as_posix()
    except ValueError:
        return str(result)
