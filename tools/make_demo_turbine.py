"""Create a small demo turbine (Turbine_Demo) from the repo's Defects/ images.

This is a convenience script to try the pipeline without a real turbine dataset:

    .venv/Scripts/python tools/make_demo_turbine.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFECTS = ROOT / "Defects"
BASE = ROOT / "Turbine_Demo"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
BLADES = ["Blade_A", "Blade_B", "Blade_C"]


def main() -> None:
    picks = [p for p in sorted(DEFECTS.iterdir())
             if p.is_file() and p.suffix.lower() in EXTS][:9]
    if not picks:
        raise SystemExit(f"No sample images found in {DEFECTS}")
    for i, src in enumerate(picks):
        blade = BLADES[i % len(BLADES)]
        dest_dir = BASE / blade / "images"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"blade_{blade.lower()}_{i}{src.suffix}"
        shutil.copy(src, dest)
    # Per-blade metadata CSVs.
    for b in BLADES:
        imgs = sorted((BASE / b / "images").glob("*"))
        lines = ["image_name,timestamp,distance_from_root_m,surface_face,edge"]
        for n, img in enumerate(imgs):
            surface = "pressure" if n % 2 == 0 else "suction"
            edge = "leading" if n % 2 == 0 else "trailing"
            lines.append(f"{img.name},2026-10-12T14:{22+n:02d}:10Z,{10+3*n}.5,{surface},{edge}")
        (BASE / b / f"Blade_{b[-1]}_metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created demo turbine at {BASE}")
    for b in BLADES:
        n = len(list((BASE / b / "images").glob("*")))
        print(f"  {b}: {n} images")


if __name__ == "__main__":
    main()
