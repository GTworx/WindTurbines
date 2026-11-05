import sys
from pathlib import Path

# Ensure the project root is importable when tests are run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from PIL import Image


def _make_image(path: Path, color=(120, 160, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), color).save(path, format="JPEG")


@pytest.fixture
def turbine_dir(tmp_path: Path) -> Path:
    """Build a synthetic turbine: Turbine_T104/{Blade_A,Blade_B}/{images,metadata.csv}."""
    turbine = tmp_path / "Turbine_T104"
    for blade in ("Blade_A", "Blade_B"):
        images = turbine / blade / "images"
        images.mkdir(parents=True)
        for i in range(1, 4):
            _make_image(images / f"IMG_{i:04d}.jpg")
        (turbine / blade / "metadata.csv").write_text(
            "image_name,timestamp,distance_from_root_m,surface_face,edge\n"
            f"IMG_0001.jpg,2026-10-12T14:22:10Z,12.5,pressure,leading\n"
            f"IMG_0002.jpg,2026-10-12T14:23:10Z,24.0,suction,trailing\n"
            f"IMG_0003.jpg,2026-10-12T14:24:10Z,36.5,pressure,leading\n",
            encoding="utf-8",
        )
    return turbine


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A fresh project root with Defects/, Logs/ and a seed taxonomy."""
    root = tmp_path / "root"
    (root / "Defects").mkdir(parents=True)
    _make_image(root / "Defects" / "erosion-sample.png", color=(200, 120, 60))
    _make_image(root / "Defects" / "crack-sample.png", color=(80, 80, 80))
    (root / "Logs").mkdir()
    return root
