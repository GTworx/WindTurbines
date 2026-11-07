"""Debug the local backend on a single image, printing the full traceback."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bladerunner.config import Config
from bladerunner.vision import LocalVisionModel
from bladerunner.vision import InferenceRequest

cfg = Config.load("bladerunner.local.json")
vm = LocalVisionModel(config=cfg)
print("Loading model...", flush=True)
vm.warmup()
print("Model loaded. device =", vm._vlm.device, flush=True)

img = Path(r"G:\My Drive\WindTurbineImages\Turbine_01\Blade_A\Blade_A_0001.jpg")
print("Image:", img, img.exists(), flush=True)
try:
    req = InferenceRequest(images=[img], taxonomy_classes=["erosion", "crack", "lightning_strike"],
                           default_severities={"erosion": 3, "crack": 4, "lightning_strike": 4})
    res = vm.infer(req)
    print("RESULT:", res.detections_by_image, res.token_usage, flush=True)
except Exception:
    traceback.print_exc()
