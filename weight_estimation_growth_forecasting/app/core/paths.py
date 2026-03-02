# ml-service/app/core/paths.py
from pathlib import Path

# .../ml-service/app/core/paths.py -> parents[2] == .../ml-service
SERVICE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = SERVICE_DIR / "artifacts"

DEEPLAB_CKPT = ARTIFACTS_DIR / "deeplabv3_resnet50_binseg_ckpt.pth"
LEAF_AREA_JSON = ARTIFACTS_DIR / "leaf_area_log_linear_params.json"
WEIGHT_BUNDLE = ARTIFACTS_DIR / "weight_mlp_bundle.pt"
GROWTH_BUNDLE = ARTIFACTS_DIR / "growth_model2_bundle.pkl"
