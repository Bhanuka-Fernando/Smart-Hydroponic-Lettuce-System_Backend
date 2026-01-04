# ml-service/app/services/weight.py

import math
import re
from functools import lru_cache
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn

from app.core.paths import WEIGHT_BUNDLE


class WeightMLP(nn.Module):
    """
    Rebuild MLP from saved state_dict:
      net = Sequential(Linear, ReLU, Linear, ReLU, Linear)
    We build Linear sizes dynamically from the state_dict.
    """
    def __init__(self, linear_shapes: List[Tuple[int, int]]):
        super().__init__()
        layers: List[nn.Module] = []
        for i, (in_f, out_f) in enumerate(linear_shapes):
            layers.append(nn.Linear(in_f, out_f))
            if i != len(linear_shapes) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@lru_cache(maxsize=1)
def _bundle() -> Dict[str, Any]:
    b = torch.load(WEIGHT_BUNDLE, map_location="cpu")
    if not isinstance(b, dict):
        raise RuntimeError("weight_mlp_bundle.pt must be a dict bundle.")
    if "state_dict" not in b:
        raise RuntimeError("weight_mlp_bundle.pt missing state_dict.")
    return b


def _linear_shapes_from_state_dict(sd: Dict[str, torch.Tensor]) -> List[Tuple[int, int]]:
    # expects keys like: net.0.weight, net.2.weight, net.4.weight
    pat = re.compile(r"^net\.(\d+)\.weight$")
    hits = []
    for k, v in sd.items():
        m = pat.match(k)
        if m:
            hits.append((int(m.group(1)), v))

    if not hits:
        raise RuntimeError("state_dict does not contain net.<i>.weight keys.")

    hits.sort(key=lambda t: t[0])
    shapes: List[Tuple[int, int]] = []
    for _, w in hits:
        out_f, in_f = w.shape  # Linear weight is (out, in)
        shapes.append((in_f, out_f))
    return shapes


@lru_cache(maxsize=1)
def _model() -> nn.Module:
    b = _bundle()
    sd = b["state_dict"]

    shapes = _linear_shapes_from_state_dict(sd)
    m = WeightMLP(shapes)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def _prepare_features(A_leaf_cm2: float, D_cm: float) -> torch.Tensor:
    b = _bundle()
    expects = b.get("expects", ["lnA_leaf_cm2", "lnD_cm"])

    if A_leaf_cm2 <= 0 or D_cm <= 0:
        return torch.tensor([[float("nan"), float("nan")]], dtype=torch.float32)

    vals = []
    for name in expects:
        n = str(name).lower()

        if ("lna" in n) or (("log" in n) and ("a" in n)):
            vals.append(math.log(A_leaf_cm2))
        elif ("lnd" in n) or (("log" in n) and ("d" in n)):
            vals.append(math.log(D_cm))
        elif n.startswith("a"):
            vals.append(float(A_leaf_cm2))
        elif n.startswith("d"):
            vals.append(float(D_cm))
        else:
            raise RuntimeError(f"Unknown feature name in bundle expects: {name}")

    return torch.tensor([vals], dtype=torch.float32)


def predict_weight_g(A_leaf_cm2: float, D_cm: float) -> float:
    """
    Bundle target is lnW, so exp() -> grams.
    """
    x = _prepare_features(A_leaf_cm2, D_cm)
    if not torch.isfinite(x).all():
        return 0.0

    with torch.no_grad():
        lnw = _model()(x).reshape(-1)[0].item()

    target = str(_bundle().get("target", "lnW")).lower()
    if ("ln" in target) or ("log" in target):
        return float(math.exp(lnw))
    return float(lnw)
