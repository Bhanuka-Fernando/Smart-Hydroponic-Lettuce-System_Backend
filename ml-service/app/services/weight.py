# ml-service/app/services/weight.py

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from app.core.paths import WEIGHT_BUNDLE


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handle DataParallel: "module.net.0.weight" -> "net.0.weight"
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _infer_hidden_from_state_dict(sd: Dict[str, torch.Tensor], in_dim: int) -> List[int]:
    """
    Infer hidden layer sizes from keys like:
      net.0.weight: (H1, in_dim)
      net.2.weight: (H2, H1)
      net.4.weight: (1, H2)
    => hidden = [H1, H2]
    """
    # collect linear layer indices from net.<idx>.weight
    idxs = []
    for k, v in sd.items():
        if not (k.startswith("net.") and k.endswith(".weight")):
            continue
        parts = k.split(".")  # ["net", "<idx>", "weight"]
        if len(parts) >= 3 and parts[1].isdigit():
            idxs.append(int(parts[1]))

    if not idxs:
        raise RuntimeError("Cannot infer architecture: no 'net.<idx>.weight' keys found in state_dict")

    idxs = sorted(idxs)

    hidden: List[int] = []
    # read weight shapes in order
    for i, idx in enumerate(idxs):
        w = sd[f"net.{idx}.weight"]
        out_f, in_f = int(w.shape[0]), int(w.shape[1])

        # sanity: first layer should take in_dim
        if i == 0 and in_f != in_dim:
            raise RuntimeError(f"in_dim mismatch: bundle says {in_dim}, but first layer expects {in_f}")

        # last layer outputs 1 -> do not include as hidden
        if out_f != 1:
            hidden.append(out_f)

    return hidden


@lru_cache(maxsize=1)
def _bundle() -> Dict[str, Any]:
    b = torch.load(WEIGHT_BUNDLE, map_location="cpu")
    if not isinstance(b, dict):
        raise RuntimeError("weight_mlp_bundle.pt must be a dict")
    return b


@lru_cache(maxsize=1)
def _model() -> nn.Module:
    b = _bundle()

    in_dim = int(b["in_dim"])
    sd = b.get("state_dict")
    if not isinstance(sd, dict):
        raise RuntimeError("Bundle missing state_dict")

    sd = _strip_module_prefix(sd)

    # Infer hidden from checkpoint (most robust)
    hidden = _infer_hidden_from_state_dict(sd, in_dim=in_dim)

    m = MLPRegressor(in_dim=in_dim, hidden=hidden)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def predict_weight_g(A_leaf_cm2: float, D_cm: float) -> float:
    """
    Bundle expects: lnA_leaf_cm2, lnD_cm
    Bundle target : lnW
    Return grams : exp(lnW)
    """
    A = max(float(A_leaf_cm2), 1e-6)
    D = max(float(D_cm), 1e-6)

    x = np.array([[np.log(A), np.log(D)]], dtype=np.float32)
    x_t = torch.from_numpy(x)

    with torch.no_grad():
        lnW = _model()(x_t).cpu().numpy().reshape(-1)[0]

    return float(np.exp(lnW))
