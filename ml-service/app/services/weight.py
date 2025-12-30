import math
import torch
import torch.nn as nn
from app.core.paths import WEIGHT_BUNDLE

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

_bundle = torch.load(WEIGHT_BUNDLE, map_location="cpu")
_model = MLP(in_dim=_bundle["in_dim"], hidden=_bundle["hidden"])
_model.load_state_dict(_bundle["state_dict"])
_model.eval()

@torch.no_grad()
def predict_weight_g(A_leaf_cm2: float, D_cm: float) -> float:
    A_leaf_cm2 = max(A_leaf_cm2, 1e-6)
    D_cm = max(D_cm, 1e-6)
    x = torch.tensor([[math.log(A_leaf_cm2), math.log(D_cm)]], dtype=torch.float32)
    lnW = float(_model(x).item())
    return float(math.exp(lnW))
