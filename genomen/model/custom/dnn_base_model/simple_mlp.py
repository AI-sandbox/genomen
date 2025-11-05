from dataclasses import dataclass

import torch as t
import torch.nn as nn
from jaxtyping import Float, Int32

from .components import PredictionHead


@dataclass
class SimpleMLPConfig:
    seq: int

    # pred
    num_pred_layers: int = 4
    norm: bool = False

    def __post_init__(self):
        self.pred_in_features = self.seq


class SimpleMLP(nn.Module):
    def __init__(self, cfg: SimpleMLPConfig):
        super().__init__()
        self.cfg = cfg

        self.model = PredictionHead(cfg)

    def forward(self, x: Int32[t.Tensor, "batch seq"]) -> Float[t.Tensor, "batch"]:
        return self.model(x)
