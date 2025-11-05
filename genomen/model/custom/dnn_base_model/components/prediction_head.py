import torch as t
import torch.nn as nn
from jaxtyping import Float


class PredictionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        proj_layers = []
        if self.cfg.norm:
            proj_layers.append(nn.BatchNorm1d(self.cfg.pred_in_features))
        in_features = self.cfg.pred_in_features
        for _ in range(self.cfg.num_pred_layers - 1):
            out_features = max(in_features // 2, 2)
            proj_layers.append(nn.Linear(in_features, out_features))
            proj_layers.append(nn.ReLU())
            in_features = out_features

        self.model = nn.Sequential(*proj_layers, nn.Linear(in_features, 1))

    def forward(self, x: Float[t.Tensor, "batch d_model"]) -> Float[t.Tensor, "batch"]:
        return self.model(x).squeeze()
