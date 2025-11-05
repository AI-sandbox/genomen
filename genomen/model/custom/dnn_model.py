import logging
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch as t
from torch.utils.data import DataLoader

from ..configs import ModelConfig
from .dnn_utils import DNNTrainer, DNNTrainerConfig, PRSDataSet, utils


class DNNModel:
    def __init__(
        self,
        cfg: ModelConfig,
        model_name: Literal["simple_mlp"] = "simple_mlp",
        device: t.device | str | None = None,
    ):
        self.cfg = cfg
        self._logger = logging.getLogger(__name__)

        if device is None:
            device = t.device("cuda" if t.cuda.is_available() else "cpu")
        else:
            device = t.device(device)
        self.device = device

        # init model
        if model_name == "simple_mlp":
            from .dnn_base_model import SimpleMLP, SimpleMLPConfig

            model = SimpleMLP(SimpleMLPConfig(seq=cfg.max_features))

        self.model = model

    def save_state_dict(self, path: str | Path):
        path = Path(path)
        t.save(self.model.state_dict(), path)

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray | None = None,
        y_val: npt.NDArray | None = None,
    ):
        trainer_cfg = DNNTrainerConfig(classification=self.cfg.classification)

        train_data = PRSDataSet(X_train, y_train)
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=trainer_cfg.batch_size,
        )
        if X_val is not None and y_val is not None:
            val_data = PRSDataSet(X_val, y_val)
            val_loader = DataLoader(
                val_data,
                shuffle=False,
                batch_size=trainer_cfg.batch_size,
            )
        else:
            val_loader = None
        trainer = DNNTrainer(self.model, trainer_cfg, self.device)

        trainer.fit(train_loader, val_loader)
        trainer.fit(train_loader, val_loader)

        utils.cleanup_gpu(self.model)

    def predict(self, X) -> npt.NDArray:
        self.model.to(self.device)
        self.model.eval()

        data = PRSDataSet(X)
        loader = DataLoader(data, batch_size=64, shuffle=False)

        preds = []
        with t.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                batch_preds = self.model(X_batch)
                preds.append(batch_preds.cpu())

        preds = t.cat(preds, dim=0).numpy()

        utils.cleanup_gpu(self.model)

        return preds

    def predict_proba(self, X) -> npt.NDArray:
        self.model.to(self.device)
        self.model.eval()

        data = PRSDataSet(X)
        loader = DataLoader(data, batch_size=64, shuffle=False)

        preds = []
        with t.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                batch_preds = self.model(X_batch).sigmoid()
                preds.append(batch_preds.cpu())

        preds = t.cat(preds, dim=0).numpy()

        utils.cleanup_gpu(self.model)

        neg_probs = 1 - preds
        return np.column_stack([neg_probs, preds])
