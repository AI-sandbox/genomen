import logging
import random
from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jaxtyping import Int32
from torch.utils.data import DataLoader
from tqdm import tqdm

from .... import utils


@dataclass
class DNNTrainerConfig:
    classification: bool

    # logging
    debug: bool = False

    # training
    epochs: int = 10
    lr: float = 5e-5
    betas: tuple[float, float] = (0.5, 0.999)
    lr_weight_decay: float = 0.0
    patience: int = 8
    batch_size: int = 128

    # val
    num_val_batches: int = 5

    def __post_init__(self):
        self.scorer = "rocauc" if self.classification else "r2"

        self.use_wandb: bool = self.debug
        self.disable_tqdm: bool = not self.debug


class DNNTrainer:
    def __init__(self, model: nn.Module, cfg: DNNTrainerConfig, device: str | t.device):
        self.cfg = cfg
        self.device = device
        self._logger = logging.getLogger(__name__)

        self.model = model.to(self.device)
        # Add weight decay to optimizer
        self.optimizer = t.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            betas=self.cfg.betas,
            weight_decay=self.cfg.lr_weight_decay,
        )

        # Initialize early stopping variables
        self.best_score = float("-inf")
        self.patience_counter = 0

    def _training_step(self, X: Int32[t.Tensor, "b seq"], y: Int32[t.Tensor, "b"]) -> float:
        self.model.train()
        logits = self.model(X)

        if self.cfg.classification:
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = F.mse_loss(logits, y)

        if self.cfg.use_wandb:
            wandb.log({"train_loss": loss.item()}, step=self.step)

        return loss

    @t.no_grad()
    def _evaluate(self, val_loader: DataLoader):
        self.model.eval()
        val_score = []
        sampled_batches = random.sample(
            list(val_loader), min(self.cfg.num_val_batches, len(val_loader))
        )

        for batch in sampled_batches:
            X_val, y_val = batch
            X_val = X_val.to(self.device)

            logits = self.model(X_val)
            val_score.append(
                utils.score(
                    y_val.numpy(),
                    logits.detach().cpu().numpy(),
                    scorer=self.cfg.scorer,
                    classification=self.cfg.classification,
                )
            )

        mean_val_score = sum(val_score) / len(sampled_batches)

        if self.cfg.use_wandb:
            wandb.log({"val_score": mean_val_score}, step=self.step)

        return mean_val_score

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None):
        self.step = 0
        if self.cfg.use_wandb:
            wandb.init(
                project="MetaPRS",
                name=str(self.model.__class__.__name__),
                config=self.cfg.__dict__,
            )

        for epoch in range(self.cfg.epochs):
            # Training phase
            self.model.train()
            progress_bar = tqdm(
                train_loader,
                total=int(len(train_loader)),
                disable=self.cfg.disable_tqdm,
            )
            num_batches = 0
            epoch_val_scores = []

            for batch, (X, y) in enumerate(progress_bar):
                X = X.to(self.device)
                y = y.to(self.device)

                loss = self._training_step(X, y)
                num_batches += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.step += X.shape[0]
                progress_bar.set_description(
                    f"{epoch=}, {batch=}, train_loss={loss.item():.4f}, examples_seen={self.step}"
                )

            if val_loader:
                epoch_val_scores.append(self._evaluate(val_loader))

            # Early stopping logic
            if self.cfg.patience > 0 and val_loader:
                if epoch_val_scores[-1] < self.best_score:
                    self.best_score = epoch_val_scores[-1]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.cfg.patience:
                        self._logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
