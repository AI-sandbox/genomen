from typing import Tuple

import numpy.typing as npt
import torch as t
from jaxtyping import Float
from torch.utils.data import Dataset


class PRSDataSet(Dataset):
    def __init__(self, X: npt.NDArray, y: npt.NDArray | None = None) -> None:
        self.X = X
        self.y = y
        self.prediction_mode = y is None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Float[t.Tensor, "seq"], Float[t.Tensor, ""] | None]:
        if self.prediction_mode:
            return t.tensor(self.X[idx], dtype=t.float32)
        else:
            return t.tensor(self.X[idx], dtype=t.float32), t.tensor(self.y[idx], dtype=t.float32)
