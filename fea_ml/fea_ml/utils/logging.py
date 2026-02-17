from pathlib import Path
from typing import Dict

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    def __init__(self, path: str, header: Dict[str, float] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if header:
            df = pd.DataFrame([header])
            df.to_csv(self.path, index=False)

    def log(self, row: Dict[str, float]) -> None:
        df = pd.DataFrame([row])
        if self.path.exists():
            df.to_csv(self.path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.path, index=False)


class TensorboardLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()
