import json
from pathlib import Path
from typing import Dict

import numpy as np


def save_normalization_stats(path: str, stats: Dict[str, np.ndarray]) -> None:
    payload = {key: value.tolist() for key, value in stats.items()}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_normalization_stats(path: str) -> Dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text())
    return {key: np.array(value, dtype=np.float32) for key, value in payload.items()}
