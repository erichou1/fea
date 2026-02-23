"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML file."""
    import yaml
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
