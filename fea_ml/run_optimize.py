"""Quick optimization runner script."""
import sys
sys.argv = [
    "optimize",
    "--config", "configs/voxel_config.yaml",
    "--checkpoint", "runs/gb200_v1/ensemble/ensemble_member_00.pt",
    "--baseline", "data/runs_real/00000",
    "--output", "runs/gb200_v1/optimization",
]

from fea_ml.scripts.optimize import main
main()
