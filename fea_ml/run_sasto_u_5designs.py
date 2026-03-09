"""
Run SASTO-U (MIN_THICK_INTERIOR=2) on the 5 selected house designs and
save optimized_occ.npz + optimization_summary.json into
  runs/v3/batch_results_all/{sample_id}/sasto_u/

Usage:
    cd fea_ml
    python run_sasto_u_5designs.py
"""
import sys, json, gc, pathlib, shutil
import numpy as np
import torch

# Must be run from inside fea_ml/
BASE = pathlib.Path(".").resolve()

# ── patch SASTO-U setting BEFORE the global is used ──────────────────
import run_batch_all as rba   # import from cwd (fea_ml/)
rba.MIN_THICK_INTERIOR = 2    # SASTO-U: uniform min_thick=2
DATA_128   = BASE / "data" / "runs_real_128"
RUNS_V3    = BASE / "runs" / "v3"
BATCH_ALL  = RUNS_V3 / "batch_results_all"
ENSEMBLE   = RUNS_V3 / "ensemble"
CONFIG     = RUNS_V3 / "config.yaml"
TEMP_OUT   = RUNS_V3 / "sasto_u_temp"

# ── 5 target designs (from generate_house_gallery.py selection) ──────
TARGET_IDS = ["00739", "08018", "05728", "01440", "02787"]

# ── load models once ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
config, norm_dict, models = rba.load_ensemble(CONFIG, ENSEMBLE, device)

# ── run SASTO-U for each design ───────────────────────────────────────
for sid in TARGET_IDS:
    final_dir = BATCH_ALL / sid / "sasto_u"
    if (final_dir / "optimization_summary.json").exists():
        print(f"[SKIP] {sid} — already done")
        continue

    print(f"\n{'='*60}")
    print(f"Running SASTO-U on {sid} ...")

    # Use a temp output dir so we don't touch the existing PA results
    temp_sid_dir = TEMP_OUT / sid
    if temp_sid_dir.exists():
        shutil.rmtree(temp_sid_dir)

    result = rba.optimize_sample(
        sid,
        DATA_128,
        TEMP_OUT,          # optimize_sample saves to TEMP_OUT / sid /
        models, norm_dict, config, device,
        verbose=True,
    )

    if result is None:
        print(f"  [FAILED] {sid}")
        continue

    # Move results into the sasto_u/ subfolder
    final_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["optimized_occ.npz", "optimization_summary.json"]:
        src = temp_sid_dir / fname
        if src.exists():
            shutil.move(str(src), str(final_dir / fname))

    red = result.get("volume_reduction_pct", 0)
    print(f"  DONE {sid}: {red:.1f}% reduction (SASTO-U)")

# Clean up temp dir
if TEMP_OUT.exists():
    shutil.rmtree(TEMP_OUT, ignore_errors=True)

print("\nAll 5 SASTO-U optimizations complete.")
print("Results in: runs/v3/batch_results_all/{id}/sasto_u/")
