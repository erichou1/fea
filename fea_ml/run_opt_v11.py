"""
V11: Sensitivity-Guided Topology Optimization

Improvements over V10 (34.3% volume reduction):
  1. SENSITIVITY-GUIDED EROSION: Use ML model gradients to compute
     structural contribution of each voxel and prioritize removing
     those that contribute least.
  2. PART-AWARE THICKNESS: Interior walls allow MIN_THICK=1,
     exterior walls / roof / floor keep MIN_THICK=2.
  3. FINE-GRAINED END-GAME: After main erosion converges, drop to
     batch=5 then batch=1 to squeeze out extra material.
  4. RELAXED UNCERTAINTY: K=1.0 (was 1.5). Ensemble already captures
     model uncertainty; less conservative margin allows more removal.
  5. SWAP MOVES: After erosion, redistribute material from thick
     (structurally redundant) to thin (structurally critical) regions,
     improving efficiency and enabling further erosion.
  6. INCREASED COMPLIANCE BUDGET: 1.15x baseline (was 1.10x).
     V10 used only 71% of compliance budget, leaving headroom.

Phases:
  Phase 1: Sensitivity-guided batch erosion (main loop)
  Phase 2: End-game fine-grained erosion (batch=5..1)
  Phase 3: Swap moves + post-swap erosion
"""
import sys, os, json, time, gc, traceback, faulthandler
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import (
    distance_transform_edt, label, generate_binary_structure,
    binary_dilation,
)
from scipy.signal import fftconvolve

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
faulthandler.enable()

print("[0] V11 script start", flush=True)

# ── paths ─────────────────────────────────────────────────────────
DATA_128 = Path("data/runs_real_128")
RUNS_V3  = Path("runs/v3")
OPT_OUT  = RUNS_V3 / "optimization_128"
OPT_OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RUNS_V3 / "ensemble"

import yaml
config    = yaml.safe_load(open(RUNS_V3 / "config.yaml"))
norm_dict = json.load(open(RUNS_V3 / "normalization.json"))
print("[1] Config loaded", flush=True)

# ── device ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[1b] Device: {device}", flush=True)
if device.type == "cuda":
    print(f"     GPU : {torch.cuda.get_device_name()}", flush=True)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"     VRAM: {vram:.1f} GB", flush=True)

# ── load ensemble ─────────────────────────────────────────────────
from fea_ml.models.cnn3d import Surrogate3DResNet

model_cfg = config.get("model", {})
members   = sorted(CHECKPOINT_DIR.glob("ensemble_member_*.pt"))
models    = []
for mp in members:
    ckpt = torch.load(mp, map_location="cpu", weights_only=False)
    m = Surrogate3DResNet(
        in_channels=7, feature_dim=10,
        target_dim=len(config["targets"]),
        base_channels=model_cfg.get("base_channels", 64),
        dropout=model_cfg.get("dropout", 0.15),
        drop_path=model_cfg.get("drop_path", 0.1),
    )
    state = ckpt.get("model_state_dict", ckpt.get("ema_state_dict", ckpt))
    m.load_state_dict(state)
    m.eval()
    m.to(device)
    models.append(m)
    del ckpt, state
    gc.collect()
print(f"[2] Loaded {len(models)} ensemble members → {device}", flush=True)

# ── load fixed sample ────────────────────────────────────────────
sample_id  = "00472"
sample_dir = DATA_128 / sample_id
targets_json = json.load(open(sample_dir / "targets.json"))

fixed_occ  = np.load(OPT_OUT / "fixed_occ.npz")["data"].astype(np.uint8)
fixed_part = np.load(OPT_OUT / "fixed_part.npz")["data"].astype(np.uint8)
D, H, W    = fixed_occ.shape
original_volume = int(fixed_occ.sum())
print(f"[3] Fixed model: {original_volume:,} voxels ({D}×{H}×{W})", flush=True)

PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF  = 3
PART_FLOOR = 4
NUM_PARTS  = 6

for pname, pid in [("Exterior Wall", PART_EXTERIOR_WALL),
                   ("Interior Wall", PART_INTERIOR_WALL),
                   ("Roof", PART_ROOF), ("Floor", PART_FLOOR)]:
    cnt = int(((fixed_part == pid) & (fixed_occ > 0)).sum())
    print(f"    {pname:16s}: {cnt:6d} voxels", flush=True)

# ── features ─────────────────────────────────────────────────────
raw_features = np.array([
    25e9/1e11, 0.20, 2400.0/1000, 30e6/1e7,
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
], dtype=np.float32)
feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
feat_std  = np.array(norm_dict["feature_std"],  dtype=np.float32)
features  = (raw_features - feat_mean) / (feat_std + 1e-8)

# ── helpers ──────────────────────────────────────────────────────
def build_voxel_input(occ_arr, part_arr):
    channels = [occ_arr[None].astype(np.float32)]
    for p in range(NUM_PARTS):
        channels.append((part_arr == p).astype(np.float32)[None])
    return np.concatenate(channels, axis=0)

def compute_surface_area(occ_arr):
    padded = np.pad(occ_arr, 1, mode='constant', constant_values=0)
    return (int(np.sum(np.abs(np.diff(padded, axis=0)))) +
            int(np.sum(np.abs(np.diff(padded, axis=1)))) +
            int(np.sum(np.abs(np.diff(padded, axis=2)))))

target_mean    = np.array(norm_dict["target_mean"], dtype=np.float32)
target_std_arr = np.array(norm_dict["target_std"],  dtype=np.float32)
log_targets    = norm_dict.get("log_transform_targets", [])

def denorm(pred_mean, pred_std):
    pm = pred_mean * target_std_arr + target_mean
    ps = pred_std  * target_std_arr
    if log_targets:
        pm_raw = np.expm1(pm)
        ps_raw = ps * np.exp(pm)
        return pm_raw, ps_raw
    return pm, ps

def ensemble_predict(voxel_np, feat_np):
    vt = torch.from_numpy(voxel_np[None]).float().to(device)
    ft = torch.from_numpy(feat_np[None]).float().to(device)
    preds = []
    for model in models:
        with torch.no_grad():
            p = model(vt, ft).cpu().numpy()
        preds.append(p)
    del vt, ft
    stacked = np.stack(preds, axis=0)
    return stacked.mean(axis=0).squeeze(0), stacked.std(axis=0).squeeze(0)

# ── constraints (V11 updates) ────────────────────────────────────
VOLUME_WEIGHT      = 1.0
SURFACE_WEIGHT     = 0.01
CONSTRAINT_PENALTY = 100.0
MAX_VON_MISES      = 5.0e6
MAX_DISPLACEMENT   = 1.0
MAX_COMPLIANCE_RATIO = 1.15     # ← V11: raised from 1.10
UNCERTAINTY_K      = 1.0        # ← V11: relaxed from 1.5
VM_IDX, DISP_IDX, COMP_IDX = 0, 1, 2

baseline_compliance = targets_json.get("compliance", None)
comp_limit = MAX_COMPLIANCE_RATIO * baseline_compliance if baseline_compliance else None
print(f"[4] Constraints:", flush=True)
print(f"    VM   < {MAX_VON_MISES:.0e}", flush=True)
print(f"    Disp < {MAX_DISPLACEMENT}", flush=True)
print(f"    Comp < {comp_limit:.4f}  (baseline={baseline_compliance:.4f}, "
      f"ratio={MAX_COMPLIANCE_RATIO})", flush=True)
print(f"    K    = {UNCERTAINTY_K}", flush=True)

# ── optimisation parameters (V11 updates) ─────────────────────────
MIN_THICK_DEFAULT  = 2          # exterior walls, roof, floor
MIN_THICK_INTERIOR = 1          # interior walls: allow 1-voxel thick
BATCH_SIZE         = 200
MIN_BATCH          = 10         # ← V11: lower floor for binary search
MAX_LAYERS         = 40         # ← V11: more layers
MAX_CONSECUTIVE_FAILS = 5
SENSITIVITY_FREQ   = 3          # recompute gradients every N layers
ENDGAME_BATCH      = 5          # end-game batch size
ENDGAME_MAX_EVALS  = 200        # max ML calls in end-game
SWAP_MAX_ATTEMPTS  = 50         # max swap evaluations

# ── topology check ───────────────────────────────────────────────
_STRUCT26 = generate_binary_structure(3, 3)
_STRUCT6  = generate_binary_structure(3, 1)

def is_simple_point(vol, z, y, x):
    """6,26 digital topology check — True if removal preserves topology.
    Uses 6-connectivity (face-sharing) for foreground so the solid never
    develops diagonal-only connections that marching cubes turns into
    floating mesh pieces.  Background uses 26-connectivity per the
    standard (6-fg, 26-bg) pairing rule."""
    if vol[z, y, x] == 0:
        return False
    zs, ze = max(0, z-1), min(vol.shape[0], z+2)
    ys, ye = max(0, y-1), min(vol.shape[1], y+2)
    xs, xe = max(0, x-1), min(vol.shape[2], x+2)
    nb = vol[zs:ze, ys:ye, xs:xe].copy()
    cz, cy, cx = z - zs, y - ys, x - xs
    nb[cz, cy, cx] = 0

    # Foreground must remain ONE 6-connected component (face-sharing)
    fg_labels, fg_count = label(nb, structure=_STRUCT6)
    if fg_count != 1:
        return False

    # Background uses 26-connectivity (standard pairing with 6-fg)
    bg = (nb == 0).astype(np.uint8)
    bg_labels, _ = label(bg, structure=_STRUCT26)
    adj_labels = set()
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < nb.shape[0] and 0 <= ny < nb.shape[1] and 0 <= nx < nb.shape[2]:
                    if bg[nz, ny, nx]:
                        adj_labels.add(bg_labels[nz, ny, nx])
    adj_labels.add(bg_labels[cz, cy, cx])
    return len(adj_labels) == 1


# ── part-aware thickness check ───────────────────────────────────
def get_min_thick(z, y, x):
    """Return part-specific minimum wall thickness."""
    if fixed_part[z, y, x] == PART_INTERIOR_WALL:
        return MIN_THICK_INTERIOR
    return MIN_THICK_DEFAULT

def passes_thickness_check(occ, z, y, x, dt):
    """Check removal keeps wall ≥ part-specific minimum thickness."""
    min_thick = get_min_thick(z, y, x)
    for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        nz, ny, nx = z + dz, y + dy, x + dx
        if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):
            continue
        if occ[nz, ny, nx] > 0:
            continue  # not air-facing
        # This face is air-adjacent — check wall behind
        bz, by, bx = z - dz, y - dy, x - dx
        if not (0 <= bz < D and 0 <= by < H and 0 <= bx < W):
            return False
        if occ[bz, by, bx] == 0:
            return False
        if dt[bz, by, bx] < min_thick:
            return False
    return True


# ── air classification ───────────────────────────────────────────
def classify_air(occ_vol, verbose=True):
    """Interior air (rooms — not touching grid boundary)."""
    air = (occ_vol == 0)
    air_labels, n_air = label(air, structure=_STRUCT6)
    boundary_labels = set()
    for face in [air_labels[0,:,:], air_labels[-1,:,:],
                 air_labels[:,0,:], air_labels[:,-1,:],
                 air_labels[:,:,0], air_labels[:,:,-1]]:
        boundary_labels.update(face[face > 0].tolist())
    interior_air = np.zeros_like(air)
    for lbl in range(1, n_air + 1):
        if lbl not in boundary_labels:
            interior_air[air_labels == lbl] = True
    if verbose:
        n_ext = sum(1 for lbl in range(1, n_air+1) if lbl in boundary_labels)
        n_int = n_air - n_ext
        print(f"    Air: {n_air} comp ({n_ext} ext, {n_int} int), "
              f"interior: {int(interior_air.sum()):,}", flush=True)
    return interior_air

def find_interior_surface(occ_vol, interior_air):
    """Solid voxels 6-adjacent to interior air (erosion candidates)."""
    dilated = binary_dilation(interior_air, structure=_STRUCT6, iterations=1)
    return (occ_vol > 0) & dilated


# ── sensitivity (gradient) computation ───────────────────────────
def compute_sensitivity(current_occ):
    """
    Compute ∂(combined_loss)/∂(occupancy) for every voxel via backprop.

    combined_loss = compliance + 0.3 * von_mises  (in normalised space)

    Returns (D, H, W) float32 array.
      Positive gradient → removing voxel *decreases* structural load → safe.
      Negative gradient → removing voxel *increases* structural load → risky.

    Candidates are sorted descending so safest removals come first.
    """
    t0 = time.time()
    masked_part = fixed_part.copy()
    masked_part[current_occ == 0] = 0
    vi = build_voxel_input(current_occ, masked_part)

    ft = torch.from_numpy(features[None]).float().to(device)
    grad_total = np.zeros((D, H, W), dtype=np.float64)

    for i, model in enumerate(models):
        vt = torch.from_numpy(vi[None]).float().to(device)
        vt.requires_grad_(True)

        try:
            pred = model(vt, ft)                 # (1, num_targets)
            # Compliance is the tightest constraint; VM secondary
            loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
            loss.backward()
            grad_total += vt.grad[0, 0].detach().cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    GPU OOM on model {i}; falling back to CPU", flush=True)
                torch.cuda.empty_cache()
                m_cpu = model.cpu()
                vt_c = torch.from_numpy(vi[None]).float().requires_grad_(True)
                ft_c = torch.from_numpy(features[None]).float()
                pred = m_cpu(vt_c, ft_c)
                loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
                loss.backward()
                grad_total += vt_c.grad[0, 0].detach().numpy()
                model.to(device)
                del vt_c, ft_c, pred, loss
            else:
                raise
        finally:
            del vt
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    del ft
    grad_total /= len(models)
    elapsed = time.time() - t0
    print(f"    Sensitivity computed in {elapsed:.1f}s  "
          f"(range [{grad_total.min():.3e}, {grad_total.max():.3e}])", flush=True)
    return grad_total.astype(np.float32)


# ── evaluation ───────────────────────────────────────────────────
def evaluate_occ(current_occ):
    volume = int(current_occ.sum())
    sa     = compute_surface_area(current_occ)

    masked_part = fixed_part.copy()
    masked_part[current_occ == 0] = 0
    vi = build_voxel_input(current_occ, masked_part)
    pmean, pstd = ensemble_predict(vi, features)
    pmean_d, pstd_d = denorm(pmean, pstd)

    vm_c   = pmean_d[VM_IDX]   + UNCERTAINTY_K * pstd_d[VM_IDX]
    disp_c = pmean_d[DISP_IDX] + UNCERTAINTY_K * pstd_d[DISP_IDX]
    comp_c = pmean_d[COMP_IDX] + UNCERTAINTY_K * pstd_d[COMP_IDX]

    vm_viol   = max(0, vm_c   - MAX_VON_MISES)  / max(MAX_VON_MISES, 1e-12)
    disp_viol = max(0, disp_c - MAX_DISPLACEMENT)
    comp_viol = max(0, comp_c - comp_limit) / comp_limit if comp_limit else 0.0

    constraints_ok = (vm_viol == 0) and (disp_viol == 0) and (comp_viol == 0)
    norm_vol = volume / max(original_volume, 1)
    norm_sa  = sa     / max(original_volume, 1)
    obj = VOLUME_WEIGHT * norm_vol + SURFACE_WEIGHT * norm_sa
    if not constraints_ok:
        obj += CONSTRAINT_PENALTY * (vm_viol + disp_viol + comp_viol)

    del masked_part, vi
    gc.collect()
    return {
        "obj": obj, "volume": volume,
        "pred_mean": pmean_d, "pred_std": pstd_d,
        "constraints_ok": constraints_ok,
        "vm_conservative": vm_c, "disp_conservative": disp_c,
        "comp_conservative": comp_c,
    }

def save_checkpoint(current_occ, result, batch_num, total_removed, history,
                    phase="erosion"):
    np.savez_compressed(OPT_OUT / "optimized_occ_v11.npz", data=current_occ)
    vol_red = 1.0 - result["volume"] / original_volume
    ckpt = {
        "batch": batch_num, "phase": phase,
        "best_obj": float(result["obj"]),
        "volume_reduction": float(vol_red),
        "constraints_ok": result["constraints_ok"],
        "pred_mean": result["pred_mean"].tolist(),
        "pred_std":  result["pred_std"].tolist(),
        "total_removed": total_removed,
    }
    with open(OPT_OUT / "checkpoint_v11.json", "w") as f:
        json.dump(ckpt, f, indent=2)

def print_part_status(tag=""):
    for pname, pid in [("ExtWall", PART_EXTERIOR_WALL),
                       ("IntWall", PART_INTERIOR_WALL),
                       ("Roof", PART_ROOF), ("Floor", PART_FLOOR)]:
        orig = int(((fixed_part == pid) & (fixed_occ > 0)).sum())
        cur  = int(((fixed_part == pid) & (current_occ > 0)).sum())
        pct  = cur / orig * 100 if orig > 0 else 100
        print(f"    {pname:8s}: {orig:6d} → {cur:6d} ({pct:5.1f}%){tag}",
              flush=True)


# ══════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════
print("\n[5] Baseline evaluation on fixed model...", flush=True)
t_base = time.time()
base_result = evaluate_occ(fixed_occ)
eval_sec = time.time() - t_base
print(f"    obj={base_result['obj']:.4f}  vol={base_result['volume']:,}  "
      f"ok={base_result['constraints_ok']}", flush=True)
print(f"    VM  ={base_result['vm_conservative']:.0f}  "
      f"(limit {MAX_VON_MISES:.0f})", flush=True)
print(f"    Disp={base_result['disp_conservative']:.2e}  "
      f"(limit {MAX_DISPLACEMENT})", flush=True)
print(f"    Comp={base_result['comp_conservative']:.4f}  "
      f"(limit {comp_limit:.4f})", flush=True)
print(f"    eval time={eval_sec:.2f}s", flush=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 1: SENSITIVITY-GUIDED BATCH EROSION
# ══════════════════════════════════════════════════════════════════
current_occ     = fixed_occ.copy()
total_removed   = 0
batch_num       = 0
history         = []
consecutive_fails = 0
sensitivity     = None          # cached gradient map

print(f"\n[6] PHASE 1: Sensitivity-guided erosion  "
      f"(batch={BATCH_SIZE}, min_thick={MIN_THICK_DEFAULT})", flush=True)

phase1_t0 = time.time()

for layer in range(MAX_LAYERS):
    layer_t0 = time.time()
    print(f"\n{'─'*60}", flush=True)
    print(f"  LAYER {layer+1}/{MAX_LAYERS}", flush=True)
    print(f"{'─'*60}", flush=True)

    # Distance transform for thickness check
    dt = distance_transform_edt(current_occ > 0)

    # Interior air
    int_air = classify_air(current_occ)
    if int(int_air.sum()) == 0:
        print("  No interior air. Stopping Phase 1.", flush=True)
        break

    # Surface candidates
    cand_mask = find_interior_surface(current_occ, int_air)
    cand_coords_all = np.argwhere(cand_mask)
    n_raw = len(cand_coords_all)
    if n_raw == 0:
        print("  No surface candidates. Stopping Phase 1.", flush=True)
        break

    for pname, pid in [("ExtWall", PART_EXTERIOR_WALL),
                       ("IntWall", PART_INTERIOR_WALL),
                       ("Roof", PART_ROOF), ("Floor", PART_FLOOR)]:
        m = cand_mask & (fixed_part == pid)
        print(f"    {pname:8s}: {int(m.sum()):5d} raw candidates", flush=True)

    # ── part-aware thickness filter ───────────────────────────
    print(f"    Filtering {n_raw} candidates by thickness...", flush=True)
    t_filt = time.time()
    approved = []
    blocked_thick = 0
    for z, y, x in cand_coords_all:
        if passes_thickness_check(current_occ, z, y, x, dt):
            approved.append((z, y, x))
        else:
            blocked_thick += 1
    print(f"    Thickness: {len(approved)} approved, {blocked_thick} blocked  "
          f"({time.time()-t_filt:.1f}s)", flush=True)

    if len(approved) == 0:
        print("  All blocked by thickness. Stopping Phase 1.", flush=True)
        break

    # ── sensitivity-guided sorting ────────────────────────────
    if sensitivity is None or layer % SENSITIVITY_FREQ == 0:
        print("    Computing sensitivity map...", flush=True)
        sensitivity = compute_sensitivity(current_occ)

    scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
    sorted_idx = np.argsort(-scores)          # descending = safest first
    approved = [approved[i] for i in sorted_idx]
    print(f"    Sensitivity (approved): "
          f"safest={scores[sorted_idx[0]]:.3e}  "
          f"riskiest={scores[sorted_idx[-1]]:.3e}", flush=True)

    # ── batch processing ──────────────────────────────────────
    ptr = 0
    layer_removed = 0
    current_batch_size = BATCH_SIZE

    while ptr < len(approved) and consecutive_fails < MAX_CONSECUTIVE_FAILS:
        batch_num += 1

        # Collect simple points (sensitivity-ordered)
        simple_pts = []
        checked = 0
        while len(simple_pts) < current_batch_size and ptr < len(approved):
            z, y, x = approved[ptr]
            ptr += 1
            if current_occ[z, y, x] == 0:
                continue
            checked += 1
            if is_simple_point(current_occ, z, y, x):
                simple_pts.append((z, y, x))
                current_occ[z, y, x] = 0      # tentative removal

        if not simple_pts:
            print(f"    Batch {batch_num}: no simple points "
                  f"(checked {checked})", flush=True)
            continue

        # ML evaluation
        t1 = time.time()
        result = evaluate_occ(current_occ)
        dt_eval = time.time() - t1
        vol_red = 1.0 - result["volume"] / original_volume

        if result["constraints_ok"]:
            total_removed += len(simple_pts)
            layer_removed += len(simple_pts)
            consecutive_fails = 0
            current_batch_size = BATCH_SIZE

            entry = {
                "batch": batch_num, "layer": layer + 1, "phase": "erosion",
                "removed": len(simple_pts), "total_removed": total_removed,
                "volume": result["volume"],
                "vol_reduction": float(vol_red),
                "obj": float(result["obj"]),
                "vm":   float(result["vm_conservative"]),
                "disp": float(result["disp_conservative"]),
                "comp": float(result["comp_conservative"]),
            }
            history.append(entry)
            save_checkpoint(current_occ, result, batch_num,
                            total_removed, history, "erosion")

            print(f"    Batch {batch_num}: ACCEPT {len(simple_pts):4d} │ "
                  f"vol_red={vol_red:5.1%} │ VM={result['vm_conservative']:.0f} │ "
                  f"Comp={result['comp_conservative']:.4f} │ "
                  f"eval={dt_eval:.1f}s", flush=True)
        else:
            # Undo
            for z, y, x in simple_pts:
                current_occ[z, y, x] = 1
            consecutive_fails += 1
            current_batch_size = max(MIN_BATCH, current_batch_size // 2)

            print(f"    Batch {batch_num}: REJECT {len(simple_pts):4d} │ "
                  f"VM={result['vm_conservative']:.0f} "
                  f"Comp={result['comp_conservative']:.4f} │ "
                  f"fail {consecutive_fails}/{MAX_CONSECUTIVE_FAILS} │ "
                  f"next_batch={current_batch_size}", flush=True)

    # ── layer summary ─────────────────────────────────────────
    layer_time = time.time() - layer_t0
    cur_vol = int(current_occ.sum())
    cur_red = 1.0 - cur_vol / original_volume
    print(f"\n  Layer {layer+1}: removed {layer_removed}, "
          f"total_red={cur_red:.1%}, time={layer_time:.0f}s", flush=True)
    print_part_status()

    if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
        print(f"\n  Max consecutive fails. → Phase 2", flush=True)
        break
    if layer_removed == 0:
        print(f"  No removals this layer. → Phase 2", flush=True)
        break

phase1_vol = int(current_occ.sum())
phase1_red = 1.0 - phase1_vol / original_volume
phase1_time = time.time() - phase1_t0
print(f"\n  Phase 1 complete: {phase1_red:.1%} reduction  "
      f"({original_volume:,} → {phase1_vol:,})  [{phase1_time:.0f}s]", flush=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 2: FINE-GRAINED END-GAME
# ══════════════════════════════════════════════════════════════════
print(f"\n[7] PHASE 2: End-game (batch={ENDGAME_BATCH}, "
      f"max_evals={ENDGAME_MAX_EVALS})", flush=True)

phase2_t0 = time.time()
endgame_evals   = 0
endgame_removed = 0
consecutive_fails = 0            # reset

# Fresh sensitivity for final push
print("    Computing fresh sensitivity...", flush=True)
sensitivity = compute_sensitivity(current_occ)

for eg_round in range(20):       # up to 20 micro-rounds
    dt = distance_transform_edt(current_occ > 0)
    int_air = classify_air(current_occ, verbose=False)
    if int(int_air.sum()) == 0:
        break

    cand_mask = find_interior_surface(current_occ, int_air)
    cand_coords = np.argwhere(cand_mask)

    approved = [(z, y, x) for z, y, x in cand_coords
                if passes_thickness_check(current_occ, z, y, x, dt)]
    if not approved:
        print(f"    EG round {eg_round+1}: no candidates", flush=True)
        break

    # Sort by sensitivity
    scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
    sorted_idx = np.argsort(-scores)
    approved = [approved[i] for i in sorted_idx]

    round_removed = 0
    ptr = 0

    while ptr < len(approved) and endgame_evals < ENDGAME_MAX_EVALS:
        batch_num += 1
        simple_pts = []
        while len(simple_pts) < ENDGAME_BATCH and ptr < len(approved):
            z, y, x = approved[ptr]
            ptr += 1
            if current_occ[z, y, x] == 0:
                continue
            if is_simple_point(current_occ, z, y, x):
                simple_pts.append((z, y, x))
                current_occ[z, y, x] = 0

        if not simple_pts:
            continue

        endgame_evals += 1
        result = evaluate_occ(current_occ)
        vol_red = 1.0 - result["volume"] / original_volume

        if result["constraints_ok"]:
            total_removed += len(simple_pts)
            round_removed += len(simple_pts)
            endgame_removed += len(simple_pts)
            consecutive_fails = 0

            entry = {
                "batch": batch_num, "phase": "endgame",
                "removed": len(simple_pts), "total_removed": total_removed,
                "volume": result["volume"],
                "vol_reduction": float(vol_red),
                "obj":  float(result["obj"]),
                "vm":   float(result["vm_conservative"]),
                "comp": float(result["comp_conservative"]),
            }
            history.append(entry)
            save_checkpoint(current_occ, result, batch_num,
                            total_removed, history, "endgame")

            if endgame_evals % 10 == 0:
                print(f"    EG eval {endgame_evals}: ACCEPT {len(simple_pts)}, "
                      f"vol_red={vol_red:.1%}", flush=True)
        else:
            for z, y, x in simple_pts:
                current_occ[z, y, x] = 1
            consecutive_fails += 1
            if consecutive_fails >= 15:
                break

    print(f"    EG round {eg_round+1}: removed {round_removed}, "
          f"evals={endgame_evals}/{ENDGAME_MAX_EVALS}, "
          f"fails={consecutive_fails}", flush=True)

    if consecutive_fails >= 15 or endgame_evals >= ENDGAME_MAX_EVALS:
        break
    if round_removed == 0:
        break

phase2_vol = int(current_occ.sum())
phase2_red = 1.0 - phase2_vol / original_volume
phase2_time = time.time() - phase2_t0
print(f"\n  Phase 2 complete: +{endgame_removed} removed  "
      f"(total_red={phase2_red:.1%}, {phase2_vol:,} voxels)  "
      f"[{phase2_time:.0f}s]", flush=True)


# ══════════════════════════════════════════════════════════════════
# PHASE 3: SWAP MOVES + POST-SWAP EROSION
# ══════════════════════════════════════════════════════════════════
print(f"\n[8] PHASE 3: Swap moves (max {SWAP_MAX_ATTEMPTS} evals)", flush=True)

phase3_t0 = time.time()
swap_accepted = 0
swap_evals    = 0
post_swap_removed = 0

# Recompute sensitivity
print("    Computing fresh sensitivity...", flush=True)
sensitivity = compute_sensitivity(current_occ)

for swap_round in range(3):
    dt = distance_transform_edt(current_occ > 0)
    int_air = classify_air(current_occ, verbose=False)
    if int(int_air.sum()) == 0:
        break

    # Removal candidates: surface voxels in thick regions
    cand_mask = find_interior_surface(current_occ, int_air)
    thick_cands = [(z, y, x) for z, y, x in np.argwhere(cand_mask)
                   if dt[z, y, x] >= MIN_THICK_DEFAULT + 1]

    # Add-back candidates: removed voxels adjacent to structure
    removed_mask = (fixed_occ > 0) & (current_occ == 0)
    adj_struct   = binary_dilation(current_occ > 0, structure=_STRUCT6, iterations=1)
    addback_cands = list(np.argwhere(removed_mask & adj_struct))

    if not thick_cands or not addback_cands:
        print(f"    Round {swap_round+1}: insufficient candidates "
              f"(thick={len(thick_cands)}, addback={len(addback_cands)})", flush=True)
        break

    # Sort: remove safest first, add back most beneficial first
    thick_scored = sorted(thick_cands,
                          key=lambda c: -sensitivity[c[0], c[1], c[2]])
    add_scored = sorted(addback_cands,
                        key=lambda c: sensitivity[c[0], c[1], c[2]])

    round_swaps = 0
    for ri in range(min(len(thick_scored), SWAP_MAX_ATTEMPTS - swap_evals)):
        rz, ry, rx = thick_scored[ri]
        if current_occ[rz, ry, rx] == 0:
            continue
        if not is_simple_point(current_occ, rz, ry, rx):
            continue

        # Try pairing with top add-back candidates
        for ai in range(min(len(add_scored), 5)):
            az, ay, ax = add_scored[ai]
            if current_occ[az, ay, ax] > 0:
                continue

            # Execute swap
            current_occ[rz, ry, rx] = 0
            current_occ[az, ay, ax] = 1

            swap_evals += 1
            result = evaluate_occ(current_occ)

            if result["constraints_ok"]:
                swap_accepted += 1
                round_swaps += 1
                save_checkpoint(current_occ, result,
                                batch_num + swap_evals,
                                total_removed, history, "swap")
                if swap_accepted % 5 == 0:
                    print(f"    Swap eval {swap_evals}: ACCEPT  "
                          f"comp={result['comp_conservative']:.4f}", flush=True)
                break
            else:
                # Undo
                current_occ[rz, ry, rx] = 1
                current_occ[az, ay, ax] = 0
                break   # try next removal candidate

        if swap_evals >= SWAP_MAX_ATTEMPTS:
            break

    print(f"    Round {swap_round+1}: {round_swaps} swaps accepted  "
          f"({swap_evals} evals)", flush=True)

    if round_swaps == 0:
        break

    # Post-swap erosion attempt
    print("    Attempting post-swap erosion...", flush=True)
    sensitivity = compute_sensitivity(current_occ)

    dt = distance_transform_edt(current_occ > 0)
    int_air = classify_air(current_occ, verbose=False)
    cand_mask = find_interior_surface(current_occ, int_air)
    approved = [(z, y, x) for z, y, x in np.argwhere(cand_mask)
                if passes_thickness_check(current_occ, z, y, x, dt)]

    if approved:
        scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
        sorted_idx = np.argsort(-scores)
        approved = [approved[i] for i in sorted_idx]

        for ptr in range(0, len(approved), ENDGAME_BATCH):
            if swap_evals >= SWAP_MAX_ATTEMPTS:
                break
            batch = []
            for i in range(ptr, min(ptr + ENDGAME_BATCH, len(approved))):
                z, y, x = approved[i]
                if current_occ[z, y, x] > 0 and is_simple_point(current_occ, z, y, x):
                    batch.append((z, y, x))
                    current_occ[z, y, x] = 0
            if not batch:
                continue

            swap_evals += 1
            result = evaluate_occ(current_occ)
            if result["constraints_ok"]:
                total_removed += len(batch)
                post_swap_removed += len(batch)
                save_checkpoint(current_occ, result,
                                batch_num + swap_evals,
                                total_removed, history, "swap_erosion")
                print(f"    Post-swap erosion: ACCEPT {len(batch)}", flush=True)
            else:
                for z, y, x in batch:
                    current_occ[z, y, x] = 1
                break   # stop post-swap attempts for this round

phase3_vol = int(current_occ.sum())
phase3_red = 1.0 - phase3_vol / original_volume
phase3_time = time.time() - phase3_t0
print(f"\n  Phase 3 complete: {swap_accepted} swaps, "
      f"+{post_swap_removed} eroded  "
      f"(total_red={phase3_red:.1%}, {phase3_vol:,} voxels)  "
      f"[{phase3_time:.0f}s]", flush=True)


# ══════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════

# ── fill small enclosed air pockets ──────────────────────────────
print(f"\n[9] Filling small enclosed air pockets...", flush=True)
air_mask = (current_occ == 0)
air_labels, n_air_comp = label(air_mask, structure=_STRUCT6)
boundary_labels = set()
for face in [air_labels[0,:,:], air_labels[-1,:,:],
             air_labels[:,0,:], air_labels[:,-1,:],
             air_labels[:,:,0], air_labels[:,:,-1]]:
    boundary_labels.update(face[face > 0].tolist())

holes_filled = 0
for lbl in range(1, n_air_comp + 1):
    if lbl in boundary_labels:
        continue
    comp = air_labels == lbl
    sz = int(comp.sum())
    if sz <= 50:
        current_occ[comp] = 1
        holes_filled += sz
        print(f"    Filled hole: {sz} voxels", flush=True)
print(f"  Total holes filled: {holes_filled}", flush=True)

# ── remove spike voxels ──────────────────────────────────────────
print(f"\n[10] Removing shards...", flush=True)
kernel = _STRUCT6.astype(np.float32)
kernel[1, 1, 1] = 0
SHARD_THRESHOLD = 2   # remove voxels with < 2 face-neighbors (i.e., ≤1)
total_spikes = 0
for i in range(15):
    nc = fftconvolve(current_occ.astype(np.float32), kernel, mode='same')
    nc = np.round(nc).astype(np.int32)
    shards = (current_occ > 0) & (nc < SHARD_THRESHOLD)
    n = int(shards.sum())
    if n == 0:
        break
    coords = np.argwhere(shards)
    nc_vals = nc[shards]
    order = np.argsort(nc_vals)   # worst shards first
    coords = coords[order]
    removed_pass = 0
    for coord in coords:
        z, y, x = coord
        if current_occ[z, y, x] == 0:
            continue
        if is_simple_point(current_occ, z, y, x):
            current_occ[z, y, x] = 0
            total_spikes += 1
            removed_pass += 1
    print(f"    Pass {i+1}: {removed_pass} removed, total={total_spikes}", flush=True)
    if removed_pass == 0:
        break
print(f"  Total shards removed: {total_spikes}", flush=True)


# ══════════════════════════════════════════════════════════════════
# FINAL RESULTS
# ══════════════════════════════════════════════════════════════════
total_time = time.time() - phase1_t0

print(f"\n{'='*60}", flush=True)
print("V11 OPTIMIZATION COMPLETE", flush=True)
print(f"{'='*60}", flush=True)

final_vol = int(current_occ.sum())
vol_red   = 1.0 - final_vol / original_volume
print(f"Volume: {original_volume:,} → {final_vol:,}  ({vol_red:.1%} reduction)",
      flush=True)
print(f"  Phase 1 (erosion):  → {phase1_vol:,}  "
      f"({1-phase1_vol/original_volume:.1%} red)", flush=True)
print(f"  Phase 2 (endgame):  → {phase2_vol:,}  "
      f"(+{phase1_vol - phase2_vol} removed)", flush=True)
print(f"  Phase 3 (swaps):    → {phase3_vol:,}  "
      f"(+{phase2_vol - phase3_vol} removed)", flush=True)
print(f"  Post-proc:          → {final_vol:,}  "
      f"(+{holes_filled} filled, -{total_spikes} spikes)", flush=True)
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

print(f"\nPer-part breakdown:", flush=True)
for pname, pid in [("Exterior Wall", PART_EXTERIOR_WALL),
                   ("Interior Wall", PART_INTERIOR_WALL),
                   ("Roof", PART_ROOF), ("Floor", PART_FLOOR)]:
    orig = int(((fixed_part == pid) & (fixed_occ > 0)).sum())
    cur  = int(((fixed_part == pid) & (current_occ > 0)).sum())
    pct  = cur / orig * 100 if orig > 0 else 100
    print(f"  {pname:16s}: {orig:6d} → {cur:6d}  ({pct:5.1f}% kept)",
          flush=True)

# Final ML evaluation
print(f"\nFinal ML evaluation...", flush=True)
final_result = evaluate_occ(current_occ)
vm_util   = final_result["vm_conservative"]   / MAX_VON_MISES
comp_util = final_result["comp_conservative"] / comp_limit if comp_limit else 0
print(f"  VM   = {final_result['vm_conservative']:.0f} Pa  "
      f"(limit {MAX_VON_MISES:.0f}, {vm_util:.0%} utilised)", flush=True)
print(f"  Disp = {final_result['disp_conservative']:.2e}  "
      f"(limit {MAX_DISPLACEMENT})", flush=True)
if comp_limit:
    print(f"  Comp = {final_result['comp_conservative']:.4f}  "
          f"(limit {comp_limit:.4f}, {comp_util:.0%} utilised)", flush=True)

# Save final
np.savez_compressed(OPT_OUT / "optimized_occ_v11.npz", data=current_occ)
summary = {
    "success": True,
    "method": "v11_sensitivity_guided",
    "improvements": [
        "sensitivity_guided_erosion",
        "part_aware_thickness",
        "endgame_fine_grained",
        "relaxed_uncertainty_k1.0",
        "swap_moves",
        "compliance_budget_1.15x",
    ],
    "params": {
        "min_thick_default": MIN_THICK_DEFAULT,
        "min_thick_interior": MIN_THICK_INTERIOR,
        "uncertainty_k": UNCERTAINTY_K,
        "max_compliance_ratio": MAX_COMPLIANCE_RATIO,
        "batch_size": BATCH_SIZE,
        "endgame_batch": ENDGAME_BATCH,
    },
    "volume_original": original_volume,
    "volume_optimized": final_vol,
    "volume_reduction": float(vol_red),
    "phase1_volume": phase1_vol,
    "phase2_volume": phase2_vol,
    "phase3_volume": phase3_vol,
    "total_removed": total_removed,
    "holes_filled": holes_filled,
    "spikes_removed": total_spikes,
    "swaps_accepted": swap_accepted,
    "constraints_satisfied": final_result["constraints_ok"],
    "total_time_seconds": total_time,
    "history": history,
}
with open(OPT_OUT / "optimization_summary_v11.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved.", flush=True)


# ── STL export ───────────────────────────────────────────────────
print(f"\nExporting STLs...", flush=True)
from scipy.ndimage import gaussian_filter as gauss_filter
from skimage.measure import marching_cubes
import trimesh

def export_mesh(occ_grid, out_path, label_str="",
                blur_sigma=0.15, smooth_iter=3, smooth_lamb=0.3):
    """Sharp SDF-based mesh export."""
    data = occ_grid.astype(np.float64)
    dt_in  = distance_transform_edt(data > 0)
    dt_out = distance_transform_edt(data == 0)
    sdf = dt_in - dt_out
    if blur_sigma > 0:
        sdf = gauss_filter(sdf, sigma=blur_sigma)
    padded = np.pad(sdf, pad_width=2, mode='constant', constant_values=-2.0)
    SCALE = 10.0 / 128
    verts, faces, normals, _ = marching_cubes(padded, level=0.0)
    verts = (verts - 2.0) * SCALE

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    if smooth_iter > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iter,
                                           lamb=smooth_lamb)
    trimesh.repair.fix_normals(mesh)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.export(str(out_path))

    ext = mesh.extents
    print(f"  {label_str}:", flush=True)
    print(f"    File : {out_path}", flush=True)
    print(f"    Verts: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}",
          flush=True)
    print(f"    Size : {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} m",
          flush=True)

try:
    export_mesh(current_occ, OPT_OUT / "optimized_v11_sharp.stl",
                "V11 sharp")
    export_mesh(current_occ, OPT_OUT / "optimized_v11_medium.stl",
                "V11 medium", blur_sigma=0.3, smooth_iter=5)
except Exception as ex:
    print(f"  STL export failed: {ex}", flush=True)
    traceback.print_exc()

print("\nALL DONE", flush=True)
