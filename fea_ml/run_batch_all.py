#!/usr/bin/env python3
"""
Large-Scale Batch SASTO Optimization — Run Part-Aware optimization on ALL available samples.

Discovers all sample directories under data/runs_real_128/ and runs SASTO-PA on each.
Supports resuming: skips samples that already have optimization_summary.json.
Saves incremental progress after every sample.

Usage:
    cd fea_ml
    python run_batch_all.py [--max N] [--workers 1]

Output: runs/v3/batch_results_all/
"""
import sys, os, json, time, gc, traceback, faulthandler, argparse
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import torch
from pathlib import Path
from scipy.ndimage import (
    distance_transform_edt, label, generate_binary_structure,
    binary_dilation,
)
from scipy.signal import fftconvolve
import yaml

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
faulthandler.enable()

# ── topology structures ──────────────────────────────────────
_STRUCT26 = generate_binary_structure(3, 3)
_STRUCT6  = generate_binary_structure(3, 1)

# ── constants ────────────────────────────────────────────────
PART_EXTERIOR_WALL = 1
PART_INTERIOR_WALL = 2
PART_ROOF  = 3
PART_FLOOR = 4
NUM_PARTS  = 6

# ── optimization constraints ────────────────────────────────
VOLUME_WEIGHT      = 1.0
SURFACE_WEIGHT     = 0.01
CONSTRAINT_PENALTY = 100.0
MAX_VON_MISES      = 5.0e6
MAX_DISPLACEMENT   = 1.0
MAX_COMPLIANCE_RATIO = 1.15

# ── samples that cause unrecoverable CUDA crashes ────────────────────
SKIP_SAMPLES = {"02962"}
UNCERTAINTY_K      = 1.0
VM_IDX, DISP_IDX, COMP_IDX = 0, 1, 2

# ── optimization parameters ─────────────────────────────────
MIN_THICK_DEFAULT  = 2
MIN_THICK_INTERIOR = 1
BATCH_SIZE         = 200
MIN_BATCH          = 10
MAX_LAYERS         = 40
MAX_CONSECUTIVE_FAILS = 5
SENSITIVITY_FREQ   = 3
ENDGAME_BATCH      = 5
ENDGAME_MAX_EVALS  = 200
SWAP_MAX_ATTEMPTS  = 50


def load_ensemble(config_path, ensemble_dir, device):
    """Load config, normalization, and ensemble models."""
    config = yaml.safe_load(open(config_path))
    norm_dict = json.load(open(config_path.parent / "normalization.json"))

    from fea_ml.models.cnn3d import Surrogate3DResNet
    model_cfg = config.get("model", {})
    members = sorted(ensemble_dir.glob("ensemble_member_*.pt"))
    models = []
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
    print(f"Loaded {len(models)} ensemble members -> {device}")
    return config, norm_dict, models


def optimize_sample(sample_id, data_dir, output_dir, models, norm_dict,
                    config, device, verbose=False):
    """Run SASTO Part-Aware optimization on a single sample. Returns dict or None."""
    sample_dir = data_dir / sample_id
    if not sample_dir.exists():
        return None

    # Check required files
    for fn in ["targets.json", "occ.npz", "part.npz"]:
        if not (sample_dir / fn).exists():
            return None

    out_dir = output_dir / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── load sample ───────────────────────────────────────────
    try:
        targets_json = json.load(open(sample_dir / "targets.json"))
        fixed_occ = np.load(sample_dir / "occ.npz")["data"].astype(np.uint8)
        fixed_part = np.load(sample_dir / "part.npz")["data"].astype(np.uint8)
    except Exception as e:
        print(f"    [SKIP] {sample_id}: load error: {e}")
        return None

    D, H, W = fixed_occ.shape
    original_volume = int(fixed_occ.sum())

    if original_volume < 100:
        return None  # degenerate

    baseline_compliance = targets_json.get("compliance", None)
    if baseline_compliance is None or baseline_compliance <= 0:
        return None
    comp_limit = MAX_COMPLIANCE_RATIO * baseline_compliance

    # ── features ──────────────────────────────────────────────
    raw_features = np.array([
        25e9/1e11, 0.20, 2400.0/1000, 30e6/1e7,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)
    feat_mean = np.array(norm_dict["feature_mean"], dtype=np.float32)
    feat_std  = np.array(norm_dict["feature_std"],  dtype=np.float32)
    features  = (raw_features - feat_mean) / (feat_std + 1e-8)

    target_mean    = np.array(norm_dict["target_mean"], dtype=np.float32)
    target_std_arr = np.array(norm_dict["target_std"],  dtype=np.float32)
    log_targets    = norm_dict.get("log_transform_targets", [])

    def build_voxel_input(occ_arr, part_arr):
        channels = [occ_arr[None].astype(np.float32)]
        for p in range(NUM_PARTS):
            channels.append((part_arr == p).astype(np.float32)[None])
        return np.concatenate(channels, axis=0)

    def compute_surface_area_local(occ_arr):
        padded = np.pad(occ_arr, 1, mode='constant', constant_values=0)
        return (int(np.sum(np.abs(np.diff(padded, axis=0)))) +
                int(np.sum(np.abs(np.diff(padded, axis=1)))) +
                int(np.sum(np.abs(np.diff(padded, axis=2)))))

    def denorm(pred_mean, pred_std):
        pm = pred_mean * target_std_arr + target_mean
        ps = pred_std  * target_std_arr
        if log_targets:
            pm_raw = np.expm1(pm)
            ps_raw = ps * np.exp(pm)
            return pm_raw, ps_raw
        return pm, ps

    def ensemble_predict(voxel_np, feat_np):
        preds = []
        try:
            vt = torch.from_numpy(voxel_np[None]).float().to(device)
            ft = torch.from_numpy(feat_np[None]).float().to(device)
            for model in models:
                with torch.no_grad():
                    p = model(vt, ft).cpu().numpy()
                preds.append(p)
            del vt, ft
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                preds = []
                vt_c = torch.from_numpy(voxel_np[None]).float()
                ft_c = torch.from_numpy(feat_np[None]).float()
                for model in models:
                    model.cpu()
                    with torch.no_grad():
                        p = model(vt_c, ft_c).numpy()
                    preds.append(p)
                    model.to(device)
                del vt_c, ft_c
            else:
                raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0).squeeze(0), stacked.std(axis=0).squeeze(0)

    def is_simple_point(vol, z, y, x):
        if vol[z, y, x] == 0:
            return False
        zs, ze = max(0, z-1), min(vol.shape[0], z+2)
        ys, ye = max(0, y-1), min(vol.shape[1], y+2)
        xs, xe = max(0, x-1), min(vol.shape[2], x+2)
        nb = vol[zs:ze, ys:ye, xs:xe].copy()
        cz, cy, cx = z - zs, y - ys, x - xs
        nb[cz, cy, cx] = 0
        fg_labels, fg_count = label(nb, structure=_STRUCT6)
        if fg_count != 1:
            return False
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

    def get_min_thick(z, y, x):
        if fixed_part[z, y, x] == PART_INTERIOR_WALL:
            return MIN_THICK_INTERIOR
        return MIN_THICK_DEFAULT

    def passes_thickness_check(occ, z, y, x, dt):
        min_thick = get_min_thick(z, y, x)
        for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nz, ny, nx = z + dz, y + dy, x + dx
            if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):
                continue
            if occ[nz, ny, nx] > 0:
                continue
            bz, by, bx = z - dz, y - dy, x - dx
            if not (0 <= bz < D and 0 <= by < H and 0 <= bx < W):
                return False
            if occ[bz, by, bx] == 0:
                return False
            if dt[bz, by, bx] < min_thick:
                return False
        return True

    def classify_air(occ_vol):
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
        return interior_air

    def find_interior_surface(occ_vol, interior_air):
        dilated = binary_dilation(interior_air, structure=_STRUCT6, iterations=1)
        return (occ_vol > 0) & dilated

    def compute_sensitivity(current_occ):
        masked_part = fixed_part.copy()
        masked_part[current_occ == 0] = 0
        vi = build_voxel_input(current_occ, masked_part)
        ft = torch.from_numpy(features[None]).float().to(device)
        grad_total = np.zeros((D, H, W), dtype=np.float64)
        for i, model in enumerate(models):
            vt = torch.from_numpy(vi[None]).float().to(device)
            vt.requires_grad_(True)
            try:
                pred = model(vt, ft)
                loss = pred[0, COMP_IDX] + 0.3 * pred[0, VM_IDX]
                loss.backward()
                grad_total += vt.grad[0, 0].detach().cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
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
        return grad_total.astype(np.float32)

    def evaluate_occ(current_occ):
        volume = int(current_occ.sum())
        sa = compute_surface_area_local(current_occ)
        masked_part = fixed_part.copy()
        masked_part[current_occ == 0] = 0
        vi = build_voxel_input(current_occ, masked_part)
        try:
            pmean, pstd = ensemble_predict(vi, features)
        except RuntimeError:
            torch.cuda.empty_cache()
            gc.collect()
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

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: SENSITIVITY-GUIDED BATCH EROSION
    # ═══════════════════════════════════════════════════════════
    current_occ = fixed_occ.copy()
    total_removed = 0
    batch_num = 0
    history = []
    consecutive_fails = 0
    sensitivity = None

    for layer in range(MAX_LAYERS):
        dt = distance_transform_edt(current_occ > 0)
        int_air = classify_air(current_occ)
        if int(int_air.sum()) == 0:
            break

        cand_mask = find_interior_surface(current_occ, int_air)
        cand_coords_all = np.argwhere(cand_mask)
        if len(cand_coords_all) == 0:
            break

        approved = []
        for z, y, x in cand_coords_all:
            if passes_thickness_check(current_occ, z, y, x, dt):
                approved.append((z, y, x))
        if not approved:
            break

        if sensitivity is None or layer % SENSITIVITY_FREQ == 0:
            sensitivity = compute_sensitivity(current_occ)
        scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
        sorted_idx = np.argsort(-scores)
        approved = [approved[i] for i in sorted_idx]

        ptr = 0
        current_batch_size = BATCH_SIZE
        while ptr < len(approved) and consecutive_fails < MAX_CONSECUTIVE_FAILS:
            batch_num += 1
            simple_pts = []
            while len(simple_pts) < current_batch_size and ptr < len(approved):
                z, y, x = approved[ptr]
                ptr += 1
                if current_occ[z, y, x] == 0:
                    continue
                if is_simple_point(current_occ, z, y, x):
                    simple_pts.append((z, y, x))
                    current_occ[z, y, x] = 0
            if not simple_pts:
                continue

            result = evaluate_occ(current_occ)
            vol_red = 1.0 - result["volume"] / original_volume

            if result["constraints_ok"]:
                total_removed += len(simple_pts)
                consecutive_fails = 0
                current_batch_size = BATCH_SIZE
            else:
                for z, y, x in simple_pts:
                    current_occ[z, y, x] = 1
                consecutive_fails += 1
                current_batch_size = max(MIN_BATCH, current_batch_size // 2)

        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            break

    phase1_vol = int(current_occ.sum())

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: END-GAME
    # ═══════════════════════════════════════════════════════════
    endgame_evals = 0
    endgame_removed = 0
    consecutive_fails = 0
    sensitivity = compute_sensitivity(current_occ)

    for eg_round in range(20):
        dt = distance_transform_edt(current_occ > 0)
        int_air = classify_air(current_occ)
        if int(int_air.sum()) == 0:
            break

        cand_mask = find_interior_surface(current_occ, int_air)
        approved = [(z, y, x) for z, y, x in np.argwhere(cand_mask)
                    if passes_thickness_check(current_occ, z, y, x, dt)]
        if not approved:
            break

        scores = np.array([sensitivity[z, y, x] for z, y, x in approved])
        sorted_idx = np.argsort(-scores)
        approved = [approved[i] for i in sorted_idx]
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

            if result["constraints_ok"]:
                total_removed += len(simple_pts)
                endgame_removed += len(simple_pts)
                consecutive_fails = 0
            else:
                for z, y, x in simple_pts:
                    current_occ[z, y, x] = 1
                consecutive_fails += 1
                if consecutive_fails >= 15:
                    break

        if consecutive_fails >= 15 or endgame_evals >= ENDGAME_MAX_EVALS:
            break
        if eg_round > 0 and endgame_removed == 0:
            break

    phase2_vol = int(current_occ.sum())

    # ═══════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ═══════════════════════════════════════════════════════════
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

    kernel = _STRUCT6.astype(np.float32)
    kernel[1, 1, 1] = 0
    total_spikes = 0
    for i in range(15):
        nc = fftconvolve(current_occ.astype(np.float32), kernel, mode='same')
        nc = np.round(nc).astype(np.int32)
        shards = (current_occ > 0) & (nc < 2)
        n = int(shards.sum())
        if n == 0:
            break
        coords = np.argwhere(shards)
        nc_vals = nc[shards]
        order = np.argsort(nc_vals)
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
        if removed_pass == 0:
            break

    # ═══════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════
    total_time = time.time() - t_start
    final_vol = int(current_occ.sum())
    vol_red = 1.0 - final_vol / original_volume
    final_result = evaluate_occ(current_occ)

    part_breakdown = {}
    for pname, pid in [("exterior_wall", PART_EXTERIOR_WALL),
                       ("interior_wall", PART_INTERIOR_WALL),
                       ("roof", PART_ROOF), ("floor", PART_FLOOR)]:
        orig = int(((fixed_part == pid) & (fixed_occ > 0)).sum())
        cur  = int(((fixed_part == pid) & (current_occ > 0)).sum())
        pct  = cur / orig * 100 if orig > 0 else 100
        part_breakdown[pname] = {
            "original": orig, "optimized": cur, "retained_pct": round(pct, 1)
        }

    np.savez_compressed(out_dir / "optimized_occ.npz", data=current_occ)

    summary = {
        "sample_id": sample_id,
        "success": True,
        "method": "sasto_part_aware",
        "volume_original": original_volume,
        "volume_optimized": final_vol,
        "volume_reduction": float(vol_red),
        "volume_reduction_pct": round(vol_red * 100, 2),
        "phase1_volume": phase1_vol,
        "phase2_volume": phase2_vol,
        "total_removed": total_removed,
        "holes_filled": holes_filled,
        "spikes_removed": total_spikes,
        "constraints_satisfied": final_result["constraints_ok"],
        "vm_conservative": float(final_result["vm_conservative"]),
        "disp_conservative": float(final_result["disp_conservative"]),
        "comp_conservative": float(final_result["comp_conservative"]),
        "comp_limit": float(comp_limit),
        "vm_utilization": float(final_result["vm_conservative"] / MAX_VON_MISES),
        "comp_utilization": float(final_result["comp_conservative"] / comp_limit),
        "pred_mean": final_result["pred_mean"].tolist(),
        "pred_std": final_result["pred_std"].tolist(),
        "total_time_seconds": total_time,
        "n_batches": batch_num,
        "part_breakdown": part_breakdown,
        "baseline_targets": targets_json,
    }

    with open(out_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Free memory
    del current_occ, fixed_occ, fixed_part, sensitivity
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def compute_aggregate(results, output_dir):
    """Compute and save aggregate statistics from all results."""
    if not results:
        return {}

    vol_reds = [r["volume_reduction_pct"] for r in results]
    times = [r["total_time_seconds"] for r in results]
    vm_utils = [r["vm_utilization"] for r in results]
    comp_utils = [r["comp_utilization"] for r in results if r.get("comp_utilization") is not None]
    constraints_ok_count = sum(1 for r in results if r["constraints_satisfied"])

    # Filter meaningful reductions (>1%)
    meaningful = [r for r in results if r["volume_reduction_pct"] > 1.0]
    meaningful_reds = [r["volume_reduction_pct"] for r in meaningful]

    # Constraint-satisfying subset
    constr_ok = [r for r in results if r["constraints_satisfied"]]
    constr_ok_reds = [r["volume_reduction_pct"] for r in constr_ok]

    # Per-part retention
    ext_ret = [r["part_breakdown"]["exterior_wall"]["retained_pct"] for r in results]
    int_ret = [r["part_breakdown"]["interior_wall"]["retained_pct"] for r in results]
    roof_ret = [r["part_breakdown"]["roof"]["retained_pct"] for r in results]
    floor_ret = [r["part_breakdown"]["floor"]["retained_pct"] for r in results]

    # Same for constraint-satisfying
    ext_ret_ok = [r["part_breakdown"]["exterior_wall"]["retained_pct"] for r in constr_ok] if constr_ok else []
    int_ret_ok = [r["part_breakdown"]["interior_wall"]["retained_pct"] for r in constr_ok] if constr_ok else []
    roof_ret_ok = [r["part_breakdown"]["roof"]["retained_pct"] for r in constr_ok] if constr_ok else []
    floor_ret_ok = [r["part_breakdown"]["floor"]["retained_pct"] for r in constr_ok] if constr_ok else []

    aggregate = {
        "n_total_samples": len(results),
        "n_constraints_satisfied": constraints_ok_count,
        "n_meaningful_reduction": len(meaningful),
        "constraints_satisfied_pct": round(constraints_ok_count / len(results) * 100, 1),

        "all_samples": {
            "volume_reduction_pct": {
                "mean": round(np.mean(vol_reds), 2),
                "std": round(np.std(vol_reds), 2),
                "median": round(np.median(vol_reds), 2),
                "min": round(np.min(vol_reds), 2),
                "max": round(np.max(vol_reds), 2),
                "p25": round(np.percentile(vol_reds, 25), 2),
                "p75": round(np.percentile(vol_reds, 75), 2),
            },
            "runtime_seconds": {
                "mean": round(np.mean(times), 1),
                "std": round(np.std(times), 1),
                "median": round(np.median(times), 1),
                "min": round(np.min(times), 1),
                "max": round(np.max(times), 1),
            },
        },

        "constraints_satisfied": {
            "count": constraints_ok_count,
            "volume_reduction_pct": {
                "mean": round(np.mean(constr_ok_reds), 2) if constr_ok_reds else None,
                "std": round(np.std(constr_ok_reds), 2) if constr_ok_reds else None,
                "median": round(np.median(constr_ok_reds), 2) if constr_ok_reds else None,
                "min": round(np.min(constr_ok_reds), 2) if constr_ok_reds else None,
                "max": round(np.max(constr_ok_reds), 2) if constr_ok_reds else None,
            },
            "per_part_retention_pct": {
                "exterior_wall": {"mean": round(np.mean(ext_ret_ok), 1), "std": round(np.std(ext_ret_ok), 1)} if ext_ret_ok else None,
                "interior_wall": {"mean": round(np.mean(int_ret_ok), 1), "std": round(np.std(int_ret_ok), 1)} if int_ret_ok else None,
                "roof": {"mean": round(np.mean(roof_ret_ok), 1), "std": round(np.std(roof_ret_ok), 1)} if roof_ret_ok else None,
                "floor": {"mean": round(np.mean(floor_ret_ok), 1), "std": round(np.std(floor_ret_ok), 1)} if floor_ret_ok else None,
            },
        },

        "meaningful_reduction": {
            "count": len(meaningful),
            "volume_reduction_pct": {
                "mean": round(np.mean(meaningful_reds), 2) if meaningful_reds else None,
                "std": round(np.std(meaningful_reds), 2) if meaningful_reds else None,
                "median": round(np.median(meaningful_reds), 2) if meaningful_reds else None,
            },
        },

        "per_part_retention_all": {
            "exterior_wall": {"mean": round(np.mean(ext_ret), 1), "std": round(np.std(ext_ret), 1)},
            "interior_wall": {"mean": round(np.mean(int_ret), 1), "std": round(np.std(int_ret), 1)},
            "roof": {"mean": round(np.mean(roof_ret), 1), "std": round(np.std(roof_ret), 1)},
            "floor": {"mean": round(np.mean(floor_ret), 1), "std": round(np.std(floor_ret), 1)},
        },

        "per_sample": sorted(
            [
                {
                    "sample_id": r["sample_id"],
                    "volume_original": r["volume_original"],
                    "volume_optimized": r["volume_optimized"],
                    "volume_reduction_pct": r["volume_reduction_pct"],
                    "constraints_satisfied": r["constraints_satisfied"],
                    "runtime_s": round(r["total_time_seconds"], 1),
                    "vm_conservative": round(r["vm_conservative"], 2),
                    "comp_utilization": round(r["comp_utilization"], 4) if r.get("comp_utilization") else None,
                }
                for r in results
            ],
            key=lambda x: x["volume_original"],
        ),
    }

    with open(output_dir / "aggregate_results_all.json", "w") as f:
        json.dump(aggregate, f, indent=2, cls=NumpyEncoder)

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Large-scale SASTO batch optimization")
    parser.add_argument("--max", type=int, default=0, help="Max samples to run (0=all)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip samples with existing results")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save aggregate results every N samples")
    args = parser.parse_args()

    print("=" * 70)
    print("LARGE-SCALE BATCH SASTO OPTIMIZATION")
    print("=" * 70)

    # Paths
    runs_v3 = Path("runs/v3")
    data_dir = Path("data/runs_real_128")
    output_dir = runs_v3 / "batch_results_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all samples
    all_samples = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(all_samples)} sample directories")

    # Filter to valid samples (must have required files)
    valid_samples = []
    for sid in all_samples:
        sd = data_dir / sid
        if (sd / "targets.json").exists() and (sd / "occ.npz").exists() and (sd / "part.npz").exists():
            valid_samples.append(sid)
    print(f"Valid samples (with targets.json, occ.npz, part.npz): {len(valid_samples)}")

    # Check for already completed
    if args.resume:
        already_done = []
        todo = []
        for sid in valid_samples:
            summary_path = output_dir / sid / "optimization_summary.json"
            if summary_path.exists():
                already_done.append(sid)
            else:
                todo.append(sid)
        print(f"Already completed: {len(already_done)}")
        print(f"Remaining: {len(todo)}")
    else:
        already_done = []
        todo = valid_samples

    if args.max > 0:
        todo = todo[:args.max]
        print(f"Limiting to {args.max} samples")

    total_to_run = len(todo)
    print(f"\nWill optimize {total_to_run} samples")
    est_hours = total_to_run * 63 / 3600  # ~63s avg per sample
    print(f"Estimated time: {est_hours:.1f} hours\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load ensemble
    config_path = runs_v3 / "config.yaml"
    ensemble_dir = runs_v3 / "ensemble"
    config, norm_dict, models = load_ensemble(config_path, ensemble_dir, device)

    # Load results from already completed samples
    results = []
    for sid in already_done:
        try:
            summary = json.load(open(output_dir / sid / "optimization_summary.json"))
            results.append(summary)
        except Exception:
            pass

    print(f"Loaded {len(results)} existing results")

    # Run optimization
    failed = []
    skipped = []
    t_batch_start = time.time()

    for i, sid in enumerate(todo):
        if sid in SKIP_SAMPLES:
            print(f"\n  SKIPPING {sid} (known crash sample)")
            skipped.append(sid)
            continue

        elapsed = time.time() - t_batch_start
        rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
        completed = len(already_done) + i
        total_all = len(already_done) + total_to_run
        eta_h = ((total_to_run - i) / rate) if rate > 0 else 0

        print(f"\n[{completed+1}/{total_all}] ({i+1}/{total_to_run} this session) "
              f"Sample {sid}  |  Rate: {rate:.0f}/hr  ETA: {eta_h:.1f}h")

        # Clean GPU memory before each sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        try:
            result = optimize_sample(
                sample_id=sid,
                data_dir=data_dir,
                output_dir=output_dir,
                models=models,
                norm_dict=norm_dict,
                config=config,
                device=device,
                verbose=False,
            )
            if result:
                results.append(result)
                red_pct = result["volume_reduction_pct"]
                ok = "OK" if result["constraints_satisfied"] else "FAIL"
                rt = result["total_time_seconds"]
                print(f"    -> {red_pct:.1f}% reduction, constraints={ok}, {rt:.0f}s")
            else:
                skipped.append(sid)
                print(f"    -> SKIPPED (missing data or degenerate)")
        except torch.cuda.OutOfMemoryError:
            print(f"    -> CUDA OOM — skipping {sid}")
            torch.cuda.empty_cache()
            gc.collect()
            failed.append(sid)
        except Exception as e:
            print(f"    -> ERROR: {e}")
            traceback.print_exc()
            failed.append(sid)

        # Flush output after every sample so log file captures everything
        sys.stdout.flush()
        sys.stderr.flush()

        # Periodic aggregate save
        if (i + 1) % args.save_every == 0:
            agg = compute_aggregate(results, output_dir)
            n_ok = agg.get("n_constraints_satisfied", 0)
            mean_red = agg.get("all_samples", {}).get("volume_reduction_pct", {}).get("mean", 0)
            print(f"\n  [CHECKPOINT] {len(results)} results, "
                  f"{n_ok} constraints-OK, mean reduction={mean_red:.1f}%\n")

    # ═══════════════════════════════════════════════════════════
    # FINAL AGGREGATE
    # ═══════════════════════════════════════════════════════════
    total_seconds = time.time() - t_batch_start
    print(f"\n\n{'='*70}")
    print(f"BATCH COMPLETE — {len(results)} samples in {total_seconds/3600:.1f} hours")
    print(f"{'='*70}")

    agg = compute_aggregate(results, output_dir)

    print(f"\nTotal samples optimized: {agg['n_total_samples']}")
    print(f"Constraints satisfied: {agg['n_constraints_satisfied']} "
          f"({agg['constraints_satisfied_pct']:.1f}%)")
    print(f"Meaningful reduction (>1%): {agg['n_meaningful_reduction']}")

    a = agg["all_samples"]["volume_reduction_pct"]
    print(f"\nVolume Reduction (all):")
    print(f"  Mean:   {a['mean']:.1f}% ± {a['std']:.1f}%")
    print(f"  Median: {a['median']:.1f}%")
    print(f"  Range:  [{a['min']:.1f}%, {a['max']:.1f}%]")
    print(f"  IQR:    [{a['p25']:.1f}%, {a['p75']:.1f}%]")

    if agg["constraints_satisfied"]["count"] > 0:
        c = agg["constraints_satisfied"]["volume_reduction_pct"]
        print(f"\nVolume Reduction (constraints-OK):")
        print(f"  Mean:   {c['mean']:.1f}% ± {c['std']:.1f}%")
        print(f"  Median: {c['median']:.1f}%")

        pp = agg["constraints_satisfied"]["per_part_retention_pct"]
        print(f"\nPer-Part Retention (constraints-OK):")
        for part in ["exterior_wall", "interior_wall", "roof", "floor"]:
            if pp[part]:
                print(f"  {part:16s}: {pp[part]['mean']:.1f}% ± {pp[part]['std']:.1f}%")

    rt = agg["all_samples"]["runtime_seconds"]
    print(f"\nRuntime: {rt['mean']:.0f}s ± {rt['std']:.0f}s "
          f"(median: {rt['median']:.0f}s)")

    if failed:
        print(f"\nFailed: {len(failed)} samples")
    if skipped:
        print(f"Skipped: {len(skipped)} samples")

    print(f"\nResults saved to {output_dir}/aggregate_results_all.json")


if __name__ == "__main__":
    main()
