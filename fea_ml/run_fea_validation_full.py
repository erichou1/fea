#!/usr/bin/env python3
"""
Full-Population FEA Validation:
  - All 355 constraint-satisfying designs (extending the 100 already done)
  - 50 rejected designs for false-negative audit

Reuses voxel_fea infrastructure from run_fea_validation_100.py.
"""
import sys, os, json, argparse, time, gc, random, traceback
import numpy as np
from pathlib import Path

# Import hex8 FEA functions from existing script
sys.path.insert(0, os.path.dirname(__file__))
from run_fea_validation_100 import hex8_Ke_and_B, voxel_fea


def select_all_cs_and_rejected(batch_dir, data_dir, splits_path, n_rejected=50, seed=42):
    """Select ALL constraint-satisfying designs + stratified rejected designs."""
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = set(p.split('/')[-1] for p in splits['test'])

    samples = []
    for d in sorted(os.listdir(batch_dir)):
        summary_p = os.path.join(batch_dir, d, 'optimization_summary.json')
        occ_p = os.path.join(batch_dir, d, 'optimized_occ.npz')
        if not os.path.isfile(summary_p) or not os.path.isfile(occ_p):
            continue
        with open(summary_p) as f:
            s = json.load(f)
        sid = s['sample_id']
        if sid not in test_ids:
            continue
        if not s.get('success', True):
            continue
        s['_dir'] = os.path.join(batch_dir, d)
        samples.append(s)

    cs = [s for s in samples if s.get('constraints_satisfied', False)]
    ncs = [s for s in samples if not s.get('constraints_satisfied', False)]

    print(f"Total test samples: {len(samples)}")
    print(f"Constraint-satisfying: {len(cs)}")
    print(f"Non-CS (rejected): {len(ncs)}")

    selected = []

    # All CS designs
    for s in cs:
        selected.append(('feasible', s))

    # Stratified rejected designs
    # Group 1: near-feasible (highest constraint utilization among rejected)
    for s in ncs:
        vm_util = s.get('vm_utilization', 0)
        comp_util = s.get('comp_utilization', 0)
        s['_max_util'] = max(vm_util, comp_util)
    ncs_sorted = sorted(ncs, key=lambda x: x['_max_util'], reverse=True)

    rng = random.Random(seed)

    # 20 near-feasible (highest utilization among rejected)
    near_feasible = ncs_sorted[:20]
    for s in near_feasible:
        selected.append(('rejected_near', s))

    # 20 zero-budget (these were rejected at baseline - lowest reduction)
    zero_budget = [s for s in ncs if s.get('volume_reduction_pct', 0) < 1.0]
    if len(zero_budget) < 20:
        zero_budget = sorted(ncs, key=lambda x: x.get('volume_reduction_pct', 0))[:20]
    used_rej = set(s['sample_id'] for s in near_feasible)
    zero_budget = [s for s in zero_budget if s['sample_id'] not in used_rej][:20]
    for s in zero_budget:
        selected.append(('rejected_zero', s))
        used_rej.add(s['sample_id'])

    # 10 random rejected
    remaining_rej = [s for s in ncs if s['sample_id'] not in used_rej]
    rng.shuffle(remaining_rej)
    for s in remaining_rej[:10]:
        selected.append(('rejected_random', s))

    groups = {}
    for g, _ in selected:
        groups[g] = groups.get(g, 0) + 1
    print(f"\nSelected {len(selected)} samples:")
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c}")

    return selected


def run_single(idx, total, group, sample_info, data_dir, use_amg=True):
    """Run paired baseline+optimized FEA for one design."""
    sid = sample_info['sample_id']
    try:
        meta_path = Path(data_dir) / sid.zfill(5) / "meta.json"
        if not meta_path.exists():
            return {"sample_id": sid, "error": f"no meta.json at {meta_path}"}
        with open(meta_path) as f:
            meta = json.load(f)

        voxel_size = meta['voxel_size']
        mat_E = meta.get('E', 25e9)
        mat_nu = meta.get('nu', 0.2)
        mat_rho = meta.get('density', 2400.0)

        occ_opt = np.load(Path(sample_info['_dir']) / "optimized_occ.npz")['data'].astype(np.uint8)
        occ_base = np.load(Path(data_dir) / sid.zfill(5) / "occ.npz")['data'].astype(np.uint8)

        n_opt = int(occ_opt.sum())
        n_base = int(occ_base.sum())

        t0 = time.time()
        res_base = voxel_fea(occ_base, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_base = time.time() - t0
        if res_base is None:
            return {"sample_id": sid, "error": "baseline FEA failed"}

        t0 = time.time()
        res_opt = voxel_fea(occ_opt, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_opt = time.time() - t0
        if res_opt is None:
            return {"sample_id": sid, "error": "optimized FEA failed"}

        comp_ratio = res_opt['compliance'] / res_base['compliance'] if res_base['compliance'] > 0 else float('inf')
        vm_ratio = res_opt['max_von_mises'] / res_base['max_von_mises'] if res_base['max_von_mises'] > 0 else float('inf')
        disp_ratio = res_opt['max_displacement'] / res_base['max_displacement'] if res_base['max_displacement'] > 0 else float('inf')

        return {
            "sample_id": sid,
            "group": group,
            "volume_reduction_pct": sample_info.get('volume_reduction_pct', 0),
            "n_voxels_opt": n_opt,
            "n_voxels_base": n_base,
            "surrogate_vm_cons": sample_info.get('vm_conservative', 0),
            "surrogate_comp_cons": sample_info.get('comp_conservative', 0),
            "voxel_base_comp": res_base['compliance'],
            "voxel_base_vm": res_base['max_von_mises'],
            "voxel_opt_comp": res_opt['compliance'],
            "voxel_opt_vm": res_opt['max_von_mises'],
            "voxel_opt_disp": res_opt['max_displacement'],
            "comp_ratio": comp_ratio,
            "vm_ratio": vm_ratio,
            "disp_ratio": disp_ratio,
            "comp_ratio_ok": comp_ratio <= 1.15,
            "vm_ratio_ok": vm_ratio <= 2.0,
            "total_time_s": dt_base + dt_opt,
            "amg_used": res_opt.get('amg_used', False),
            "constraints_satisfied_surrogate": sample_info.get('constraints_satisfied', False),
        }
    except Exception as e:
        return {"sample_id": sid, "error": str(e), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description="Full-population FEA validation (355 CS + 50 rejected)")
    parser.add_argument("--n-rejected", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amg", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    BATCH_DIR = Path("runs/v3/batch_results_all")
    DATA_DIR = Path("data/runs_real_128")
    RUNS_V3 = Path("runs/v3")
    OUT_FILE = RUNS_V3 / "fea_validation_full.json"
    LOG_FILE = RUNS_V3 / "fea_val_full_out.txt"

    # Redirect stdout to both console and file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    log_f = open(LOG_FILE, 'a', encoding='utf-8', errors='replace')
    sys.stdout = Tee(sys.__stdout__, log_f)

    # Load existing results (from both 100-design and full runs)
    existing = {}
    for src in [RUNS_V3 / "fea_validation_100.json", OUT_FILE]:
        if src.exists():
            with open(src) as f:
                data = json.load(f)
            for r in data:
                if 'error' not in r and 'comp_ratio' in r:
                    existing[r['sample_id']] = r
    print(f"Existing valid results: {len(existing)}")

    selected = select_all_cs_and_rejected(
        str(BATCH_DIR), str(DATA_DIR), str(RUNS_V3 / "splits.json"),
        n_rejected=args.n_rejected, seed=args.seed
    )

    # Filter already done
    to_run = []
    results = list(existing.values())
    for i, (group, s) in enumerate(selected):
        sid = s['sample_id']
        if sid in existing:
            # Update group label if needed
            existing[sid]['group'] = group
            continue
        to_run.append((i+1, len(selected), group, s))

    print(f"\nTo run: {len(to_run)} (skipping {len(existing)} already done)")
    use_amg = not args.no_amg
    print(f"AMG: {'on' if use_amg else 'off'}")
    sys.stdout.flush()

    t_start = time.time()

    for idx, total, group, s in to_run:
        sid = s['sample_id']
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] Sample {sid} ({group}, red={s.get('volume_reduction_pct',0):.1f}%)")
        sys.stdout.flush()

        r = run_single(idx, total, group, s, str(DATA_DIR), use_amg)

        if 'error' in r:
            print(f"  FAILED: {r['error']}")
        else:
            print(f"  Baseline:  {r['n_voxels_base']:,} voxels, C={r['voxel_base_comp']:.4g} J")
            print(f"  Optimized: {r['n_voxels_opt']:,} voxels, C={r['voxel_opt_comp']:.4g} J")
            print(f"  C_opt/C_base = {r['comp_ratio']:.4f} ({'PASS' if r['comp_ratio_ok'] else 'FAIL'} <= 1.15)")
            print(f"  Time: {r['total_time_s']:.1f}s (AMG={'Y' if r.get('amg_used') else 'N'})")

        results.append(r)
        with open(OUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        sys.stdout.flush()
        gc.collect()

    elapsed = time.time() - t_start

    with open(OUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results if 'error' not in r and 'comp_ratio' in r]
    cs_valid = [r for r in valid if r.get('group') == 'feasible' or r.get('group') in ('high_reduction','near_boundary','random')]
    rej_valid = [r for r in valid if r.get('group','').startswith('rejected')]

    print(f"\n{'='*70}")
    print(f"FULL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total valid: {len(valid)} | CS: {len(cs_valid)} | Rejected: {len(rej_valid)}")
    print(f"New elapsed: {elapsed/3600:.2f}h")

    if cs_valid:
        cr = np.array([r['comp_ratio'] for r in cs_valid])
        print(f"\n--- FEASIBLE SET (n={len(cs_valid)}) ---")
        print(f"  C_ratio: mean={cr.mean():.4f} std={cr.std():.4f} max={cr.max():.4f}")
        print(f"  Survival (<=1.15): {(cr<=1.15).sum()}/{len(cr)} ({100*(cr<=1.15).mean():.1f}%)")
        print(f"  > 0.9: {(cr>0.9).sum()} | > 1.0: {(cr>1.0).sum()} | > 1.15: {(cr>1.15).sum()}")

    if rej_valid:
        cr_rej = np.array([r['comp_ratio'] for r in rej_valid])
        print(f"\n--- FALSE NEGATIVE AUDIT (n={len(rej_valid)}) ---")
        print(f"  C_ratio: mean={cr_rej.mean():.4f} std={cr_rej.std():.4f} max={cr_rej.max():.4f}")
        n_would_pass = int((cr_rej <= 1.15).sum())
        print(f"  Would pass FEA: {n_would_pass}/{len(rej_valid)} ({100*n_would_pass/len(rej_valid):.1f}%)")
        for grp in ['rejected_near', 'rejected_zero', 'rejected_random']:
            sub = [r for r in rej_valid if r['group'] == grp]
            if sub:
                scr = np.array([r['comp_ratio'] for r in sub])
                n_pass = int((scr <= 1.15).sum())
                print(f"  {grp} (n={len(sub)}): would_pass={n_pass}/{len(sub)}, mean_C={scr.mean():.4f}")

    log_f.close()


if __name__ == "__main__":
    main()
