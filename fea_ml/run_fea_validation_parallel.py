#!/usr/bin/env python3
"""
Parallel Full-Population FEA Validation with timeout per sample.
Uses ProcessPoolExecutor for 3x speedup over sequential.
Resumes from existing results in fea_validation_full.json + fea_validation_100.json.
"""
import sys, os, json, argparse, time, gc, random, traceback
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout

# We import the worker function at top level so it's picklable on Windows
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker_run_single(args_dict):
    """Wrapper for multiprocessing - imports and runs FEA for one design."""
    import numpy as np
    import time, gc, json, traceback
    from pathlib import Path
    
    # Import FEA functions inside worker (Windows spawn requires this)
    from run_fea_validation_100 import hex8_Ke_and_B, voxel_fea
    
    sid = args_dict['sample_id']
    group = args_dict['group']
    data_dir = args_dict['data_dir']
    batch_dir_path = args_dict['batch_dir_path']
    vol_red = args_dict['volume_reduction_pct']
    use_amg = args_dict.get('use_amg', True)
    surr_vm_cons = args_dict.get('surrogate_vm_cons', 0)
    surr_comp_cons = args_dict.get('surrogate_comp_cons', 0)
    cs_surr = args_dict.get('constraints_satisfied_surrogate', False)
    
    try:
        meta_path = Path(data_dir) / sid.zfill(5) / "meta.json"
        if not meta_path.exists():
            return {"sample_id": sid, "group": group, "error": f"no meta.json"}
        with open(meta_path) as f:
            meta = json.load(f)
        
        voxel_size = meta['voxel_size']
        mat_E = meta.get('E', 25e9)
        mat_nu = meta.get('nu', 0.2)
        mat_rho = meta.get('density', 2400.0)
        
        occ_opt = np.load(Path(batch_dir_path) / "optimized_occ.npz")['data'].astype(np.uint8)
        occ_base = np.load(Path(data_dir) / sid.zfill(5) / "occ.npz")['data'].astype(np.uint8)
        
        n_opt = int(occ_opt.sum())
        n_base = int(occ_base.sum())
        
        t0 = time.time()
        res_base = voxel_fea(occ_base, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_base = time.time() - t0
        if res_base is None:
            return {"sample_id": sid, "group": group, "error": "baseline FEA failed (too few elements)"}
        
        t0 = time.time()
        res_opt = voxel_fea(occ_opt, voxel_size, E=mat_E, nu=mat_nu, rho=mat_rho, use_amg=use_amg)
        dt_opt = time.time() - t0
        if res_opt is None:
            return {"sample_id": sid, "group": group, "error": "optimized FEA failed (too few elements)"}
        
        comp_ratio = res_opt['compliance'] / res_base['compliance'] if res_base['compliance'] > 0 else float('inf')
        vm_ratio = res_opt['max_von_mises'] / res_base['max_von_mises'] if res_base['max_von_mises'] > 0 else float('inf')
        disp_ratio = res_opt['max_displacement'] / res_base['max_displacement'] if res_base['max_displacement'] > 0 else float('inf')
        
        return {
            "sample_id": sid,
            "group": group,
            "volume_reduction_pct": vol_red,
            "n_voxels_opt": n_opt,
            "n_voxels_base": n_base,
            "surrogate_vm_cons": surr_vm_cons,
            "surrogate_comp_cons": surr_comp_cons,
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
            "constraints_satisfied_surrogate": cs_surr,
        }
    except Exception as e:
        return {"sample_id": sid, "group": group, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        gc.collect()


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
    for s in cs:
        selected.append(('feasible', s))

    # Stratified rejected
    for s in ncs:
        vm_util = s.get('vm_utilization', 0)
        comp_util = s.get('comp_utilization', 0)
        s['_max_util'] = max(vm_util, comp_util)
    ncs_sorted = sorted(ncs, key=lambda x: x['_max_util'], reverse=True)

    rng = random.Random(seed)
    near_feasible = ncs_sorted[:20]
    for s in near_feasible:
        selected.append(('rejected_near', s))

    zero_budget = [s for s in ncs if s.get('volume_reduction_pct', 0) < 1.0]
    if len(zero_budget) < 20:
        zero_budget = sorted(ncs, key=lambda x: x.get('volume_reduction_pct', 0))[:20]
    used_rej = set(s['sample_id'] for s in near_feasible)
    zero_budget = [s for s in zero_budget if s['sample_id'] not in used_rej][:20]
    for s in zero_budget:
        selected.append(('rejected_zero', s))
        used_rej.add(s['sample_id'])

    remaining_rej = [s for s in ncs if s['sample_id'] not in used_rej]
    rng.shuffle(remaining_rej)
    for s in remaining_rej[:10]:
        selected.append(('rejected_random', s))

    return selected


def main():
    parser = argparse.ArgumentParser(description="Parallel FEA validation (355 CS + 50 rejected)")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per design in seconds")
    parser.add_argument("--n-rejected", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amg", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    BATCH_DIR = Path("runs/v3/batch_results_all")
    DATA_DIR = Path("data/runs_real_128")
    RUNS_V3 = Path("runs/v3")
    OUT_FILE = RUNS_V3 / "fea_validation_full.json"

    # Load existing results
    existing = {}
    for src in [RUNS_V3 / "fea_validation_100.json", OUT_FILE]:
        if src.exists():
            with open(src) as f:
                data = json.load(f)
            for r in data:
                if 'error' not in r and 'comp_ratio' in r:
                    existing[r['sample_id']] = r

    if args.resume:
        print(f"Existing valid results: {len(existing)}")

    selected = select_all_cs_and_rejected(
        str(BATCH_DIR), str(DATA_DIR), str(RUNS_V3 / "splits.json"),
        n_rejected=args.n_rejected, seed=args.seed
    )

    # Build task list
    to_run = []
    results = list(existing.values())
    done_ids = set(existing.keys())

    for i, (group, s) in enumerate(selected):
        sid = s['sample_id']
        if sid in done_ids:
            existing[sid]['group'] = group
            continue
        to_run.append({
            'sample_id': sid,
            'group': group,
            'data_dir': str(DATA_DIR),
            'batch_dir_path': s['_dir'],
            'volume_reduction_pct': s.get('volume_reduction_pct', 0),
            'use_amg': not args.no_amg,
            'surrogate_vm_cons': s.get('vm_conservative', 0),
            'surrogate_comp_cons': s.get('comp_conservative', 0),
            'constraints_satisfied_surrogate': s.get('constraints_satisfied', False),
        })

    print(f"\nTo run: {len(to_run)} | Already done: {len(done_ids)}")
    print(f"Workers: {args.workers} | Timeout: {args.timeout}s per design")
    print(f"AMG: {'on' if not args.no_amg else 'off'}")
    sys.stdout.flush()

    if len(to_run) == 0:
        print("Nothing to run!")
        return

    t_start = time.time()
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task in to_run:
            f = executor.submit(worker_run_single, task)
            future_to_task[f] = task

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            sid = task['sample_id']
            group = task['group']

            try:
                r = future.result(timeout=args.timeout)
            except FuturesTimeout:
                r = {"sample_id": sid, "group": group, "error": "timeout"}
            except Exception as e:
                r = {"sample_id": sid, "group": group, "error": str(e)}

            completed += 1
            total_remaining = len(to_run) - completed

            if 'error' in r:
                errors += 1
                print(f"[{completed}/{len(to_run)}] {sid} ({group}) FAILED: {r['error']}")
            else:
                status = 'PASS' if r['comp_ratio_ok'] else 'FAIL'
                print(f"[{completed}/{len(to_run)}] {sid} ({group}) C_ratio={r['comp_ratio']:.4f} {status} t={r['total_time_s']:.0f}s | ~{total_remaining} left")

            results.append(r)

            # Save incrementally every result
            with open(OUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            sys.stdout.flush()

    elapsed = time.time() - t_start

    # Final summary
    valid = [r for r in results if 'error' not in r and 'comp_ratio' in r]
    cs_valid = [r for r in valid if r.get('group') == 'feasible' or r.get('group') in ('high_reduction','near_boundary','random')]
    rej_valid = [r for r in valid if r.get('group','').startswith('rejected')]

    print(f"\n{'='*70}")
    print(f"FULL VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total valid: {len(valid)} | CS: {len(cs_valid)} | Rejected: {len(rej_valid)} | Errors: {errors}")
    print(f"New run elapsed: {elapsed/3600:.2f}h")

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
                print(f"    {grp} (n={len(sub)}): would_pass={n_pass}/{len(sub)}, C_mean={scr.mean():.4f}, C_max={scr.max():.4f}")


if __name__ == "__main__":
    main()
