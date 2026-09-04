"""Amendment 10: is the sigma AUC inversion a property of sigma or of the bound?

The deployed bound U = mu + kappa*sigma + q contains sigma, so a large-sigma
state gets a wider bound and is mechanically less likely to be uncovered.
Ranking bound-failures by sigma therefore cannot distinguish 'sigma is
anti-predictive' from 'sigma widens its own bound'.

Two controls:
  A. Does sigma rank the raw residual y - mu, with the bound not involved?
  B. Under a sigma-free bound U = mu + c calibrated to the SAME marginal miss
     rate, does sigma still look inverted?
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
import os
_D1 = os.environ.get("SASTO_D1_ROOT")
_SUFFIX = "-d1" if _D1 else ""
GB200 = Path(_D1) if _D1 else Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
REPO = Path(__file__).resolve().parents[1]
TN = ("compliance", "max_von_mises", "max_displacement")
KEY = {"compliance": "compliance_j", "max_von_mises": "max_gauss_von_mises_pa",
       "max_displacement": "max_displacement_m"}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    lab = label[order]
    ranks = np.arange(1, len(lab) + 1)
    npos = int(lab.sum())
    nneg = len(lab) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    k6 = json.loads((CONTROL / ("k6-coverage-gb200-2096" + _SUFFIX + ".json")).read_text())
    kappa, q_base = k6["kappa"], k6["q_base"]
    norm = json.loads((REPO / "artifacts/g2/ensemble-v1/normalization-stats.json").read_text())

    def truth(solver, name):
        raw = solver.get(KEY[name], solver.get("max_von_mises_pa"))
        return (math.log(float(raw)) - norm["means"][name]) / norm["scales"][name]

    out = {}
    for band in ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]:
        rows = []
        for path in sorted(GB200.glob("trajectory-development-*.json")):
            record = json.loads(path.read_text())
            rows += [s for s in record["selected_states"] if s["bin_label"] == band]
        if not rows:
            continue
        mu = np.array([[s["prediction"]["mu"][n] for n in TN] for s in rows])
        sig = np.array([[s["prediction"]["sigma"][n] for n in TN] for s in rows])
        y = np.array([[truth(s["solver"], n) for n in TN] for s in rows])
        kap = np.array([kappa[n] for n in TN])
        qb = np.array([q_base[n] for n in TN])

        miss = ~np.all(y <= mu + kap * sig + qb, axis=1)
        resid = y[:, 0] - mu[:, 0]

        # Control B: sigma-free bound matched to the same marginal miss rate
        c = float(np.quantile(resid, 1.0 - miss.mean())) if 0 < miss.mean() < 1 else float("nan")
        miss_free = resid > c

        big_resid = resid > np.median(resid)
        out[band] = {
            "n": len(rows),
            "miss_rate_deployed": float(miss.mean()),
            "miss_rate_sigma_free": float(miss_free.mean()),
            "auc_sigma_vs_deployed": auc(sig[:, 0], miss),
            "auc_mu_vs_deployed": auc(mu[:, 0], miss),
            "auc_sigma_vs_sigma_free": auc(sig[:, 0], miss_free),
            "auc_mu_vs_sigma_free": auc(mu[:, 0], miss_free),
            "auc_sigma_vs_large_residual": auc(sig[:, 0], big_resid),
            "corr_sigma_abs_residual": float(np.corrcoef(sig[:, 0], np.abs(resid))[0, 1]),
            "median_slack_low_sigma": float(np.median(
                (mu[:, 0] + kap[0] * sig[:, 0] + qb[0] - y[:, 0])[sig[:, 0] <= np.median(sig[:, 0])])),
            "median_slack_high_sigma": float(np.median(
                (mu[:, 0] + kap[0] * sig[:, 0] + qb[0] - y[:, 0])[sig[:, 0] > np.median(sig[:, 0])])),
        }
        r = out[band]
        print(f"{band:10s} n={r['n']:5d} | sigma AUC deployed {r['auc_sigma_vs_deployed']:.3f} "
              f"-> sigma-free {r['auc_sigma_vs_sigma_free']:.3f} | "
              f"corr(sigma,|resid|) {r['corr_sigma_abs_residual']:+.3f}")

    dest = CONTROL / ("k6-amendment-10-sigma-artifact" + _SUFFIX + ".json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
