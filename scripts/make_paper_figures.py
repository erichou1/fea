"""Generate the two paper figures from the frozen K6 coverage record.

Reads only the adjudicated JSON and the trajectory records; computes nothing new.
Figure 1 is the coverage-versus-depth curve specified in the paper skeleton before
any data existed.  Figure 2 is the mechanism panel: error and sigma on one axis,
showing that the predictor's uncertainty does not track its error.

Style follows the user's standing figure rules: uniform geometry, equal label
sizes, centered headers, matched padding, readable non-overlapping text.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sasto.g3_trajectory_calibration import _normalized_targets, _selected_trajectory_rows, _verified_json  # noqa: E402
from sasto.k6_coverage import wilson_lower_bound  # noqa: E402

PAPER = Path("/Users/eric/workspace/sasto-modernization-control/v2/g4/paper")
FIGS = PAPER / "figures"
CONTROL = Path("/Users/eric/workspace/sasto-modernization-control/v2/g3")
GB200 = Path("/Users/eric/workspace/sasto-g3-gb200-inbound/trajectory-calibration-gb200")
FROZEN = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g3/trajectory-calibration-v2")
NORM = Path("/Users/eric/workspace/fea-sasto-v/artifacts/g2/ensemble-v1/normalization-stats.json")

BINS = ["(5,10%]", "(10,15%]", "(15,20%]", "(20,25%]", ">25%"]
XTICKS = ["5–10", "10–15", "15–20", "20–25", ">25"]
TARGETS = ("compliance", "max_displacement", "max_von_mises")
LABELS = {"compliance": "Compliance", "max_displacement": "Displacement", "max_von_mises": "Von Mises"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.dpi": 200,
})

INK = "#1a1a1a"
ACCENT = "#b02418"
MUTED = "#5a7fa6"


def wilson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    z = 1.959963984540054
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c + r) / d)


def wilson_lower_two_sided(x: int, n: int) -> float:
    z = 1.959963984540054
    p = x / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d)


def figure_one(report: dict) -> None:
    per_bin = {row["bin_label"]: row for row in report["adjudication"]["per_bin"]}
    coverage = [per_bin[b]["coverage"] for b in BINS]
    n = [per_bin[b]["n"] for b in BINS]
    covered = [per_bin[b]["covered"] for b in BINS]
    lower = [wilson_lower_two_sided(c, m) for c, m in zip(covered, n)]
    upper = [wilson_upper(c, m) for c, m in zip(covered, n)]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    x = np.arange(len(BINS))

    ax.axhline(0.95, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    ax.text(len(BINS) - 0.52, 0.957, "target $1-\\alpha = 0.95$", ha="right",
            va="bottom", fontsize=7.5, color=INK)

    yerr = np.array([np.array(coverage) - np.array(lower), np.array(upper) - np.array(coverage)])
    ax.errorbar(x, coverage, yerr=yerr, fmt="o-", color=ACCENT, ecolor=ACCENT,
                elinewidth=1.1, capsize=3, markersize=5, linewidth=1.6, zorder=3,
                label="baseline-calibrated $U_j$")

    for xi, (cov, m) in enumerate(zip(coverage, n)):
        # Shallow values clear the line above; the deep value clears its error bar
        # to the left, where the axis is empty.
        if cov > 0.9:
            ax.annotate(f"{cov:.3f}", (xi, cov), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7.5, color=ACCENT)
        else:
            ax.annotate(f"{cov:.3f}", (xi, cov), textcoords="offset points",
                        xytext=(-13, 6), ha="right", fontsize=7.5, color=ACCENT)
        ax.annotate(f"$n={m}$", (xi, 0.596), ha="center", fontsize=7, color="#666666")

    ax.axvspan(2.5, 3.5, color=MUTED, alpha=0.10, zorder=0)
    ax.annotate("crossover", xy=(3.0, 0.80), ha="center", fontsize=7.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel("Trajectory depth: material removed (percent)")
    ax.set_ylabel("Joint coverage over $J=3$ targets")
    ax.set_ylim(0.575, 1.03)
    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGS / "coverage-vs-depth.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote coverage-vs-depth.pdf")


def figure_two() -> None:
    normalization = json.loads(NORM.read_text())
    kappa_rec = _verified_json(FROZEN / "kappa-development-evidence.json", "G3 kappa evidence", "kappa_evidence_sha256")
    kappa = {k: float(v) for k, v in kappa_rec["kappa"].items()}
    cases = [_verified_json(p, "G3 trajectory case", "trajectory_digest")
             for p in sorted(GB200.glob("trajectory-development-*.json"))]
    rows, _, _ = _selected_trajectory_rows(cases)

    err = defaultdict(lambda: defaultdict(list))
    sig = defaultdict(lambda: defaultdict(list))
    for row in rows:
        solver = row["solver"]
        y = _normalized_targets({
            "compliance": solver["compliance_j"],
            "max_von_mises": solver.get("max_gauss_von_mises_pa", solver.get("max_von_mises_pa")),
            "max_displacement": solver["max_displacement_m"],
        }, normalization)
        label = row["bin_label"]
        for name in TARGETS:
            mu = float(row["prediction"]["mu"][name])
            err[label][name].append(y[name] - mu)
            sig[label][name].append(float(row["prediction"]["sigma"][name]))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55))
    x = np.arange(len(BINS))

    ax = axes[0]
    for name, marker in zip(TARGETS, ("o", "s", "^")):
        mean_err = [float(np.mean(err[b][name])) for b in BINS]
        ax.plot(x, mean_err, marker=marker, markersize=4, linewidth=1.4, label=LABELS[name])
    ax.axhline(0.0, color=INK, linewidth=0.8, linestyle="--")
    ax.annotate("bias crosses zero", xy=(1.5, 0.30), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel("Material removed (percent)")
    ax.set_ylabel("Mean residual $y - \\mu$")
    ax.set_title("Bias inverts with depth")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)

    ax = axes[1]
    for name, marker in zip(TARGETS, ("o", "s", "^")):
        base_err = float(np.mean(np.abs(err[BINS[0]][name])))
        base_sig = float(np.mean(sig[BINS[0]][name]))
        rel_err = [float(np.mean(np.abs(err[b][name]))) / base_err for b in BINS]
        rel_sig = [float(np.mean(sig[b][name])) / base_sig for b in BINS]
        line, = ax.plot(x, rel_err, marker=marker, markersize=4, linewidth=1.4, label=LABELS[name])
        ax.plot(x, rel_sig, marker=marker, markersize=4, linewidth=1.2, linestyle=":",
                color=line.get_color(), alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel("Material removed (percent)")
    ax.set_ylabel("Growth relative to 5-10 percent bin")
    ax.set_title("Error outruns uncertainty")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.plot([], [], color=INK, linewidth=1.4, label="$|y-\\mu|$")
    ax.plot([], [], color=INK, linewidth=1.2, linestyle=":", label="$\\sigma$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-2:], labels[-2:], frameon=False, loc="upper left", handlelength=1.6)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    fig.savefig(FIGS / "mechanism.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote mechanism.pdf")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    report = json.loads((CONTROL / "k6-coverage-gb200-2096.json").read_text())
    figure_one(report)
    figure_two()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
