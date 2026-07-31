#!/usr/bin/env python3
"""Render the supplementary's topology figures from the committed CSVs.

Two figures, each a grid of per-topology panels:

  results/supplementary/bai_topologies.pdf
      Total arm pulls to identify the best arm, EigenTree-BAI against the
      centralized reference, versus the number of agents.

  results/supplementary/regret_topologies.pdf
      Group cumulative regret versus total arm pulls, EigenTree-UCB against
      Central-UCB and Coop-UCB2.

The supplementary has no length limit, so each topology gets its own full
panel rather than a row in a cramped table. Error bands and bars are the
standard deviation across runs, matching the main paper's figures.

Usage:  python scripts/supplementary_figures.py
"""
import csv
import collections
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(pathlib.Path(__file__).resolve().parent / "merw.mplstyle")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

# Full text width in inches for a two-column AAAI page, so a figure spanning
# both columns has room for a grid of panels.
TEXT_WIDTH_IN = 7.0

# Topologies on which the elected tree spans the graph. Path-like graphs
# (chain, cycle, lollipop) and the star are omitted; see the supplementary.
TOPOLOGIES = [("Erd\\H{o}s--R\\'{e}nyi", "er"), ("Barab\\'{a}si--Albert", "ba"),
              ("Grid", "grid"), ("Barbell", "barbell")]

BAI_STYLES = {
    "Hillel-BAI":    ("C3", "o", "Centralized"),
    "EigenTree-BAI": ("C2", "s", "EigenTree-BAI"),
}

REGRET_STYLES = {
    "Central-UCB":  ("C3", "--", "Central-UCB"),
    "Coop-UCB2":    ("C0", ":",  "Coop-UCB2"),
    "EigenTreeUCB": ("C2", "-",  "EigenTree-UCB"),
}

N, K, T = 100, 5, 5000
BAI_K = 10


def _grid(n):
    """Rows/cols for n panels: two columns, as many rows as needed."""
    cols = 2
    return (n + cols - 1) // cols, cols


def _finish(fig, axes_flat, used, handles, labels, out, ncol=None):
    """Hide unused panels, add one shared legend, save."""
    for ax in axes_flat[used:]:
        ax.set_visible(False)
    fig.legend(handles, labels, loc="lower center",
               ncol=ncol or len(labels), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_bai():
    rows, missing = [], []
    for display, slug in TOPOLOGIES:
        path = DATA / f"best_arm_id_{slug}_K{BAI_K}.csv"
        if not path.exists():
            missing.append(display)
            continue
        by = collections.defaultdict(dict)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                by[int(r["N"])][r["algo"]] = (float(r["mean_pulls"]),
                                              float(r["std_pulls"]))
        rows.append((display, sorted(by), by))

    if not rows:
        print("  no BAI data")
        return

    nrows, ncols = _grid(len(rows))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(TEXT_WIDTH_IN, 2.3 * nrows),
                             sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    handles, labels = [], []
    for ax, (display, sizes, by) in zip(axes_flat, rows):
        for algo, (color, marker, label) in BAI_STYLES.items():
            means = np.array([by[n][algo][0] for n in sizes])
            stds = np.array([by[n][algo][1] for n in sizes])
            line = ax.errorbar(sizes, means, yerr=stds, label=label,
                               color=color, marker=marker, linewidth=1.2,
                               capsize=3, markersize=4)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(display)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Total arm pulls")
    for ax in axes_flat[len(rows) - ncols:len(rows)]:
        ax.set_xlabel("Number of agents $N$")

    _finish(fig, axes_flat, len(rows), handles, labels,
            OUT / "bai_topologies.pdf")
    if missing:
        print(f"  WARNING: no BAI data for {', '.join(missing)}")


def plot_regret():
    rows, missing = [], []
    for display, slug in TOPOLOGIES:
        path = DATA / f"regret_min_{slug}_N{N}_K{K}_T{T}.csv"
        if not path.exists():
            missing.append(display)
            continue
        by = collections.defaultdict(list)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                by[r["algo"]].append(r)
        if not all(a in by for a in REGRET_STYLES):
            missing.append(display)
            continue
        series = {}
        for algo in REGRET_STYLES:
            rs = sorted(by[algo], key=lambda r: int(r["t"]))
            series[algo] = (np.array([float(r["cum_pulls"]) for r in rs]),
                            np.array([float(r["mean_group"]) for r in rs]),
                            np.array([float(r["std_group"]) for r in rs]))
        rows.append((display, series))

    if not rows:
        print("  no regret data")
        return

    nrows, ncols = _grid(len(rows))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(TEXT_WIDTH_IN, 2.3 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    handles, labels = [], []
    for ax, (display, series) in zip(axes_flat, rows):
        # Compare on equal samples: every curve is drawn only as far as the
        # smallest budget all three algorithms reach, so the panel shows the
        # same number of pulls for each and the tree's depth cost stays in
        # time rather than appearing as a regret difference.
        budget = min(s[0][-1] for s in series.values())
        axis = np.linspace(0, budget, 400)
        for algo, (color, ls, label) in REGRET_STYLES.items():
            pulls, mean, std = series[algo]
            m = np.interp(axis, pulls, mean)
            s = np.interp(axis, pulls, std)
            line, = ax.plot(axis, m, label=label, color=color, linestyle=ls,
                            linewidth=1.3)
            ax.fill_between(axis, m - s, m + s, color=color, alpha=0.15)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(display)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Total arm pulls (all agents)")
        ax.set_ylabel("$\\sum_i R_i(t)$")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))

        # Where one baseline runs orders of magnitude above the others, a
        # linear axis collapses the two curves the panel is about into the
        # bottom pixel row. Switch that panel to a log scale so all three stay
        # readable; the decision is made from the data, not hard-coded per
        # topology.
        finals = [np.interp(budget, s[0], s[1]) for s in series.values()]
        if max(finals) > 20 * max(min(finals), 1e-9):
            ax.set_yscale("log")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        else:
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))

    _finish(fig, axes_flat, len(rows), handles, labels,
            OUT / "regret_topologies.pdf")
    if missing:
        print(f"  WARNING: no regret data for {', '.join(missing)}")


if __name__ == "__main__":
    plot_bai()
    plot_regret()
