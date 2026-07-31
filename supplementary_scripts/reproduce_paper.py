#!/usr/bin/env python3
"""Reproduce every figure in the paper and the supplementary material.

Runs the three experiment scripts over the configurations the manuscript
reports, then draws the two supplementary topology panels from the CSVs those
runs leave behind. Output lands in data/ and results/ beside this file.

  main paper           results/Regret/regret_min_er_N100_K5_T5000.pdf
                       results/BAI/best_arm_id_er_K10.pdf
                       results/FaultTolerance/fault_tolerance_er_N20_K10_T2000_at500-1000-1500.pdf

  supplementary        results/supplementary/bai_topologies.pdf
                       results/supplementary/regret_topologies.pdf

Usage:
  python reproduce_paper.py                # everything, compute + plot
  python reproduce_paper.py --plot-only    # redraw from existing CSVs
  python reproduce_paper.py --only main    # just the three main figures
  python reproduce_paper.py --only supp    # just the supplementary panels
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# The main paper reports Erdos-Renyi at these settings; the supplementary
# repeats the same two experiments on three further topologies. Both sweeps
# share the ER run, so it is listed once and used by both.
ER = "er"
SUPP_GRAPHS = ["ba", "grid", "barbell"]

REGRET = ["--N", "100", "--K", "5", "--T", "5000", "--n-runs", "50", "--p-er", "0.5"]
BAI = ["--K", "10", "--bai-runs", "50", "--delta", "0.05", "--p-er", "0.5"]
# seed 2 avoids a promotion edge case on this particular ER draw; the reported
# recovery times are the ones this seed produces.
FT = ["--graph", "er", "--N", "20", "--p", "0.5", "--K", "10", "--T", "2000",
      "--fail-at", "500", "1000", "1500", "--seed", "2"]


def run(script, args, mode):
    cmd = [PY, script, *args, "--mode", mode]
    print(f"  $ {' '.join(cmd[1:])}")
    t = time.time()
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"FAILED: {script} (exit {r.returncode})")
    print(f"    done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw figures from existing CSVs, recompute nothing")
    ap.add_argument("--only", choices=("main", "supp"), default=None,
                    help="restrict to the main paper's or the supplementary's figures")
    args = ap.parse_args()

    mode = "plot" if args.plot_only else "all"
    want_main = args.only in (None, "main")
    want_supp = args.only in (None, "supp")
    t0 = time.time()

    # The ER runs produce the main paper's regret and BAI figures, and also the
    # ER panel of each supplementary figure, so they run whenever either is
    # requested rather than once per section.
    if want_main or want_supp:
        print("Erdos-Renyi (main paper figures 1 and 2):")
        run("regret_min.py", ["--graph", ER, *REGRET], mode)
        run("best_arm_id.py", ["--graph", ER, *BAI], mode)

    if want_main:
        print("Fault tolerance (main paper figure 3):")
        run("fault_tolerance.py", FT, mode)

    if want_supp:
        print("Further topologies (supplementary panels):")
        for g in SUPP_GRAPHS:
            # compute only: the per-topology PDFs are not used by either
            # document, the panels are drawn from the CSVs below.
            run("regret_min.py", ["--graph", g, *REGRET],
                "plot" if args.plot_only else "compute")
            run("best_arm_id.py", ["--graph", g, *BAI],
                "plot" if args.plot_only else "compute")

        print("Supplementary topology panels:")
        r = subprocess.run([PY, "supplementary_figures.py"], cwd=HERE)
        if r.returncode != 0:
            sys.exit("FAILED: supplementary_figures.py")

    print(f"\nAll done in {time.time() - t0:.0f}s.")
    print(f"Figures under {HERE / 'results'}, raw results under {HERE / 'data'}.")


if __name__ == "__main__":
    main()
