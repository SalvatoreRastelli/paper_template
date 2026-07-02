"""
Entry point for figure generation.

Runs every MERW experiment/visualization script as a subprocess, with the
same arguments as MERW's own (currently disabled) GitHub Actions recipe.

Each underlying script supports --mode {compute,plot,all}:
  compute - run the (possibly expensive) simulation/eigenvector computation
            and write the raw results to paper/data/*.csv. No figure is drawn.
  plot    - read paper/data/*.csv and render the PDF into paper/results/.
            Cheap; this is what CI runs.
  all     - compute then plot in one go (useful for local one-shot runs).

This script mirrors that split:
    uv run python scripts/generate_figures.py --mode compute   # local, slow
    uv run python scripts/generate_figures.py --mode plot      # CI, fast
    uv run python scripts/generate_figures.py --mode all       # local, one-shot

The compute step should be run locally and its paper/data/*.csv output
committed to the repository; CI then only re-renders figures from that
committed data, so it never re-runs the Monte Carlo experiments.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

# graph_type -> N, matching MERW's build.yml
GRAPH_N = {
    "ba": 20,
    "er": 20,
    "barbell": 20,
    "clustered": 24,
    "grid": 25,
    "lollipop": 20,
    "chain": 10,
    "star": 20,
}

# Extra tree visualizations beyond build.yml's set, reproducing additional
# configs found in the original MERW results/ (not cited by either paper,
# but kept here since they were already part of the project's output).
EXTRA_GRAPH_N = [
    ("cycle", 8),
    ("cycle", 10),
    ("ba", 100),
]

REGRET_GRAPHS = ["ba", "er", "barbell", "lollipop", "chain", "star"]

# Extra Regret configs beyond the standard N=20/25 sweep.
EXTRA_REGRET = [
    ("ba", 100),
    ("er", 100),
]

# Extra BAI K values beyond the standard K=10.
EXTRA_BAI_K = [5, 20]


def run(args):
    print(f"$ {' '.join(args)}")
    subprocess.run(args, cwd=SCRIPTS_DIR, check=True)


def generate(mode):
    # MERW tree visualizations (one PDF per graph type)
    for graph, n in GRAPH_N.items():
        run([PYTHON, "visualize_merw.py", "--graph", graph, "--N", str(n),
             "--seed", "0", "--mode", mode])
    for graph, n in EXTRA_GRAPH_N:
        run([PYTHON, "visualize_merw.py", "--graph", graph, "--N", str(n),
             "--seed", "0", "--mode", mode])

    # AAAI paper's Figure 1: ER, N=20, p=0.5.
    run([PYTHON, "visualize_merw.py", "--graph", "er", "--N", "20", "--p-er", "0.5",
         "--seed", "0", "--mode", mode])

    # Relay cycle diagram
    run([PYTHON, "visualize_cycle.py", "--mode", mode])

    # Regret experiments (N=20, K=5, T=5000), grid uses N=25
    for graph in REGRET_GRAPHS:
        run([PYTHON, "regret_min.py", "--graph", graph, "--N", "20",
             "--K", "5", "--T", "5000", "--n-runs", "50", "--mode", mode])
    run([PYTHON, "regret_min.py", "--graph", "grid", "--N", "25",
         "--K", "5", "--T", "5000", "--n-runs", "50", "--mode", mode])
    for graph, n in EXTRA_REGRET:
        run([PYTHON, "regret_min.py", "--graph", graph, "--N", str(n),
             "--K", "5", "--T", "5000", "--n-runs", "50", "--mode", mode])

    # Best-arm-identification experiment
    run([PYTHON, "best_arm_id.py", "--graph", "ba", "--K", "10",
         "--bai-runs", "50", "--delta", "0.05", "--mode", mode])
    for k in EXTRA_BAI_K:
        run([PYTHON, "best_arm_id.py", "--graph", "ba", "--K", str(k),
             "--bai-runs", "50", "--delta", "0.05", "--mode", mode])
    run([PYTHON, "best_arm_id.py", "--graph", "er", "--K", "10",
         "--bai-runs", "50", "--delta", "0.05", "--mode", mode])

    # Fault-tolerance experiment: hub killed at t=500,1000,1500, seed chosen
    # to avoid a promotion edge case on this specific ER graph draw.
    run([PYTHON, "fault_tolerance.py", "--graph", "er", "--N", "20", "--p", "0.5",
         "--K", "10", "--T", "2000", "--fail-at", "500", "1000", "1500",
         "--seed", "2", "--mode", mode])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("compute", "plot", "all"), default="all",
                   help="compute: run experiments, save paper/data/*.csv only; "
                        "plot: render figures from existing paper/data/*.csv; "
                        "all: compute then plot (default)")
    args = p.parse_args()
    generate(args.mode)


if __name__ == "__main__":
    main()
