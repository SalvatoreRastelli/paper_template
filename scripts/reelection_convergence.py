"""
Re-election convergence time: warm start vs. cold start, as a function of N.

After a hub failure, the survivors re-run the decentralized gossip power
iteration on the surviving subgraph G \\ {h*} to recompute eigenvector
centrality. This experiment measures how many iterations that takes, comparing
two initializations:

  cold:  w_i = 1 for every survivor (the from-scratch initialization used at
         the very start of the protocol);
  warm:  w_i = psi_i, each survivor's own pre-failure centrality (the
         warm start of the Fault Tolerance section).

Both runs use the exact same decentralized gossip power iteration
(gossip_power_iteration in fault_tolerance.py): no node ever performs a global
reduce; each round every node gossip-averages its local log-growth rate with
random neighbors and rescales locally. The only difference between the two is
the starting vector. Convergence is declared when the per-node relative change
falls below `tol`, and the reported quantity is the number of rounds to reach
it.

For each N we average over many ER graphs; on each graph we remove the hub
(the highest-centrality node, the largest perturbation and the worst case for
re-election). The output is mean rounds-to-converge vs. N, cold and warm, with
standard-error bands. Trials are independent and run in parallel over a process
pool.

Usage:
    python reelection_convergence.py --mode all
    python reelection_convergence.py --N 20 40 60 80 100 120 140 160 180 200 \\
        --graphs-per-N 30 --p 0.5 --seed 0 --workers 8 --mode all
"""

import argparse
import csv
import os
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parent / "merw.mplstyle")
import numpy as np
import networkx as nx
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")

# Reuse the exact graph generator, initial centrality, hub selection, and the
# shared decentralized gossip power iteration from the fault-tolerance script,
# so this experiment measures the same primitive the protocol actually runs.
from fault_tolerance import (
    make_er_graph,
    gossip_power_iteration,
    build_routing_tree,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "paper" / "results"
OUT_DIR     = RESULTS_DIR / "FaultTolerance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Graph families
# ============================================================
#
# We compare two Erdos-Renyi regimes that isolate the paper's claim that
# expansion, not edge count, is what makes re-election cheap. Both have
# heterogeneous (Poisson) degrees, so both have a genuine highest-centrality
# hub and a genuine warm start; the only thing that changes is density:
#
#   er:        G(N, p) with constant p (dense). Mean degree ~pN grows with N.
#   sparse_er: G(N, c/(N-1)) with constant mean degree c (sparse). Edge count
#              is O(N) instead of O(N^2), yet the graph stays a connected
#              expander with a spectral gap bounded away from 0 as N grows.
#
# A random d-regular graph would also be a sparse expander, but its degrees
# are all equal, so its eigenvector centrality is uniform: there is no real
# hub and the warm start equals the cold start. That makes it a degenerate
# test for this experiment, so we use sparse ER instead, which keeps degree
# heterogeneity while dropping the edge count.

SPARSE_ER_MEANDEG = 10  # constant mean degree for the sparse-ER family


def make_sparse_er_graph(N, c=SPARSE_ER_MEANDEG, seed=None):
    """Connected ER graph with constant mean degree c, i.e. p = c/(N-1), so
    the edge count is O(N). Retries seeds until connected, like make_er_graph."""
    return make_er_graph(N, p=c / (N - 1), seed=seed)


def make_family_graph(family, N, p, seed):
    if family == "er":
        return make_er_graph(N, p=p, seed=seed)
    if family == "sparse_er":
        return make_sparse_er_graph(N, c=SPARSE_ER_MEANDEG, seed=seed)
    raise ValueError(f"Unknown graph family: {family}")


FAMILY_LABEL = {
    "er":        r"ER ($p={p}$, dense)",
    "sparse_er": rf"ER (mean degree ${SPARSE_ER_MEANDEG}$, sparse)",
}


# ============================================================
# One trial: remove the hub, time cold vs. warm re-election
# ============================================================

def _survivor_subgraph(G, dead_node):
    """G restricted to all nodes except dead_node, relabeled 0..N-2 in sorted
    order so the returned node k corresponds to the k-th surviving node."""
    alive = [i for i in range(G.number_of_nodes()) if i != dead_node]
    G_sub = G.subgraph(alive).copy()
    G_sub = nx.convert_node_labels_to_integers(G_sub, ordering="sorted")
    return G_sub, alive


def _run_trial(trial, p, tau_max, tol, gossip_rounds, tau_init):
    """
    One independent trial, safe to run in a worker process. `trial` is
    (family, N, graph_idx, graph_seed, run_seed). Generates a connected graph
    of the given family, computes its initial centrality and hub, removes the
    hub, and times the decentralized re-election from a cold and a warm start
    on the surviving subgraph. Returns
    (family, N, graph_idx, rounds_cold, rounds_warm).
    """
    family, N, graph_idx, graph_seed, run_seed = trial
    G = make_family_graph(family, N, p, graph_seed)

    # Initial (pre-failure) centrality and hub. This is the warm-start vector
    # the survivors will resume from, so unlike the protocol's own fixed-round
    # tau_init, we run it to full convergence here (tol-based) to get the true
    # psi rather than an approximation -- otherwise the "warm start" would be
    # warm-started from noise. Seed numpy's global RNG per trial so the gossip
    # iteration's random neighbor choices are reproducible regardless of which
    # worker or in what order this trial runs.
    np.random.seed(run_seed)
    psi, _ = gossip_power_iteration(G, np.ones(N), tau=tau_init, tol=tol,
                                    gossip_rounds=gossip_rounds)
    hub, _, _, _, _ = build_routing_tree(G, psi)

    G_sub, alive = _survivor_subgraph(G, hub)
    n_sub = len(alive)

    # Cold: every survivor starts at 1.
    _, rounds_cold = gossip_power_iteration(
        G_sub, np.ones(n_sub), tau=tau_max, tol=tol, gossip_rounds=gossip_rounds)

    # Warm: every survivor starts from its own pre-failure centrality.
    _, rounds_warm = gossip_power_iteration(
        G_sub, psi[alive].copy(), tau=tau_max, tol=tol, gossip_rounds=gossip_rounds)

    return (family, N, graph_idx, rounds_cold, rounds_warm)


# ============================================================
# Sweep over families and N, averaging over graphs (parallel over trials)
# ============================================================

def compute_convergence_data(families, N_values, graphs_per_N, p, seed, workers,
                             tau_max=500, tol=1e-8, gossip_rounds=100,
                             tau_init=500):
    """
    Build the full list of independent trials (one per (family, N, graph)),
    assign each a deterministic seed derived from the master seed, and dispatch
    them across a process pool. Returns per-(family, N) means and standard
    errors for cold and warm.
    """
    rng = np.random.RandomState(seed)
    trials = []
    for family in families:
        for N in N_values:
            for g in range(graphs_per_N):
                graph_seed = int(rng.randint(0, 2**31))
                run_seed = int(rng.randint(0, 2**31))
                trials.append((family, N, g, graph_seed, run_seed))

    worker = partial(_run_trial, p=p, tau_max=tau_max, tol=tol,
                     gossip_rounds=gossip_rounds, tau_init=tau_init)

    if workers == 1:
        rows = [worker(t) for t in trials]
    else:
        with Pool(processes=workers) as pool:
            rows = pool.map(worker, trials)

    # Aggregate per (family, N).
    summary = {}
    for family in families:
        summary[family] = {}
        for N in N_values:
            cold = np.array([r[3] for r in rows if r[0] == family and r[1] == N], dtype=float)
            warm = np.array([r[4] for r in rows if r[0] == family and r[1] == N], dtype=float)
            summary[family][N] = {
                "cold_mean": cold.mean(), "cold_sem": cold.std(ddof=1) / np.sqrt(len(cold)),
                "warm_mean": warm.mean(), "warm_sem": warm.std(ddof=1) / np.sqrt(len(warm)),
            }

    return {"rows": rows, "summary": summary, "families": list(families),
            "N_values": list(N_values), "graphs_per_N": graphs_per_N, "p": p}


# ============================================================
# Save / load / plot
# ============================================================

def _csv_path(families, N_values, graphs_per_N, p):
    fam = "-".join(families)
    tag = f"{fam}_N{N_values[0]}-{N_values[-1]}_g{graphs_per_N}_p{p}"
    return DATA_DIR / f"reelection_convergence_{tag}.csv"


def save_csv(data):
    out = _csv_path(data["families"], data["N_values"], data["graphs_per_N"], data["p"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "N", "cold_mean", "cold_sem", "warm_mean", "warm_sem"])
        for family in data["families"]:
            for N in data["N_values"]:
                s = data["summary"][family][N]
                w.writerow([family, N, s["cold_mean"], s["cold_sem"],
                            s["warm_mean"], s["warm_sem"]])

    raw = out.with_name(out.stem + "_raw.csv")
    with open(raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "N", "graph_idx", "rounds_cold", "rounds_warm"])
        for row in data["rows"]:
            w.writerow(row)

    print(f"  Saved {out} and {raw}")


def load_csv(families, N_values, graphs_per_N, p):
    out = _csv_path(families, N_values, graphs_per_N, p)
    summary = {f: {} for f in families}
    with open(out, newline="") as f:
        for r in csv.DictReader(f):
            summary[r["family"]][int(r["N"])] = {
                "cold_mean": float(r["cold_mean"]), "cold_sem": float(r["cold_sem"]),
                "warm_mean": float(r["warm_mean"]), "warm_sem": float(r["warm_sem"]),
            }
    return {"summary": summary, "families": list(families),
            "N_values": list(N_values), "graphs_per_N": graphs_per_N, "p": p}


def plot(data):
    """One figure per family: cold vs. warm re-election rounds vs. N, so each
    family's comparison is read on its own y-scale (the families differ by a
    lot in absolute rounds)."""
    families = data["families"]
    N_values = data["N_values"]
    summary = data["summary"]
    Ns = np.array(N_values, dtype=float)

    for family in families:
        label = FAMILY_LABEL[family].format(p=data["p"])
        cold_m = np.array([summary[family][N]["cold_mean"] for N in N_values])
        cold_e = np.array([summary[family][N]["cold_sem"] for N in N_values])
        warm_m = np.array([summary[family][N]["warm_mean"] for N in N_values])
        warm_e = np.array([summary[family][N]["warm_sem"] for N in N_values])

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(Ns, cold_m, "-o", color="C3", label="cold start ($w_i=1$)")
        ax.fill_between(Ns, cold_m - cold_e, cold_m + cold_e, color="C3", alpha=0.2)
        ax.plot(Ns, warm_m, "-o", color="C0", label=r"warm start ($w_i=\psi_i$)")
        ax.fill_between(Ns, warm_m - warm_e, warm_m + warm_e, color="C0", alpha=0.2)

        ax.set_xlabel("Number of nodes $N$")
        ax.set_ylabel("Re-election rounds to converge")
        ax.set_title(f"Re-election convergence after hub removal, {label} "
                     f"({data['graphs_per_N']} graphs per $N$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        tag = f"{family}_N{N_values[0]}-{N_values[-1]}_g{data['graphs_per_N']}_p{data['p']}"
        out = OUT_DIR / f"reelection_convergence_{tag}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


def run_experiment(families, N_values, graphs_per_N, p, seed, workers, mode,
                   tau_max, tol, gossip_rounds, tau_init):
    if mode in ("compute", "all"):
        data = compute_convergence_data(
            families, N_values, graphs_per_N, p, seed, workers,
            tau_max=tau_max, tol=tol, gossip_rounds=gossip_rounds, tau_init=tau_init)
        save_csv(data)
    if mode == "compute":
        return
    if mode == "plot":
        data = load_csv(families, N_values, graphs_per_N, p)
    plot(data)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Warm- vs cold-start re-election convergence time vs. N")
    ap.add_argument("--families", nargs="+", choices=("er", "sparse_er"),
                    default=["er", "sparse_er"],
                    help="graph families to compare (er: dense ER; "
                         "sparse_er: ER with constant mean degree, sparse)")
    ap.add_argument("--N", type=int, nargs="+",
                    default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    ap.add_argument("--graphs-per-N", type=int, default=30)
    ap.add_argument("--p", type=float, default=0.5, help="ER edge probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="parallel worker processes (default: all CPUs)")
    ap.add_argument("--tau-max", type=int, default=500,
                    help="max power-iteration rounds before giving up")
    ap.add_argument("--tol", type=float, default=1e-8,
                    help="convergence tolerance on per-node relative change")
    ap.add_argument("--gossip-rounds", type=int, default=100)
    ap.add_argument("--tau-init", type=int, default=500,
                    help="rounds for the initial (cold, full-graph) centrality")
    ap.add_argument("--mode", choices=("compute", "plot", "all"), default="all")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.families, args.N, args.graphs_per_N, args.p, args.seed,
                   args.workers, args.mode, args.tau_max, args.tol,
                   args.gossip_rounds, args.tau_init)
