"""
Best arm identification (BAI) experiment.

Compares two fixed-confidence BAI algorithms by the total number of arm pulls
needed to identify the best arm with probability at least 1 - delta:
  - Hillel-BAI     : distributed successive elimination under all-to-all broadcast.
  - EigenTree-BAI  : the same elimination schedule routed over the EigenTree hub.

Rewards are drawn from Normal(mu_k, sigma^2) distributions.

Usage:
  python best_arm_id.py --graph er --K 10 --bai-runs 50 --delta 0.05
  python best_arm_id.py --graph er --K 10 --mode plot   # re-plot from cached data

Output:
  results/BAI/merw_ucb_bai_*.pdf
"""

import argparse
import csv
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parent / "merw.mplstyle")
# AAAI single-column width in inches (\columnwidth = 239.39pt / 72.27pt-per-in).
# Paper figures are authored at exactly this width and included at
# width=\columnwidth, so a point in matplotlib equals a point on the page.
COLUMN_WIDTH_IN = 3.317
import numpy as np
import networkx as nx
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")

# Self-contained: this archive writes its own data/ and results/ beside
# the scripts, so it can be unpacked and run anywhere without the
# surrounding repository.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BAI_DIR     = RESULTS_DIR / "BAI"
REGRET_DIR  = RESULTS_DIR / "Regret"
BAI_DIR.mkdir(parents=True, exist_ok=True)
REGRET_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Doubles as the CSV "algo" key and the plot legend label.
EIGENTREE_BAI_NAME = "EigenTree-BAI"


# ============================================================
# Bandit environment  (reward-based, Normal distributions)
# ============================================================

class BanditEnv:
    """
    Shared stochastic reward environment.
    All agents face the same arm means; rewards are Normal(mu_k, sigma^2).
    """
    def __init__(self, means, sigma=1.0):
        self.means = np.asarray(means, dtype=float)   # shape (K,)
        self.K = len(self.means)
        self.sigma = sigma

    def pull(self, arm):
        return float(np.random.normal(self.means[arm], self.sigma))

    @property
    def best_mean(self):
        return self.means.max()

    def gap(self, arm):
        return self.best_mean - self.means[arm]


# ============================================================
# Graph utilities
# ============================================================

def make_ba_graph(N, m=2, seed=None):
    """Barabasi-Albert preferential attachment graph."""
    for attempt in range(50):
        s = seed + attempt if seed is not None else None
        G = nx.barabasi_albert_graph(N, m, seed=s)
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected BA graph.")


def make_er_graph(N, seed=None, p=None):
    if p is None:
        p = 2.5 * np.log(N) / N
    rng = np.random.RandomState(seed)
    for _ in range(100):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.randint(0, 2**31)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected ER graph.")


def make_barbell_graph(N, seed=None):
    """Two complete graphs of size N//2 joined by a single bridge edge."""
    n1 = N // 2
    n2 = N - n1
    G = nx.complete_graph(n1)
    clique2 = nx.relabel_nodes(nx.complete_graph(n2), {i: i + n1 for i in range(n2)})
    G = nx.compose(G, clique2)
    G.add_edge(n1 - 1, n1)
    return G


def make_grid_graph(N, seed=None):
    """2D grid graph with side length floor(sqrt(N)), relabeled 0..n-1."""
    k = int(np.floor(np.sqrt(N)))
    G = nx.grid_2d_graph(k, k)
    G = nx.convert_node_labels_to_integers(G)
    return G


def make_star_graph(N, seed=None):
    """Star graph with one hub connected to N-1 leaves."""
    return nx.star_graph(N - 1)


def make_graph(graph_type, N, seed=None, p=None):
    if graph_type == "ba":
        return make_ba_graph(N, m=2, seed=seed)
    elif graph_type == "er":
        return make_er_graph(N, seed=seed, p=p)
    elif graph_type == "barbell":
        return make_barbell_graph(N, seed=seed)
    elif graph_type == "grid":
        return make_grid_graph(N, seed=seed)
    elif graph_type == "star":
        return make_star_graph(N, seed=seed)
    raise ValueError(f"Unknown graph type: {graph_type}")


# ============================================================
# MERW eigenvector (power iteration on adjacency matrix)
# ============================================================

def merw_eigenvector(G, tau=None, gossip_rounds=None, gossip_seed=0):
    """Distributed power iteration with gossip-based normalization.

    Implements Jelasity, Canright, Engo-Monsen (EuroPar 2007):
      - Each node i holds w_i initialized to 1.
      - Iteration: w_i <- sum_{j in N(i)} w_j  (one round of neighbor exchange).
      - Normalization: each node tracks its local log-growth rate r_i = log(w_new_i / w_old_i),
        then gossips r_i by pairwise averaging for gossip_rounds rounds to approximate
        the global geometric mean growth rate. Each node divides w_i by exp(r_i).
      - No global norm or knowledge of N required.

    Runs a FIXED tau rounds with no convergence test, so every node finishes on
    the same synchronous round and enters max-flooding together (a convergence
    test would be a global reduce and would desynchronize nodes at different
    depths). Both tau (the tau_init hyperparameter) and gossip_rounds (g)
    default to ceil(2 ln N), the protocol's O(log N) budgets.
    """
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    if tau is None:
        tau = int(np.ceil(2.0 * np.log(N)))
    if gossip_rounds is None:
        gossip_rounds = int(np.ceil(2.0 * np.log(N)))

    # Gossip partner selection draws from a generator of its own rather than the
    # global one used for rewards, so building the tree does not advance the
    # shared reward stream and every algorithm sees the same rewards per seed.
    gossip_rng = np.random.RandomState(gossip_seed)

    w = np.ones(N)
    log_growth = np.zeros(N)

    for _ in range(tau):
        w_old = w.copy()
        w_new = A @ w_old

        with np.errstate(divide="ignore", invalid="ignore"):
            log_growth = np.where(w_old > 1e-15, np.log(np.abs(w_new) / np.abs(w_old)), 0.0)

        r = log_growth.copy()
        for _ in range(gossip_rounds):
            for i in range(N):
                nbrs = list(G.neighbors(i))
                if nbrs:
                    j = nbrs[gossip_rng.randint(len(nbrs))]
                    r[i] = r[j] = (r[i] + r[j]) / 2

        w = w_new / np.exp(r)

    w = np.abs(w)
    lam = np.exp(np.mean(log_growth))
    w /= w.max()
    return w, lam


# ============================================================
# MERW routing tree construction (max-flooding, single hub)
# ============================================================

def build_routing_tree(G, psi):
    """
    Builds a spanning routing tree rooted at the global psi maximum.
    Step 1: local gradient (each node points to highest-psi neighbor).
    Step 2: max-flooding to merge all local maxima into a single tree.
    Returns hub, parent, children, depth, tree_depth.
    """
    N = G.number_of_nodes()

    parent = np.full(N, -1, dtype=int)
    for i in range(N):
        nbrs = list(G.neighbors(i))
        if not nbrs:
            continue
        best_j = max(nbrs, key=lambda j: psi[j])
        if psi[best_j] > psi[i]:
            parent[i] = best_j

    local_maxima = [i for i in range(N) if parent[i] < 0]

    m   = psi.copy()
    via = np.arange(N)
    while True:
        m_new   = m.copy()
        via_new = via.copy()
        changed = False
        for i in range(N):
            for j in G.neighbors(i):
                if m[j] > m_new[i]:
                    m_new[i]   = m[j]
                    via_new[i] = j
                    changed    = True
        m   = m_new
        via = via_new
        if not changed:
            break

    for i in local_maxima:
        if m[via[i]] > psi[i]:
            parent[i] = via[i]

    hub = next(i for i in range(N) if parent[i] < 0)

    children = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            children[parent[i]].append(i)

    depth = np.full(N, -1, dtype=int)
    depth[hub] = 0
    queue = [hub]
    while queue:
        node = queue.pop(0)
        for c in children[node]:
            depth[c] = depth[node] + 1
            queue.append(c)

    tree_depth = int(depth.max())
    return hub, parent, children, depth, tree_depth


# ============================================================
# Algorithm: Independent UCB (baseline, no communication)
# ============================================================


# ============================================================
# BAI algorithms (fixed confidence, measure sample complexity)
# ============================================================

def bai_hillel(env, N, delta=0.05, sigma=1.0):
    """
    Hillel et al. (NeurIPS 2013) Algorithm 3 (Multi-Round epsilon-Arm): distributed successive elimination.
    Returns total arm pulls across all agents until best arm is identified.

    Per the paper: t_0=0, r starts at 0 and increments first each iteration.
      epsilon_r = 2^{-r}
      t_r = (2 / (N * epsilon_r^2)) * ln(4 * K * r^2 / delta)
      L_r = t_r - t_{r-1}  (incremental pulls per arm per agent this round)
    Elimination: drop arm i if p_tilde_i < p_tilde_star - epsilon_r
    """
    K = env.K

    surviving = list(range(K))
    total_pulls = 0
    r = 0
    t_prev = 0.0

    while len(surviving) > 1:
        r += 1
        epsilon_r = 2.0 ** (-r)

        # Terminate when epsilon_r is small enough (epsilon=0 target => r grows until |S|=1)
        # Guard against runaway: stop if epsilon is already very small
        if epsilon_r < 1e-10:
            break

        log_arg = max(4.0 * K * r * r / delta, np.e)
        t_r = (2.0 / (N * epsilon_r ** 2)) * np.log(log_arg)
        L = int(np.ceil(t_r - t_prev))
        L = max(L, 1)
        t_prev = t_r

        local_sums = {k: 0.0 for k in surviving}
        for k in surviving:
            for i in range(N):
                for _ in range(L):
                    local_sums[k] += env.pull(k)
                    total_pulls += 1

        # Global average per arm (all N agents pulled L times)
        global_mean = {k: local_sums[k] / (N * L) for k in surviving}

        p_star = max(global_mean[k] for k in surviving)
        surviving = [k for k in surviving if global_mean[k] >= p_star - epsilon_r]

    return total_pulls


def bai_merw(env, G, N, delta=0.05, sigma=1.0, c=2.0):
    """
    EigenTree-BAI: Hillel successive elimination routed over the spanning tree.
    Each round pulls L_r times, uplinks the sums, eliminates at the hub, and
    downlinks the surviving set. The elimination schedule is

        L_r = t_r - t_{r-1},  t_r = (2/(N * epsilon_r^2)) * ln(4*K*r^2/delta),
        epsilon_r = 2^{-r},  t_0 = 0.
    """
    K = env.K
    psi, _ = merw_eigenvector(G)
    hub, parent, children, depth, tree_depth = build_routing_tree(G, psi)

    surviving = list(range(K))
    total_pulls = 0
    r = 0
    t_prev = 0.0

    while len(surviving) > 1:
        r += 1
        epsilon_r = 2.0 ** (-r)
        if epsilon_r < 1e-10:
            break

        log_arg = max(4.0 * K * r * r / delta, np.e)
        t_r = (2.0 / (N * epsilon_r ** 2)) * np.log(log_arg)
        L = int(np.ceil(t_r - t_prev))
        L = max(L, 1)
        t_prev = t_r

        # --- Pull phase: every node pulls each surviving arm L times ---
        # Iterate (arm, node, repeat) in the same order as bai_hillel. Both draw
        # the same number of rewards per arm, but the comparison shares a seed,
        # so a different traversal order hands each arm a different slice of the
        # reward stream. The two would then eliminate differently from sampling
        # noise rather than from anything about the protocol, which is exactly
        # the difference this experiment is meant to rule out.
        local_sums = np.zeros((N, K))
        for k in surviving:
            for i in range(N):
                for _ in range(L):
                    local_sums[i, k] += env.pull(k)
                    total_pulls += 1

        # Uplink: pending[i] is the subtree sum node i still owes its parent;
        # one hop per step, so a node at depth d reaches the hub after d steps.
        pending = [local_sums[i].copy() for i in range(N)]
        hub_agg = np.zeros(K)

        # A node at depth d needs d forwarding steps, and the deepest is at
        # tree_depth, so the loop must run that many times for every packet to
        # drain. max(...,1) covers a depth-0 tree, where the hub is the only
        # node with a parent-less pointer.
        for _ in range(max(tree_depth, 1)):
            next_pending = [None] * N
            for i in range(N):
                if i == hub or pending[i] is None:
                    continue
                p = parent[i]
                if p == hub:
                    hub_agg += pending[i]
                else:
                    if next_pending[p] is None:
                        next_pending[p] = pending[i].copy()
                    else:
                        next_pending[p] += pending[i]
            pending = next_pending
        hub_agg += local_sums[hub]

        # Every live contribution must reach the hub exactly once, so the
        # aggregate is the exact sum a centralized instance would hold. This
        # holds on any graph where build_routing_tree returns a spanning tree.
        assert all(p is None for p in pending), "uplink dropped in-flight packets"

        # Hub eliminates arms more than epsilon_r below the best global mean.
        global_mean = {k: hub_agg[k] / (N * L) for k in surviving}
        p_star = max(global_mean[k] for k in surviving)
        surviving = [k for k in surviving if global_mean[k] >= p_star - epsilon_r]

        # Downlink broadcasts S_r and D; communication only, no pulls, so the
        # simulation just advances without touching total_pulls.

    return total_pulls


# ============================================================
# Parallel worker
# ============================================================

def _worker_bai(task):
    (algo_name, run_seed, graph_seed, graph_type, N, K,
     means, sigma, delta, c, nu, p) = task
    np.random.seed(run_seed)
    G = make_graph(graph_type, N, seed=graph_seed, p=p)
    env = BanditEnv(means, sigma=sigma)

    # A grid on N nodes is built with side floor(sqrt(N)), so its realized
    # agent count is not the requested N. Read it off the graph and give both
    # algorithms the same count: the elimination schedule depends on it.
    N_eff = G.number_of_nodes()

    if algo_name == "Hillel-BAI":
        pulls = bai_hillel(env, N_eff, delta=delta, sigma=sigma)
    elif algo_name == EIGENTREE_BAI_NAME:
        pulls = bai_merw(env, G, N_eff, delta=delta, sigma=sigma, c=c)
    else:
        raise ValueError(algo_name)

    return algo_name, pulls


def _run_parallel(tasks, algo_names, n_workers, tag=""):
    n_workers = min(n_workers, max(1, len(tasks)))
    print(f"[{tag}] dispatching {len(tasks)} tasks to {n_workers} workers")
    t0 = time.time()
    results_list = []
    report_every = max(1, len(tasks) // 20)
    with mp.Pool(processes=n_workers) as pool:
        for done, r in enumerate(pool.imap_unordered(_worker, tasks), start=1):
            results_list.append(r)
            if done % report_every == 0 or done == len(tasks):
                print(f"[{tag}] {done}/{len(tasks)} done ({time.time()-t0:.1f}s)")
    by_label = {name: ([], []) for name in algo_names}
    for name, group, hub in results_list:
        by_label[name][0].append(group)
        by_label[name][1].append(hub)
    return {name: (np.array(g).mean(axis=0), np.array(g).std(axis=0),
                   np.array(h).mean(axis=0), np.array(h).std(axis=0))
            for name, (g, h) in by_label.items()}


# ============================================================
# BAI experiment: sample complexity vs N
# ============================================================

BAI_ALGO_NAMES = ["Hillel-BAI", EIGENTREE_BAI_NAME]
BAI_N_VALS = [10, 20, 30, 40, 50]

BAI_STYLES = {
    "Hillel-BAI":       ("C3", "o", "Hillel-BAI"),
    EIGENTREE_BAI_NAME: ("C2", ".", EIGENTREE_BAI_NAME),
}


def _bai_csv_path(graph_type, K):
    return DATA_DIR / f"merw_ucb_bai_{graph_type}_K{K}.csv"


def compute_bai_data(n_runs, K, graph_type, sigma, c, n_workers, nu=0.1,
                     delta=0.05, seed=2, p=None):
    """
    Sweeps N (number of agents) and measures total arm pulls for BAI.
    Hillel theory predicts O(1/N) pulls per agent => O(1) total pulls
    with sqrt(N) per-agent speedup. EigenTree-BAI targets similar scaling.

    Returns {algo: {"mean": [...], "std": [...], "N_vals": [...]}}.
    """
    means = np.linspace(0.0, 1.0, K)[::-1]

    results = {name: {"mean": [], "std": [], "N_vals": BAI_N_VALS} for name in BAI_ALGO_NAMES}
    for N in BAI_N_VALS:
        sub_tasks = []
        sub_rng = np.random.RandomState(seed + N)
        graph_seed = int(sub_rng.randint(0, 2**31))  # fixed graph for all runs at this N
        for _ in range(n_runs):
            # Each algorithm draws its own reward stream. Sharing one seed
            # across both would couple them: they request the same arms in the
            # same order, so they would receive identical rewards, eliminate
            # identically, and report the same pull count with exactly zero
            # variance in the difference. That is a valid check that routing
            # loses no information, but it is not a measurement of sample
            # complexity -- the agreement would be true by construction rather
            # than observed. Independent streams make the comparison an
            # ordinary two-sample one, with the run-to-run spread of a
            # fixed-confidence stopping time showing up honestly on both sides.
            for name in BAI_ALGO_NAMES:
                run_seed = int(sub_rng.randint(0, 2**31))
                sub_tasks.append((name, run_seed, graph_seed, graph_type,
                                  N, K, means, sigma, delta, c, nu, p))

        pulls_by_algo = {name: [] for name in BAI_ALGO_NAMES}
        nw = min(n_workers, max(1, len(sub_tasks)))
        with mp.Pool(processes=nw) as pool:
            for name, pulls in pool.imap_unordered(_worker_bai, sub_tasks):
                pulls_by_algo[name].append(pulls)

        for name in BAI_ALGO_NAMES:
            arr = np.array(pulls_by_algo[name])
            results[name]["mean"].append(arr.mean())
            # Standard deviation across runs, not the standard error of the
            # mean. A fixed-confidence stopping time is genuinely spread out
            # (coefficient of variation ~0.6 here), and s.e.m. shrinks with the
            # run count until it describes how well the mean is pinned down
            # rather than how variable a single run is. The latter is what a
            # reader of this plot needs.
            results[name]["std"].append(arr.std())
        print(f"[BAI] N={N} done")

    return results


def save_bai_csv(results, graph_type, K):
    out = _bai_csv_path(graph_type, K)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "N", "mean_pulls", "std_pulls"])
        for name in BAI_ALGO_NAMES:
            if name not in results:
                continue
            for N, mean_p, std_p in zip(results[name]["N_vals"],
                                         results[name]["mean"],
                                         results[name]["std"]):
                writer.writerow([name, N, mean_p, std_p])
    print(f"  Saved {out}")
    return out


def load_bai_csv(graph_type, K):
    path = _bai_csv_path(graph_type, K)
    rows_by_algo = {name: [] for name in BAI_ALGO_NAMES}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_algo[row["algo"]].append(row)
    results = {}
    for name, rows in rows_by_algo.items():
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["N"]))
        results[name] = {
            "N_vals": [int(r["N"]) for r in rows],
            "mean": [float(r["mean_pulls"]) for r in rows],
            "std": [float(r["std_pulls"]) for r in rows],
        }
    return results


def plot_bai(results, graph_type, K, n_runs, delta=0.05):
    N_vals = next(iter(results.values()))["N_vals"]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 4.1))

    handles, labels = [], []
    for name, (color, marker, label) in BAI_STYLES.items():
        if name not in results:
            continue
        means_arr = np.array(results[name]["mean"])
        stds_arr = np.array(results[name]["std"])
        line = axes[0].errorbar(N_vals, means_arr, yerr=stds_arr,
                                label=label, color=color, marker=marker,
                                linewidth=1.2, capsize=3, markersize=5)
        handles.append(line)
        labels.append(label)

    axes[0].set_xlabel("Number of agents $N$")
    axes[0].set_ylabel("Total arm pulls")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: pulls per agent vs N (should decrease for Hillel: ~1/sqrt(N)) ---
    for name, (color, marker, label) in BAI_STYLES.items():
        if name not in results:
            continue
        means_arr = np.array(results[name]["mean"])
        stds_arr = np.array(results[name]["std"])
        per_agent = means_arr / np.array(N_vals)
        per_agent_std = stds_arr / np.array(N_vals)
        axes[1].errorbar(N_vals, per_agent, yerr=per_agent_std,
                         label=label, color=color, marker=marker,
                         linewidth=1.2, capsize=3, markersize=5)

    axes[1].set_xlabel("Number of agents $N$")
    axes[1].set_ylabel("Pulls per agent")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(45)

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    # Move the y-axis 1e4 offset from the top of each plot to the left edge,
    # above the tick labels rather than floating over the curve. The offset
    # string is only populated after a draw.
    fig.canvas.draw()
    for ax in axes:
        offset = ax.yaxis.get_offset_text()
        exp = offset.get_text()
        offset.set_visible(False)
        if exp:
            ax.text(-0.01, 1.0, exp, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0))
    out = BAI_DIR / f"merw_ucb_bai_{graph_type}_K{K}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def run_bai_experiment(n_runs, K, graph_type, sigma, c, n_workers, nu=0.1,
                       delta=0.05, seed=2, mode="all", p=None):
    if mode in ("compute", "all"):
        results = compute_bai_data(n_runs, K, graph_type, sigma, c, n_workers,
                                    nu=nu, delta=delta, seed=seed, p=p)
        save_bai_csv(results, graph_type, K)
    if mode == "compute":
        return
    if mode == "plot":
        results = load_bai_csv(graph_type, K)
    plot_bai(results, graph_type, K, n_runs, delta=delta)


# ============================================================


# The AAAI source tree, where supplementary.tex \input's these tables.
# DATA_DIR already points at the paper/ tree this script belongs to, so
# derive from it rather than counting parents (the repo and the archived
# copy of this script sit at different depths).




# ============================================================
# CLI
# ============================================================

def resolve_n_workers(cli_value):
    if cli_value is not None:
        return max(1, int(cli_value))
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    return max(1, mp.cpu_count())


def parse_args():
    p = argparse.ArgumentParser(description="EigenTreeUCB experiment")
    p.add_argument("--n-runs",    type=int,   default=50)
    p.add_argument("--T",         type=int,   default=5_000)
    p.add_argument("--N",         type=int,   default=20,
                   help="Number of agents (graph nodes)")
    p.add_argument("--K",         type=int,   default=5,
                   help="Number of arms")
    p.add_argument("--graph",     choices=("ba", "er", "barbell", "grid", "star"),
                   default="ba",
                   help="Graph type: ba=Barabasi-Albert, er=Erdos-Renyi")
    p.add_argument("--p-er",      type=float, default=None,
                   help="ER edge probability override (default: 2.5*ln(N)/N; ignored for ba)")
    p.add_argument("--sigma",     type=float, default=1.0,
                   help="Reward noise std (Normal rewards)")
    p.add_argument("--c",         type=float, default=2.0,
                   help="UCB exploration constant")
    p.add_argument("--nu",        type=float, default=0.1,
                   help="Transfer weight exponent: alpha_i = 1 - (psi_i/psi_parent)^nu")
    p.add_argument("--n-workers", type=int,   default=None)
    p.add_argument("--delta",     type=float, default=0.05,
                   help="Confidence parameter for BAI experiment")
    p.add_argument("--bai-runs",  type=int,   default=30,
                   help="Runs for the BAI sample-complexity experiment")
    p.add_argument("--mode", choices=("compute", "plot", "all"), default="all",
                   help="compute: run the experiment and save data/*.csv only; "
                        "plot: render the figure from an existing CSV; "
                        "all: compute then plot (default)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_workers = resolve_n_workers(args.n_workers)
    print(
        f"[config] graph={args.graph}, N={args.N}, K={args.K}, T={args.T}, "
        f"sigma={args.sigma}, c={args.c}, runs={args.n_runs}, workers={n_workers}, "
        f"mode={args.mode}"
    )
    start = time.time()
    run_bai_experiment(
        n_runs=args.bai_runs,
        K=args.K,
        graph_type=args.graph,
        sigma=args.sigma,
        c=args.c,
        n_workers=n_workers,
        nu=args.nu,
        delta=args.delta,
        mode=args.mode,
        p=args.p_er,
    )
    print(f"\nTotal runtime: {time.time() - start:.2f}s")
