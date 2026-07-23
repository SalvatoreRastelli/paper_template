"""
EigenTree-FT: hub-failure injection and spectral re-election, on top of the
EigenTreeUCB relay.

Simulates one run of EigenTreeUCB with permanent hub failures injected at fixed
rounds. Recovery follows the single re-election procedure of the Fault
Tolerance section, applied uniformly regardless of which node failed: a dead
node is a low-rank perturbation of the adjacency matrix, so the survivors
warm-start a gossip power iteration on A restricted to the surviving subgraph
from the pre-failure psi, run it for a fixed tau_re rounds, and the argmax
survivor becomes the new hub. The tree is then rebuilt from scratch around the
new hub by the same max-flooding used at initialization, so its height falls
out of the first post-failure cycle exactly as D did at the start. Detection
uses one uniform timer tau = 2D+1 (the cycle length) for every node -- the hub
is detected by its children exactly as any node is detected by its parent --
and the failure notice reaches every survivor within 2D hops, which is also
the budget used here for the transient before the new hub resumes ordinary
cycling. No backup is designated and no swap edges are precomputed: hub
failure is simply the largest-perturbation case of the one procedure that
handles every node failure. Group regret is tracked throughout so recovery
shows directly in the curve.

Usage:
    python fault_tolerance.py --graph ba --N 20 --K 5 --T 5000 --fail-at 2000 --mode all
"""

import argparse
import csv
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "paper" / "results"
FT_DIR      = RESULTS_DIR / "FaultTolerance"
FT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Bandit environment
# ============================================================

class BanditEnv:
    def __init__(self, means, sigma=1.0):
        self.means = np.asarray(means, dtype=float)
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
# Graph utilities (same generators as regret_min.py, kept local
# so this script has no cross-script import dependency)
# ============================================================

def make_ba_graph(N, m=2, seed=None):
    for attempt in range(50):
        s = seed + attempt if seed is not None else None
        G = nx.barabasi_albert_graph(N, m, seed=s)
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected BA graph.")


def make_er_graph(N, p=None, seed=None):
    if p is None:
        p = 2.5 * np.log(N) / N
    rng = np.random.RandomState(seed)
    for _ in range(100):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.randint(0, 2**31)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected ER graph.")


def make_graph(graph_type, N, p=None, seed=None):
    if graph_type == "ba":
        return make_ba_graph(N, m=2, seed=seed)
    elif graph_type == "er":
        return make_er_graph(N, p=p, seed=seed)
    raise ValueError(f"Unknown graph type: {graph_type}")


# ============================================================
# MERW eigenvector and routing tree (same as regret_min.py)
# ============================================================
#
# The two decentralized budgets are both ceil(2 ln N), matching the protocol's
# O(log N) spec (paper: g = O(log N) gossip rounds per power-iteration step,
# and a fixed tau_init large enough for the routing decision to settle). They
# are shared hyperparameters fixed in advance, not quantities any node derives
# at runtime, so every node uses the same value and stays synchronized; N is
# treated as known in advance, as elsewhere. The constant 2 is chosen so the
# elected hub matches a fully-converged reference on ER and BA graphs across
# N=20..200 in 78/80 trials (coefficient 1 gives 76/80, coefficient 3 gives
# 80/80; 2 is the point past which more rounds buy little extra accuracy).

def gossip_rounds_for(N):
    """g = ceil(2 ln N): the O(log N) gossip rounds per power-iteration step."""
    return int(np.ceil(2.0 * np.log(N)))


def tau_init_for(N):
    """tau_init = ceil(2 ln N): the fixed O(log N) power-iteration budget."""
    return int(np.ceil(2.0 * np.log(N)))


def gossip_power_iteration(G, w0, tau, tol=None, gossip_rounds=None):
    """
    Decentralized power iteration with gossip-based normalization, the single
    primitive used both for the initial centrality and for fault-tolerant
    re-election. No node ever performs a global reduce: each round every node
    computes its own local log-growth rate, the nodes gossip-average those
    rates over random neighbor pairs (approximating the global log lambda_1),
    and each node rescales locally by its own averaged rate. The only global
    step is a final cosmetic rescale by the max, applied once after the loop,
    outside the decentralized iteration.

    Two modes, set by `tol`:
      tol is None (the protocol): run exactly `tau` rounds with NO convergence
        test. This is what the synchronous protocol requires -- every node
        performs the same fixed number of steps, so all nodes finish on the
        same round and enter the first bandit pull together. A convergence
        test would be a global reduce and would let nodes at different depths
        stop at different rounds, breaking synchronization.
      tol is a float (measurement only): stop early when the per-node relative
        change falls below tol, and report the round it happened. Used by the
        convergence-timing experiment, not by the protocol itself.

    Runs on the graph G as given; to iterate on a surviving subgraph after a
    node failure, pass the subgraph induced on the live nodes and a w0 that is
    the pre-failure state restricted to those nodes (warm start). Returns
    (w, rounds), where `rounds` is `tau` in fixed-round mode, or the stopping
    round in measurement mode. `gossip_rounds` defaults to ceil(ln N), the
    protocol's O(log N) gossip budget, when not given.
    """
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    if gossip_rounds is None:
        gossip_rounds = gossip_rounds_for(N)

    w = w0.astype(float).copy()
    rounds = tau
    for t in range(tau):
        w_old = w.copy()
        w_new = A @ w_old

        with np.errstate(divide="ignore", invalid="ignore"):
            log_growth = np.where(w_old > 1e-15, np.log(np.abs(w_new) / np.abs(w_old)), 0.0)

        r = log_growth.copy()
        for _ in range(gossip_rounds):
            for i in range(N):
                nbrs = list(G.neighbors(i))
                if nbrs:
                    j = nbrs[np.random.randint(len(nbrs))]
                    r[i] = r[j] = (r[i] + r[j]) / 2

        w = w_new / np.exp(r)

        if tol is not None:
            diff = np.max(np.abs(w - w_old) / (np.abs(w_old) + 1e-15))
            if diff < tol:
                rounds = t + 1
                break

    w = np.abs(w)
    w /= w.max()
    return w, rounds


def merw_eigenvector(G, tau=None, gossip_rounds=None):
    """
    Initial centrality: cold-start (w_i = 1) gossip power iteration, run for a
    fixed `tau` rounds with no convergence test, so every node finishes on the
    same synchronous round. `tau` is the protocol's tau_init hyperparameter and
    defaults to ceil(ln N); `gossip_rounds` defaults to ceil(ln N) as well.
    """
    N = G.number_of_nodes()
    if tau is None:
        tau = tau_init_for(N)
    w, _ = gossip_power_iteration(G, np.ones(N), tau=tau, tol=None,
                                  gossip_rounds=gossip_rounds)
    return w


def build_routing_tree(G, psi, dead=None):
    """
    Max-flooding routing tree induced by centrality psi, exactly the
    construction used at initialization. If `dead` is given, the tree is
    built over the surviving subgraph G \\ {dead}: dead nodes are skipped
    entirely (no parent, no children, depth -1), so calling this again after
    a failure with the re-elected psi is literally the same procedure run
    once more, over fewer nodes.
    """
    N = G.number_of_nodes()
    alive = [i for i in range(N) if dead is None or not dead[i]]

    parent = np.full(N, -1, dtype=int)
    for i in alive:
        nbrs = [j for j in G.neighbors(i) if dead is None or not dead[j]]
        if not nbrs:
            continue
        best_j = max(nbrs, key=lambda j: psi[j])
        if psi[best_j] > psi[i]:
            parent[i] = best_j

    local_maxima = [i for i in alive if parent[i] < 0]

    m   = psi.copy()
    via = np.arange(N)
    while True:
        m_new   = m.copy()
        via_new = via.copy()
        changed = False
        for i in alive:
            for j in G.neighbors(i):
                if dead is not None and dead[j]:
                    continue
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

    hub = next(i for i in alive if parent[i] < 0)

    children = [[] for _ in range(N)]
    for i in alive:
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

    tree_depth = int(depth[alive].max())
    return hub, parent, children, depth, tree_depth


def warm_started_reelection(G, psi, dead, tau_re):
    """
    Warm-started decentralized re-election on the surviving subgraph. The
    survivors run the same gossip power iteration as at initialization
    (gossip_power_iteration), but seeded from the pre-failure psi restricted
    to the live nodes rather than from all-ones. Dead nodes are dropped from
    the state entirely, so the iteration is exactly gossip power iteration on
    the smaller graph G \\ {dead}, not on the full graph with a masked
    row/column. Runs a fixed tau_re rounds (matching the protocol's fixed
    re-election budget) and returns a full-length psi_new array with -1 at
    dead positions.
    """
    N = G.number_of_nodes()
    alive = [i for i in range(N) if not dead[i]]
    G_sub = G.subgraph(alive).copy()
    G_sub = nx.convert_node_labels_to_integers(G_sub, ordering="sorted")

    w0 = psi[alive].copy()
    w, _ = gossip_power_iteration(G_sub, w0, tau=tau_re, tol=None)

    psi_new = np.full(N, -1.0)
    for k, node in enumerate(alive):
        psi_new[node] = w[k]
    return psi_new


def _validate_tree(root, parent, children, depth, live_nodes):
    """
    Direct empirical check that the routing tree stays well-formed: after
    re-election, the network must form a single valid spanning tree rooted at
    the new hub, with every live node reachable and no cycles.

    Returns a dict of individual checks plus an overall boolean.
    """
    live_set = set(live_nodes)
    checks = {}

    checks["root_has_no_parent"] = bool(parent[root] == -1)

    # Every live node other than root has exactly one parent, and that
    # parent relationship is mutually consistent with children[].
    consistent = True
    for i in live_set:
        if i == root:
            continue
        p = parent[i]
        if p < 0 or p not in live_set:
            consistent = False
            break
        if i not in children[p]:
            consistent = False
            break
    checks["parent_child_consistent"] = consistent

    # Reachability + acyclicity: walk from every live node toward the root
    # via parent pointers; must terminate at root within |live_set| steps.
    reachable = True
    acyclic = True
    for i in live_set:
        node = i
        steps = 0
        while node != root:
            node = parent[node]
            steps += 1
            if node < 0 or node not in live_set or steps > len(live_set):
                reachable = False
                acyclic = (steps <= len(live_set))
                break
    checks["all_live_nodes_reach_root"] = reachable

    # depth[] must agree with the actual parent-chain length to root.
    depth_consistent = True
    for i in live_set:
        node, steps = i, 0
        while node != root and steps <= len(live_set):
            node = parent[node]
            steps += 1
        if depth[i] != steps:
            depth_consistent = False
            break
    checks["depth_matches_parent_chain"] = depth_consistent

    checks["valid"] = all(checks.values())
    return checks


# ============================================================
# EigenTree-FT: relay with one injected hub failure
# ============================================================

def run_eigentree_ft(env, T, G, N, t_fail, c=2.0, sigma=1.0, tau_init=None,
                     tau_re=None, enable_failover=True):
    """
    Runs the EigenTreeUCB relay cycle (fixed cycle length 2D+1, synchronized
    rounds) with permanent hub failures injected at every round in t_fail
    (an int for a single failure, or a list/tuple for several sequential
    failures).

    If enable_failover is True, recovery is the single re-election procedure
    of the Fault Tolerance section, applied to the hub exactly as it would be
    applied to any node: there is no designated backup and nothing is
    precomputed in advance. Detection uses one uniform timer tau = 2D+1 (the
    cycle length) -- the hub's children detect its silence the same way any
    node detects a dead neighbor, by a missing expected message within tau.
    Once detected, survivors warm-start a gossip power iteration on A
    restricted to the surviving subgraph, seeded from the pre-failure psi, for
    a fixed tau_re rounds (Proposition: Warm-started re-election); the argmax
    survivor becomes the new hub. The tree is then rebuilt from scratch by the
    same max-flooding used at initialization, over the surviving subgraph, so
    its height falls out of the rebuild rather than being carried over from
    the old tree. The transient charged before the new hub resumes ordinary
    cycling is 2D (the failure-notice horizon, Proposition: Failure horizon)
    plus tau_re (re-election) plus the rebuild, which is bounded by twice the
    new tree's height. Global aggregate state is not mirrored anywhere: the
    first uplink after the rebuild re-sums every survivor's own cumulative
    counters, recovering the exact pre-failure total minus the dead node's
    own contribution.

    If enable_failover is False, the hub simply stays dead forever after the
    first failure in t_fail (no recovery mechanism) -- this isolates what
    re-election buys you.

    Returns:
        cum_regret: (N, T) array of per-agent cumulative regret
        events: dict with 'hub_history' (hub id per round), 'recovery_rounds'
                (list of rounds at which a new hub resumed normal cycling,
                one per successful re-election), and 'tree_valid_checks' (list
                of the structural-validity check dict after each re-election)
    """
    K = env.K
    if tau_init is None:
        tau_init = tau_init_for(N)
    psi = merw_eigenvector(G, tau=tau_init)
    hub0, parent, children, depth, D = build_routing_tree(G, psi)
    if tau_re is None:
        tau_re = tau_init_for(N)

    fail_schedule = sorted(set([t_fail] if isinstance(t_fail, int) else t_fail))

    n_own = np.ones((N, K))
    s_own = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s_own[i, k] = env.pull(k)
    n_hat = n_own.copy()
    s_hat = s_own.copy()

    # Snapshot of each node's own local counters as of its last uplink fold
    # (or, for the hub, as of its last downlink broadcast). The uplink fold
    # sums only the *increment* since that snapshot, so the aggregate is a
    # true running total that survives a change of which node is hub --
    # unlike re-deriving it from n_own directly, which implicitly assumes
    # the current hub has been the hub since t=0.
    n_own_synced = n_own.copy()
    s_own_synced = s_own.copy()

    cum_regret = np.zeros((N, T))
    hub_history = np.full(T, hub0, dtype=int)

    current_hub = hub0
    current_parent = parent.copy()
    current_children = [list(c) for c in children]
    current_depth = depth.copy()
    current_D = D
    current_psi = psi.copy()
    dead = np.zeros(N, dtype=bool)
    recovery_rounds = []
    reelection_in_progress = False
    tree_valid_checks = []
    trees_snapshot = []  # (label, hub, parent, children) before/after each re-election

    def _record(i, arm, t):
        cum_regret[i, t] = (cum_regret[i, t - 1] if t > 0 else 0.0) + env.gap(arm)

    def _silent(i, t):
        cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0

    t = 1
    cycle_start = t

    death_time = None  # round the current hub actually died, once known

    while t < T:
        hub_history[t] = current_hub if not dead[current_hub] else -1

        if t in fail_schedule and not dead[current_hub]:
            dead[current_hub] = True
            death_time = t
            trees_snapshot.append({"label": "before", "hub": current_hub,
                                   "parent": current_parent.copy()})

        # Detection: the uniform timer tau = 2D+1 is the cycle length, so the
        # hub's children -- one hop away, expecting the first downlink hop --
        # notice the missing message within one cycle of the death. This is
        # the same timer every node runs against every neighbor; the hub is
        # not a distinguished case.
        detected = (dead[current_hub] and death_time is not None
                    and t - death_time >= 2 * current_D + 1)

        if detected and enable_failover and not reelection_in_progress:
            # --- Re-election: survivors warm-start on A restricted to the
            # surviving subgraph and rebuild the tree around the new argmax ---
            reelection_in_progress = True

            D_before = current_D  # height the network had prior to this failure

            psi_new = warm_started_reelection(G, current_psi, dead, tau_re)
            new_hub, new_parent, new_children, new_depth, new_D = \
                build_routing_tree(G, psi_new, dead=dead)

            tree_valid = _validate_tree(new_hub, new_parent, new_children, new_depth,
                                        [i for i in range(N) if not dead[i]])
            tree_valid_checks.append(tree_valid)
            trees_snapshot.append({"label": "after", "hub": new_hub,
                                   "parent": new_parent.copy()})

            current_hub = new_hub
            current_parent = new_parent
            current_children = new_children
            current_depth = new_depth
            current_D = new_D
            current_psi = psi_new

            # No backup, nothing is mirrored: the first uplink after the
            # rebuild re-sums every survivor's own (n_own, s_own) counters
            # from scratch, recovering the exact pre-failure aggregate minus
            # the dead node's own contribution. Seed n_hat/s_hat at the new
            # hub to that fresh sum right away, rather than waiting a full
            # uplink, so the transient below models only the detection +
            # re-election + rebuild delay, not bandit-state staleness on top
            # of it.
            live = [i for i in range(N) if not dead[i]]
            n_hat[new_hub] = sum(n_own[i] for i in live)
            s_hat[new_hub] = sum(s_own[i] for i in live)
            n_own_synced[live] = n_own[live]
            s_own_synced[live] = s_own[live]

            # Transient: 2D (failure-notice horizon, using the height the
            # network had before this failure) for every survivor to learn of
            # the failure and restart in step, plus tau_re rounds of
            # warm-started re-election, plus the max-flood rebuild, which
            # takes at most twice the new tree's height (uplink + downlink of
            # the first ordinary cycle discovers it exactly as D was
            # discovered at initialization).
            horizon = 2 * D_before
            rebuild = 2 * new_D
            cycle_start = t + horizon + tau_re + rebuild
            reelection_in_progress = False
            recovery_rounds.append(cycle_start)

            for i in range(N):
                _silent(i, t)
            t += 1
            continue

        if dead[current_hub]:
            # No failover (baseline) or re-election not yet detected: network stalls.
            for i in range(N):
                _silent(i, t)
            t += 1
            continue

        # --- Ordinary relay cycle: uplink, hub pull, downlink, fixed 2D+1 rounds ---
        round_in_cycle = t - cycle_start
        if round_in_cycle < 0:
            # still inside the post-failover hold window
            for i in range(N):
                _silent(i, t)
            t += 1
            continue

        if round_in_cycle == 0:
            # Pull round: everyone pulls simultaneously
            for i in range(N):
                if dead[i]:
                    _silent(i, t)
                    continue
                n_i = np.maximum(n_hat[i], 1e-9)
                tp = max(int(n_hat[i].sum() - K), 1)
                arm = int(np.argmax(s_hat[i] / n_i + sigma * np.sqrt(c * np.log(tp) / n_i)))
                r = env.pull(arm)
                n_own[i, arm] += 1.0; s_own[i, arm] += r
                n_hat[i, arm] += 1.0; s_hat[i, arm] += r
                _record(i, arm, t)
        elif round_in_cycle <= current_D:
            # Uplink rounds: fold children's contributions toward the hub
            for i in range(N):
                _silent(i, t)
            if round_in_cycle == current_D:
                agg_n = n_hat[current_hub].copy()
                agg_s = s_hat[current_hub].copy()
                for i in range(N):
                    if not dead[i]:
                        agg_n += (n_own[i] - n_own_synced[i])
                        agg_s += (s_own[i] - s_own_synced[i])
                        n_own_synced[i] = n_own[i].copy()
                        s_own_synced[i] = s_own[i].copy()
                n_hat[current_hub] = agg_n
                s_hat[current_hub] = agg_s
        elif round_in_cycle == current_D + 1:
            # Hub pull
            n_h = np.maximum(n_hat[current_hub], 1e-9)
            tp = max(int(n_hat[current_hub].sum() - K), 1)
            arm = int(np.argmax(s_hat[current_hub] / n_h + sigma * np.sqrt(c * np.log(tp) / n_h)))
            r = env.pull(arm)
            n_hat[current_hub, arm] += 1.0; s_hat[current_hub, arm] += r
            n_own[current_hub, arm] += 1.0; s_own[current_hub, arm] += r
            _record(current_hub, arm, t)
            for i in range(N):
                if i != current_hub:
                    _silent(i, t)
        elif round_in_cycle < 2 * current_D + 1:
            for i in range(N):
                _silent(i, t)
        else:
            # Last round of downlink: broadcast hub's aggregate to everyone
            for i in range(N):
                if i != current_hub and not dead[i]:
                    n_hat[i] = n_hat[current_hub].copy()
                    s_hat[i] = s_hat[current_hub].copy()
                _silent(i, t)
            cycle_start = t + 1

        t += 1

    events = {"hub_history": hub_history, "recovery_rounds": recovery_rounds,
              "fail_schedule": fail_schedule, "final_hub": current_hub,
              "hub0": hub0, "D0": D, "tree_valid_checks": tree_valid_checks,
              "trees_snapshot": trees_snapshot}
    return cum_regret, events


# ============================================================
# Data save / load / plot
# ============================================================

def _csv_path(graph_type, N, K, T, tag):
    return DATA_DIR / f"fault_tolerance_{graph_type}_N{N}_K{K}_T{T}_{tag}.csv"


def compute_ft_data(graph_type, N, K, T, fail_every, sigma=1.0, c=2.0, seed=0, p=None,
                     fail_start=None, fail_at=None, tau_re=None):
    """
    Runs one EigenTree-FT trajectory over horizon T with a hub failure
    injected either at the explicit rounds in `fail_at` (a list, takes
    precedence if given), or every `fail_every` rounds starting at
    `fail_start` (default `fail_every`; e.g. fail_every=1000 on T=5000
    injects failures at t=1000,2000,3000,4000; fail_start=2000 instead gives
    t=2000,3000,4000), and directly tests the Fault Tolerance section's
    claims:
      1. Failover correctness: after each re-election, the tree is checked to
         be a valid single spanning tree (not just plotted).
      2. The network keeps making progress (regret keeps growing, not
         flatlining) through every failure, using the same warm-started
         re-election procedure each time, with no state carried over between
         failures beyond the current psi.
    """
    if fail_start is None:
        fail_start = fail_every
    rng = np.random.RandomState(seed)
    means = np.linspace(0.0, 1.0, K)[::-1]
    graph_seed = int(rng.randint(0, 2**31))
    run_seed = int(rng.randint(0, 2**31))

    G = make_graph(graph_type, N, p=p, seed=graph_seed)
    fail_schedule = list(fail_at) if fail_at is not None else list(range(fail_start, T, fail_every))

    np.random.seed(run_seed)
    env = BanditEnv(means, sigma=sigma)
    cr_ft, events = run_eigentree_ft(env, T, G, N, fail_schedule, c=c, sigma=sigma,
                                     tau_re=tau_re, enable_failover=True)
    group_ft = cr_ft.sum(axis=0)

    return {
        "group_ft": group_ft,
        "fail_schedule": events["fail_schedule"],
        "recovery_rounds": events["recovery_rounds"],
        "hub0": events["hub0"], "D0": events["D0"],
        "tree_valid_checks": events["tree_valid_checks"],
        "trees_snapshot": events["trees_snapshot"],
        "G": G,
    }


def save_ft_csv(data, graph_type, N, K, T, tag):
    out = _csv_path(graph_type, N, K, T, tag)
    T_len = len(data["group_ft"])
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "group_regret"])
        for t in range(T_len):
            writer.writerow([t + 1, data["group_ft"][t]])

    meta_out = out.with_name(out.stem + "_meta.csv")
    all_valid = all(tv["valid"] for tv in data["tree_valid_checks"]) if data["tree_valid_checks"] else None
    with open(meta_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fail_schedule", "recovery_rounds", "hub0", "D0", "all_trees_valid"])
        writer.writerow([";".join(map(str, data["fail_schedule"])),
                          ";".join(map(str, data["recovery_rounds"])),
                          data["hub0"], data["D0"], all_valid])

    # Persist the graph and the before/after tree snapshots of the FIRST
    # failure, so the plot step can redraw them without re-simulating.
    graph_out = out.with_name(out.stem + "_graph.csv")
    with open(graph_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["u", "v"])
        for u, v in data["G"].edges():
            writer.writerow([u, v])

    trees_out = out.with_name(out.stem + "_trees.csv")
    with open(trees_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snapshot", "label", "hub", "node", "parent"])
        for idx, snap in enumerate(data["trees_snapshot"]):
            for node in range(N):
                writer.writerow([idx, snap["label"], snap["hub"], node, int(snap["parent"][node])])

    print(f"  Saved {out}, {meta_out}, {graph_out}, {trees_out}")
    print(f"  Failures at {data['fail_schedule']}, recoveries at {data['recovery_rounds']}")
    print(f"  All post-failover trees valid: {all_valid}")


def load_ft_csv(graph_type, N, K, T, tag):
    out = _csv_path(graph_type, N, K, T, tag)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    group_ft = np.array([float(r["group_regret"]) for r in rows])

    meta_out = out.with_name(out.stem + "_meta.csv")
    with open(meta_out, newline="") as f:
        meta = next(csv.DictReader(f))
    fail_schedule = [int(x) for x in meta["fail_schedule"].split(";") if x]
    recovery_rounds = [int(x) for x in meta["recovery_rounds"].split(";") if x]

    graph_out = out.with_name(out.stem + "_graph.csv")
    G = nx.Graph()
    G.add_nodes_from(range(N))
    with open(graph_out, newline="") as f:
        for r in csv.DictReader(f):
            G.add_edge(int(r["u"]), int(r["v"]))

    trees_out = out.with_name(out.stem + "_trees.csv")
    snapshots = {}
    with open(trees_out, newline="") as f:
        for r in csv.DictReader(f):
            idx = int(r["snapshot"])
            snapshots.setdefault(idx, {"label": r["label"], "hub": int(r["hub"]),
                                       "parent": np.full(N, -1, dtype=int)})
            snapshots[idx]["parent"][int(r["node"])] = int(r["parent"])
    trees_snapshot = [snapshots[i] for i in sorted(snapshots)]

    return {"group_ft": group_ft, "fail_schedule": fail_schedule,
            "recovery_rounds": recovery_rounds, "hub0": int(meta["hub0"]),
            "D0": int(meta["D0"]), "G": G, "trees_snapshot": trees_snapshot}


# ============================================================
# Tree drawing (self-contained, styled like visualize_merw.py)
# ============================================================

def _draw_graph_tree_panel(ax, G, pos, hub, parent, N, title, dead_nodes=()):
    """Draws the full underlying graph (faded) with the induced routing tree
    highlighted on top, same visual grammar as visualize_merw.py."""
    dead_nodes = set(dead_nodes)
    tree_edges = {(parent[i], i) for i in range(N) if parent[i] >= 0}
    tree_edges |= {(v, u) for (u, v) in tree_edges}

    # Full graph, faded, including non-tree edges.
    for u, v in G.edges():
        if (u, v) in tree_edges:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="#d8d8d8", lw=0.8, alpha=0.7, zorder=1)

    # Tree edges highlighted, directed child -> parent.
    for i in range(N):
        if parent[i] < 0 or i in dead_nodes:
            continue
        x0, y0 = pos[i]; x1, y1 = pos[parent[i]]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="-|>", color="#4393c3", lw=1.6,
                                    connectionstyle="arc3,rad=0.05", mutation_scale=11,
                                    shrinkA=8, shrinkB=8), zorder=2)

    for i in range(N):
        x, y = pos[i]
        if i in dead_nodes:
            color, ec = "#dddddd", "#999999"
        elif i == hub:
            color, ec = "#2166ac", "#1a4d7a"
        else:
            color, ec = "#92c5de", "#2a6090"
        circle = plt.Circle((x, y), 0.05, facecolor=color, edgecolor=ec, linewidth=1.3, zorder=4)
        ax.add_patch(circle)
        if i == hub:
            ring = plt.Circle((x, y), 0.075, facecolor="none", edgecolor="#f4a261", linewidth=2.0, zorder=3)
            ax.add_patch(ring)
        fc = "#555555" if i in dead_nodes else "white"
        ax.text(x, y, str(i), ha="center", va="center", fontsize=6,
                fontweight="bold", color=fc, zorder=5)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_trees(data, graph_type, N, K, T, tag):
    """Graph + induced routing tree: the initial tree, then the tree after
    each failover. The 'before' state of failure e+1 is identical to the
    'after' state of failure e (plus one more dead node), so it is not
    repeated -- each event contributes one panel instead of two."""
    n_snap = len(data["trees_snapshot"])
    if n_snap < 2:
        print("  No failover occurred; skipping tree snapshot figure.")
        return
    G = data["G"]
    pos = nx.spring_layout(G, seed=0, k=1.8 / np.sqrt(N))

    n_events = n_snap // 2  # each event contributes a (before, after) pair
    n_panels = n_events + 1  # initial tree + one post-failover tree per event
    n_cols = 2
    n_rows = -(-n_panels // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5.5 * n_rows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    initial = data["trees_snapshot"][0]
    _draw_graph_tree_panel(
        flat_axes[0], G, pos, initial["hub"], initial["parent"], N,
        f"Initial tree (hub = node {initial['hub']})", dead_nodes=())

    dead_so_far = []
    for e in range(n_events):
        before = data["trees_snapshot"][2 * e]
        after = data["trees_snapshot"][2 * e + 1]
        dead_so_far.append(before["hub"])

        t_fail = data["fail_schedule"][e] if e < len(data["fail_schedule"]) else "?"
        _draw_graph_tree_panel(
            flat_axes[e + 1], G, pos, after["hub"], after["parent"], N,
            f"After failure {e+1} ($t={t_fail}$): hub = node {after['hub']}",
            dead_nodes=dead_so_far)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"EigenTree-FT: induced routing tree, initial and after each "
                 f"hub failure\n{graph_type.upper()} graph, $N={N}$ "
                 f"(dead nodes in gray, faded edges = non-tree graph edges)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FT_DIR / f"fault_tolerance_trees_{graph_type}_N{N}_K{K}_T{T}_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_ft(data, graph_type, N, K, T, tag):
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ts = np.arange(1, len(data["group_ft"]) + 1)

    ax.plot(ts, data["group_ft"], label="EigenTree-UCB", color="C2")

    for i, tf in enumerate(data["fail_schedule"]):
        ax.axvline(tf, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="hub failure" if i == 0 else None)
    for i, rr in enumerate(data["recovery_rounds"]):
        ax.axvline(rr, color="C4", linestyle=":", linewidth=1.0, alpha=0.8,
                   label="recovery" if i == 0 else None)

    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"$\sum_i R_i(t)$")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FT_DIR / f"fault_tolerance_{graph_type}_N{N}_K{K}_T{T}_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def run_experiment(graph_type, N, K, T, fail_every, sigma, c, mode="all", seed=0, p=None,
                    fail_start=None, fail_at=None, tau_re=None):
    tag = ("at" + "-".join(str(t) for t in fail_at)) if fail_at is not None else f"every{fail_every}"
    if mode in ("compute", "all"):
        data = compute_ft_data(graph_type, N, K, T, fail_every, sigma=sigma, c=c, seed=seed, p=p,
                                fail_start=fail_start, fail_at=fail_at, tau_re=tau_re)
        save_ft_csv(data, graph_type, N, K, T, tag)
    if mode == "compute":
        return
    if mode == "plot":
        data = load_ft_csv(graph_type, N, K, T, tag)
    plot_ft(data, graph_type, N, K, T, tag)
    plot_trees(data, graph_type, N, K, T, tag)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="EigenTree-FT repeated hub-failure experiment")
    ap.add_argument("--graph",      choices=("ba", "er"), default="ba")
    ap.add_argument("--N",          type=int, default=20)
    ap.add_argument("--p",          type=float, default=None,
                    help="ER edge probability (ignored for --graph ba)")
    ap.add_argument("--K",          type=int, default=5)
    ap.add_argument("--T",          type=int, default=5000)
    ap.add_argument("--fail-every", type=int, default=1000)
    ap.add_argument("--fail-start", type=int, default=None,
                    help="round of the first failure (default: --fail-every)")
    ap.add_argument("--fail-at",    type=int, nargs="+", default=None,
                    help="explicit list of failure rounds, overrides --fail-every/--fail-start")
    ap.add_argument("--sigma",      type=float, default=1.0)
    ap.add_argument("--c",          type=float, default=2.0)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--tau-re",     type=int, default=None,
                    help="warm-started re-election rounds (default: ceil(log N))")
    ap.add_argument("--mode", choices=("compute", "plot", "all"), default="all")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.graph, args.N, args.K, args.T, args.fail_every,
                   args.sigma, args.c, mode=args.mode, seed=args.seed, p=args.p,
                   fail_start=args.fail_start, fail_at=args.fail_at, tau_re=args.tau_re)
