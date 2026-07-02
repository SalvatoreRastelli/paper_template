"""
EigenTree-FT: hub-failure injection and recovery, on top of the EigenTreeUCB relay.

Simulates one run of EigenTreeUCB with a single permanent hub failure injected
at a fixed round t_fail. The hub's designated backup (its highest-psi direct
child) detects the failure via timeout, self-promotes, floods a promotion
message over graph edges (not tree edges) to notify its former siblings, and
resumes the relay protocol as the new hub. Group regret is tracked throughout
so the recovery can be seen directly in the regret curve, and compared against
a no-failover baseline where the network never recovers after the hub dies.

Usage:
    python fault_tolerance.py --graph ba --N 20 --K 5 --T 5000 --t-fail 2000 --mode all
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(Path(__file__).resolve().parent / "merw.mplstyle")
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

def merw_eigenvector(G, tau=500, tol=1e-8, gossip_rounds=100):
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)

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
                    j = nbrs[np.random.randint(len(nbrs))]
                    r[i] = r[j] = (r[i] + r[j]) / 2

        w = w_new / np.exp(r)

        diff = np.max(np.abs(w - w_old) / (np.abs(w_old) + 1e-15))
        if diff < tol:
            break

    w = np.abs(w)
    w /= w.max()
    return w


def build_routing_tree(G, psi):
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


def _validate_tree(root, parent, children, depth, live_nodes):
    """
    Direct empirical check of the Failover correctness proposition: after
    promotion, the network must form a single valid spanning tree rooted at
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

def run_eigentree_ft(env, T, G, N, t_fail, c=2.0, sigma=1.0, tau_init=200,
                     enable_failover=True):
    """
    Runs the EigenTreeUCB relay cycle (fixed cycle length 2D+1, synchronized
    rounds) with permanent hub failures injected at every round in t_fail
    (an int for a single failure, or a list/tuple for several sequential
    failures).

    If enable_failover is True, the highest-psi direct child of the hub is
    the designated backup. Upon detecting hub silence (no downlink within the
    expected window), it self-promotes, floods a promotion message over graph
    edges to its former siblings (who re-route through it), and resumes the
    relay as the new hub, using its last-known mirrored state. Immediately
    after promotion, the new hub selects its OWN backup from its own
    children (the paper's recursive-backup rule), so the network can survive
    any number of sequential hub failures, one at a time.

    If enable_failover is False, the hub simply stays dead forever after the
    first failure in t_fail (no recovery mechanism) -- this isolates what
    the failover mechanism buys you.

    Returns:
        cum_regret: (N, T) array of per-agent cumulative regret
        events: dict with 'hub_history' (hub id per round), 'recovery_rounds'
                (list of rounds at which a new hub resumed normal cycling,
                one per successful failover), and 'tree_valid_checks' (list
                of the structural-validity check dict after each failover)
    """
    K = env.K
    psi = merw_eigenvector(G, tau=tau_init)
    hub0, parent, children, depth, D = build_routing_tree(G, psi)

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
    dead = np.zeros(N, dtype=bool)
    recovery_rounds = []
    failover_in_progress = False
    tree_valid_checks = []
    trees_snapshot = []  # (label, hub, parent, children) before/after each failover

    # Backup is announced once in the first downlink after the hub learns its
    # children (matches the paper). After each promotion the new hub picks
    # its own backup the same way (recursive backup selection).
    backup = max(current_children[hub0], key=lambda j: psi[j]) if current_children[hub0] else None

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

        # Detection: the backup only notices the hub is gone once an
        # expected downlink fails to arrive -- i.e. once a full cycle
        # (2*current_D + 1 rounds) has elapsed since the hub died without a
        # downlink reaching it. Before that, the backup (like everyone else)
        # is still silently waiting on the current cycle, so failover cannot
        # start immediately at the round of death.
        detected = (dead[current_hub] and death_time is not None
                    and t - death_time >= 2 * current_D + 1)

        if detected and enable_failover and not failover_in_progress and backup is not None and not dead[backup]:
            # --- Failover: backup detects silence and promotes itself ---
            failover_in_progress = True

            # (1) Self-promotion: backup becomes new hub, using its
            # mirrored (n_hat, s_hat) state -- last full downlink it received.
            new_hub = backup
            former_siblings = [j for j in current_children[current_hub] if j != backup]

            # (2) Flood promotion message over GRAPH edges (not tree edges)
            # to former siblings: each sibling re-routes via the actual
            # shortest path in G \ {old hub}, not a direct hop to the backup
            # (the paper's Failover correctness proposition routes each
            # sibling through whichever path actually connects it).
            G_minus_hub = G.copy()
            G_minus_hub.remove_node(current_hub)
            reroute_path = {}  # sibling -> shortest path (list of nodes) from new_hub
            for s in former_siblings:
                try:
                    reroute_path[s] = nx.shortest_path(G_minus_hub, new_hub, s)
                except nx.NetworkXNoPath:
                    # Would violate 2-connectivity; shouldn't happen on the
                    # graphs used here, but guard against it defensively.
                    reroute_path[s] = [new_hub, s]

            # (3) Rebuild the tree: new_hub at depth 0, its own subtree
            # shifts up by one. Each former sibling re-attaches one hop at a
            # time along its actual shortest path to new_hub; intermediate
            # nodes on that path become real relay hops in the new tree,
            # not a same-depth shortcut.
            new_parent = current_parent.copy()
            new_children = [list(c) for c in current_children]

            new_parent[new_hub] = -1
            new_children[current_hub].remove(new_hub)
            for s in former_siblings:
                path = reroute_path[s]  # [new_hub, ..., s]
                for u, v in zip(path[:-1], path[1:]):
                    if new_parent[v] != u:
                        # detach v from wherever it used to route and
                        # re-attach it one hop closer to new_hub
                        old_p = new_parent[v]
                        if old_p >= 0 and v in new_children[old_p]:
                            new_children[old_p].remove(v)
                        new_parent[v] = u
                        new_children[u].append(v)

            # Recompute depths from new_hub via BFS on the (now possibly
            # disconnected-from-old-hub) tree.
            new_depth = np.full(N, -1, dtype=int)
            new_depth[new_hub] = 0
            queue = [new_hub]
            while queue:
                node = queue.pop(0)
                for child in new_children[node]:
                    new_depth[child] = new_depth[node] + 1
                    queue.append(child)
            new_D = int(new_depth[new_depth >= 0].max())

            # Sibling subtrees are held with a Hold signal (no spurious
            # non-hub failovers) until the flood reaches them; we model this
            # as those nodes going silent until the flood (their own new
            # depth, one round per hop) reaches them, rather than
            # simulating the Hold packets individually.
            hold_until = np.zeros(N, dtype=int)
            for i in range(N):
                if i != new_hub and new_depth[i] >= 0:
                    hold_until[i] = t + int(new_depth[i])

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
            cycle_start = t + max(hold_until.max() - t, 0) + 1
            failover_in_progress = False
            recovery_rounds.append(cycle_start)

            # Recursive backup selection: the newly-promoted hub picks its
            # own highest-psi direct child as its own backup, so a later
            # failure of new_hub can be tolerated the same way.
            backup = (max(current_children[new_hub], key=lambda j: psi[j])
                     if current_children[new_hub] else None)

            for i in range(N):
                _silent(i, t)
            t += 1
            continue

        if dead[current_hub]:
            # No failover (baseline) or backup also dead: network stalls.
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
              "fail_schedule": fail_schedule, "final_backup": backup,
              "hub0": hub0, "D0": D, "tree_valid_checks": tree_valid_checks,
              "trees_snapshot": trees_snapshot}
    return cum_regret, events


# ============================================================
# Data save / load / plot
# ============================================================

def _csv_path(graph_type, N, K, T, tag):
    return DATA_DIR / f"fault_tolerance_{graph_type}_N{N}_K{K}_T{T}_{tag}.csv"


def compute_ft_data(graph_type, N, K, T, fail_every, sigma=1.0, c=2.0, seed=0, p=None,
                     fail_start=None, fail_at=None):
    """
    Runs one EigenTree-FT trajectory over horizon T with a hub failure
    injected either at the explicit rounds in `fail_at` (a list, takes
    precedence if given), or every `fail_every` rounds starting at
    `fail_start` (default `fail_every`; e.g. fail_every=1000 on T=5000
    injects failures at t=1000,2000,3000,4000; fail_start=2000 instead gives
    t=2000,3000,4000), and directly tests the Fault Tolerance section's
    claims:
      1. Failover correctness: after each promotion, the tree is checked to
         be a valid single spanning tree (not just plotted).
      2. The network keeps making progress (regret keeps growing, not
         flatlining) through every failure, using the recursive-backup rule
         so each newly-promoted hub has its own backup for the next failure.
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
                                     enable_failover=True)
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
    fig, ax = plt.subplots(figsize=(9, 5))
    ts = np.arange(1, len(data["group_ft"]) + 1)

    ax.plot(ts, data["group_ft"], label="EigenTree-FT", color="C2", linewidth=1.8)

    for i, tf in enumerate(data["fail_schedule"]):
        ax.axvline(tf, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(tf, ax.get_ylim()[1] * 0.02, "  fail", fontsize=7, rotation=90,
                va="bottom", ha="left")
    for rr in data["recovery_rounds"]:
        ax.axvline(rr, color="C2", linestyle=":", linewidth=1.0, alpha=0.6)

    n_fail = len(data["fail_schedule"])
    n_recovered = len(data["recovery_rounds"])
    ax.text(0.02, 0.98,
           f"{n_recovered}/{n_fail} failures recovered from",
           transform=ax.transAxes, fontsize=9, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    fail_str = ",".join(str(t) for t in data["fail_schedule"])
    ax.set_xlabel("Round $t$")
    ax.set_ylabel(r"Group cumulative regret $\sum_i R_i(t)$")
    ax.set_title(f"EigenTree-FT under repeated hub failure -- {graph_type.upper()} graph\n"
                 f"$N={N}$, $K={K}$, $T={T}$, hub fails at $t={fail_str}$")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FT_DIR / f"fault_tolerance_{graph_type}_N{N}_K{K}_T{T}_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def run_experiment(graph_type, N, K, T, fail_every, sigma, c, mode="all", seed=0, p=None,
                    fail_start=None, fail_at=None):
    tag = ("at" + "-".join(str(t) for t in fail_at)) if fail_at is not None else f"every{fail_every}"
    if mode in ("compute", "all"):
        data = compute_ft_data(graph_type, N, K, T, fail_every, sigma=sigma, c=c, seed=seed, p=p,
                                fail_start=fail_start, fail_at=fail_at)
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
    ap.add_argument("--mode", choices=("compute", "plot", "all"), default="all")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.graph, args.N, args.K, args.T, args.fail_every,
                   args.sigma, args.c, mode=args.mode, seed=args.seed, p=args.p,
                   fail_start=args.fail_start, fail_at=args.fail_at)
