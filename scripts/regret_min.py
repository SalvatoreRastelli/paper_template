"""
EigenTreeUCB: bidirectional relay cycles with hub-only UCB.

Algorithm overview
------------------
Phase 0 — Local init (rounds 0..K-1):
    Every agent pulls each arm once independently.  No communication.
    n_own_i^k = 1,  s_own_i^k = r_i^k.

Phase 1 — Repeated bidirectional cycles (rounds K .. commit):
    Each cycle lasts 2 * tree_depth rounds.

    Uplink (tree_depth rounds):
        Nodes relay in reverse-depth order (leaves first, one depth layer per round).
        Each activated node pulls its best arm (UCB on its current local estimate),
        then sends the delta (new observations only) toward the hub.
        Packets travel one hop per round; hub absorbs arrivals immediately.
        Hub and non-activated nodes are silent during the uplink.

    Hub pull (1 round):
        After all uplink packets are absorbed, the hub pulls UCB exactly once
        using the fully aggregated statistics.
        All non-hub nodes are silent this round.
        Commit check runs immediately after.

    Downlink (tree_depth rounds):
        Hub broadcasts its updated (n_hat[hub], s_hat[hub]) outward.
        A node at depth d receives the snapshot at downlink step d and replaces
        its local estimate with the hub's global view.
        All nodes are silent during the downlink.
        The next cycle's uplink then uses this enriched estimate.

Phase 2 — Broadcast + exploit:
    Hub broadcasts a* outward along the tree (one hop per round).
    Every agent that receives the signal pulls a* for all remaining rounds.

Regret accounting
-----------------
- Phase 0: K pulls per agent -> regret from suboptimal arms.
- Each cycle: each non-hub node pulls once (uplink); hub pulls once (post-aggregation).
- Phase 2: all agents pull a* -> zero regret.
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

RESULTS_DIR  = Path(__file__).resolve().parent.parent / "paper" / "results"
REGRET_DIR   = RESULTS_DIR / "Regret"
MERW_VIZ_DIR = RESULTS_DIR / "MERW_visualization"
REGRET_DIR.mkdir(parents=True, exist_ok=True)
MERW_VIZ_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Display name for the EigenTree-routed UCB algorithm. Change this single
# constant to rename it everywhere (CSV "algo" column, plot legend, CLI).
EIGENTREE_UCB_NAME = "EigenTreeUCB"


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
# Graph utilities
# ============================================================

def make_ba_graph(N, m=2, seed=None):
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


def make_cycle_graph(N):
    """Even cycle on N nodes."""
    if N % 2 != 0:
        N -= 1
    return nx.cycle_graph(N)


def make_lollipop_graph(N, seed=None):
    """Clique of size N//2 attached to a path of length N - N//2."""
    n_clique = N // 2
    n_path   = N - n_clique
    G = nx.complete_graph(n_clique)
    for i in range(n_path):
        G.add_node(n_clique + i)
        G.add_edge(n_clique + i - 1, n_clique + i)
    return G


def make_chain_graph(N, seed=None):
    """Path graph on N nodes."""
    return nx.path_graph(N)


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
    elif graph_type == "cycle":
        return make_cycle_graph(N)
    elif graph_type == "lollipop":
        return make_lollipop_graph(N, seed=seed)
    elif graph_type == "chain":
        return make_chain_graph(N, seed=seed)
    elif graph_type == "star":
        return make_star_graph(N, seed=seed)
    raise ValueError(f"Unknown graph type: {graph_type}")


# ============================================================
# MERW eigenvector
# ============================================================

def merw_eigenvector(G, tau=None, gossip_rounds=None):
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

    w = np.abs(w)
    lam = np.exp(np.mean(log_growth))
    w /= w.max()
    return w, lam


# ============================================================
# Routing tree
# ============================================================

def build_routing_tree(G, psi):
    """
    Builds a single spanning routing tree rooted at the global psi maximum.

    Step 1 — local gradient: each node points to its highest-psi neighbor
    (standard MERW routing). This may produce multiple local maxima.

    Step 2 — max-flooding: each node broadcasts its psi outward for
    diameter-many rounds. After round tau, m_i = max psi within distance tau.
    Local maxima then re-route toward the neighbor that carries the largest
    flooded value, chaining all local maxima toward the global maximum.
    Result: a single spanning tree with one hub.

    Returns:
        hub        : global psi maximum
        hubs       : [hub] (always a single element after flooding)
        parent     : parent[i] = routing target (-1 only for hub)
        children   : children[i] = list of nodes routing to i
        depth      : depth[i] = distance from hub
        tree_depth : max depth
        subtree_of : subtree_of[i] = hub for all i
    """
    N = G.number_of_nodes()

    # Step 1: local gradient routing
    parent = np.full(N, -1, dtype=int)
    for i in range(N):
        nbrs = list(G.neighbors(i))
        if not nbrs:
            continue
        best_j = max(nbrs, key=lambda j: psi[j])
        if psi[best_j] > psi[i]:
            parent[i] = best_j

    local_maxima = [i for i in range(N) if parent[i] < 0]

    # Step 2: max-flooding to merge all local maxima into a single tree.
    # Each node floods its psi value to neighbors one hop per round.
    # A node updates only when a neighbor carries a strictly better value.
    # When a round produces no update anywhere, the flood has converged:
    # every node holds the global maximum psi and via[i] points toward it.
    # No fixed round count or diameter knowledge is needed.
    m   = psi.copy()    # best psi value seen so far at each node
    via = np.arange(N)  # neighbor through which that best value arrived

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

    # Each local maximum re-routes toward the neighbor carrying a better
    # psi than its own. The one node whose psi is the global maximum never
    # sees a better neighbor value, so it keeps parent = -1 and becomes hub.
    for i in local_maxima:
        if m[via[i]] > psi[i]:
            parent[i] = via[i]

    remaining = [i for i in range(N) if parent[i] < 0]
    hub = remaining[0]

    # rebuild children from updated parent
    children = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            children[parent[i]].append(i)

    # BFS from hub to compute depths
    depth      = np.full(N, -1, dtype=int)
    subtree_of = np.full(N, hub, dtype=int)
    depth[hub] = 0
    queue = [hub]
    while queue:
        node = queue.pop(0)
        for c in children[node]:
            depth[c] = depth[node] + 1
            queue.append(c)

    tree_depth = int(depth.max())
    return hub, [hub], parent, children, depth, tree_depth, subtree_of


# ============================================================
# EigenTreeUCB
# ============================================================

def run_merw_ucb_relay(env, T, G, N, c=2.0, sigma=1.0, tau_init=None):
    """
    EigenTreeUCB: one independent relay cycle per local-maximum hub.

    Each hub h runs its own synchronized pipeline over its subtree:
      Uplink:   all non-hub nodes in subtree pull and relay deltas toward h.
      Hub pull: h aggregates, pulls UCB once, runs commit check.
      Downlink: h broadcasts (n_hat, s_hat, D, hops) to its subtree.

    Multiple hubs run in lockstep each round. Each hub commits independently;
    once all hubs have committed the run ends (every node exploits its hub's arm).
    """
    K = env.K
    psi, _ = merw_eigenvector(G, tau=tau_init)
    _, hubs, parent, children, depth, _, subtree_of = build_routing_tree(G, psi)

    # Lone hubs (no children) fall back to independent UCB - no relay needed
    lone_hubs  = {h for h in hubs if not children[h]}
    relay_hubs = [h for h in hubs if children[h]]

    # If every node is its own hub, run plain independent UCB
    if not relay_hubs:
        return run_ucb_ind(env, T, N, c=c, sigma=sigma), np.arange(1, T + 1, dtype=float) * N

    n_own  = np.ones((N, K))
    s_own  = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s_own[i, k] = env.pull(k)
    n_sent = np.zeros((N, K))
    s_sent = np.zeros((N, K))
    n_hat  = n_own.copy()
    s_hat  = s_own.copy()

    cum_regret      = np.zeros((N, T))
    pulls_per_round = np.zeros(T, dtype=int)
    t = 1  # index 0 stays 0 (no regret before first pull)

    if t >= T:
        return cum_regret, np.zeros(T, dtype=int)

    phase            = {h: 'uplink' for h in hubs}
    cycle_start_t    = {h: 1        for h in hubs}
    hub_last_t       = {h: 1        for h in hubs}
    learned_D        = {h: 0        for h in hubs}
    # initialized to 1 to account for the hub's own kick-off pull before the while loop
    hub_total_pulls  = {h: 1        for h in hubs}
    uplink_count     = {h: 0        for h in hubs}
    # total pulls broadcast by hub; each node uses this in its UCB index
    node_total_pulls = np.zeros(N, dtype=int)

    def _silent(i):
        cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0

    ul_inject = [[] for _ in range(N)]

    def _pull_and_inject(i):
        n_i = np.maximum(n_hat[i], 1e-9)
        tp  = max(node_total_pulls[i], 1)
        arm = int(np.argmax(s_hat[i] / n_i + sigma * np.sqrt(c * np.log(tp) / n_i)))
        r   = env.pull(arm)
        n_own[i, arm] += 1.0; s_own[i, arm] += r
        n_hat[i, arm] += 1.0; s_hat[i, arm] += r
        cum_regret[i, t] = (cum_regret[i, t-1] if t > 0 else 0.0) + env.gap(arm)
        pulls_per_round[t] += 1
        p = parent[i]
        if p >= 0:
            delta_n = n_own[i] - n_sent[i]
            delta_s = s_own[i] - s_sent[i]
            ul_inject[p].append({'n': delta_n.copy(), 's': delta_s.copy()})
            n_sent[i] = n_own[i].copy()
            s_sent[i] = s_own[i].copy()

    ul_buf     = [[] for _ in range(N)]
    dl_buf     = [None] * N
    wait_until = np.full(N, -1, dtype=int)

    # Cycle 0 kick-off: every node pulls (node_total_pulls is 0, log clamped to log(1)=0)
    for h in relay_hubs:
        _pull_and_inject(h)
    for h in lone_hubs:
        n_h = np.maximum(n_hat[h], 1e-9)
        arm = int(np.argmax(s_hat[h] / n_h + sigma * np.sqrt(c * np.log(max(t, 2)) / n_h)))
        r   = env.pull(arm)
        n_own[h, arm] += 1.0; s_own[h, arm] += r
        n_hat[h, arm] += 1.0; s_hat[h, arm] += r
        cum_regret[h, t] = (cum_regret[h, t-1] if t > 0 else 0.0) + env.gap(arm)
        pulls_per_round[t] += 1
    for i in range(N):
        if subtree_of[i] != i:
            _pull_and_inject(i)
    for i in range(N):
        ul_buf[i].extend(ul_inject[i]); ul_inject[i].clear()
    t += 1

    while t < T:

        ul_next = [[] for _ in range(N)]
        dl_next = [None] * N

        # --- Lone hubs: independent UCB each round ---
        for h in lone_hubs:
            n_h = np.maximum(n_hat[h], 1e-9)
            arm = int(np.argmax(s_hat[h] / n_h + sigma * np.sqrt(c * np.log(max(t, 2)) / n_h)))
            r   = env.pull(arm)
            n_own[h, arm] += 1.0; s_own[h, arm] += r
            n_hat[h, arm] += 1.0; s_hat[h, arm] += r
            cum_regret[h, t] = (cum_regret[h, t-1] if t > 0 else 0.0) + env.gap(arm)
            pulls_per_round[t] += 1

        # --- Relay hub state machines ---
        for h in relay_hubs:
            subtree_nodes = [i for i in range(N) if subtree_of[i] == h and i != h]

            if phase[h] == 'uplink':
                if ul_buf[h]:
                    for pkt in ul_buf[h]:
                        n_hat[h] += pkt['n']; s_hat[h] += pkt['s']
                    uplink_count[h] += len(ul_buf[h])
                    hub_last_t[h] = t

                uplink_clear = (
                    not ul_buf[h] and
                    not any(ul_buf[i] for i in subtree_nodes)
                )
                if uplink_clear:
                    learned_D[h] = hub_last_t[h] - cycle_start_t[h]
                    phase[h] = 'pull'
                    # uplink_count[h] messages = N-1 pulls (non-hub downlink/kick-off);
                    # +1 for hub pull; hub's own initial kick-off already in init value
                    hub_total_pulls[h] += uplink_count[h] + 1
                    uplink_count[h] = 0
                    node_total_pulls[h] = hub_total_pulls[h]
                    n_h = np.maximum(n_hat[h], 1e-9)
                    tp  = max(hub_total_pulls[h], 1)
                    arm = int(np.argmax(s_hat[h] / n_h + sigma * np.sqrt(c * np.log(tp) / n_h)))
                    r   = env.pull(arm)
                    n_hat[h, arm] += 1.0; s_hat[h, arm] += r
                    cum_regret[h, t] = (cum_regret[h, t-1] if t > 0 else 0.0) + env.gap(arm)
                    pulls_per_round[t] += 1
                    phase[h] = 'downlink'
                    snap = {'n': n_hat[h].copy(), 's': s_hat[h].copy(),
                            'D': learned_D[h], 'hops': 0,
                            'total_pulls': hub_total_pulls[h]}
                    for child in children[h]:
                        dl_next[child] = snap
                    for i in subtree_nodes:
                        _silent(i)
                else:
                    _silent(h)

            else:  # downlink
                _silent(h)

        # --- Non-hub nodes ---
        for i in range(N):
            h = subtree_of[i]
            if h == i:
                continue

            if phase[h] == 'uplink' and ul_buf[i]:
                p = parent[i]
                if p >= 0:
                    ul_next[p].extend(ul_buf[i])
                _silent(i)

            elif dl_buf[i] is not None:
                pkt  = dl_buf[i]
                n_hat[i] = pkt['n'].copy(); s_hat[i] = pkt['s'].copy()
                node_total_pulls[i] = pkt['total_pulls']
                hops = pkt['hops'] + 1
                D_pkt = pkt['D']
                wait  = D_pkt - hops
                wait_until[i] = t + wait
                child_pkt = {'n': pkt['n'], 's': pkt['s'], 'D': D_pkt, 'hops': hops,
                             'total_pulls': pkt['total_pulls']}
                for child in children[i]:
                    dl_next[child] = child_pkt
                if wait == 0:
                    _pull_and_inject(i); wait_until[i] = -1
                else:
                    _silent(i)

            elif wait_until[i] == t:
                _pull_and_inject(i); wait_until[i] = -1

            else:
                _silent(i)

        # Reset cycle when downlink is complete
        for h in relay_hubs:
            if phase[h] != 'downlink':
                continue
            subtree_nodes = [i for i in range(N) if subtree_of[i] == h and i != h]
            if (not any(wait_until[i] > t for i in subtree_nodes) and
                    not any(dl_next[i] is not None for i in subtree_nodes)):
                phase[h]         = 'uplink'
                cycle_start_t[h] = t

        for i in range(N):
            ul_next[i].extend(ul_inject[i]); ul_inject[i].clear()

        ul_buf = ul_next
        dl_buf = dl_next
        t += 1

    return cum_regret, np.cumsum(pulls_per_round)


def run_merw_ucb_relay_localonly(env, T, G, N, c=2.0, sigma=1.0, tau_init=None):
    """
    EigenTreeUCB (local-only variant): hub sends only D back, not (n_hat, s_hat).

    Same cycle structure as run_merw_ucb_relay, but the downlink carries only the
    timing signal D. Each non-hub node pulls using its own local estimate (n_own, s_own)
    only — no aggregated state ever flows down. The hub still aggregates all deltas
    and commits based on the global picture, but nodes never benefit from it for
    their own UCB decisions.

    This isolates whether the synchronization structure alone (vs. the enriched
    downlink estimate) is responsible for the performance gain.
    """
    K = env.K
    psi, _ = merw_eigenvector(G, tau=tau_init)
    hub, _, parent, children, depth, _, _ = build_routing_tree(G, psi)

    n_own = np.ones((N, K))
    s_own = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s_own[i, k] = env.pull(k)
    n_sent = np.zeros((N, K))
    s_sent = np.zeros((N, K))
    # Hub aggregated state (only hub uses this)
    n_hub = n_own[hub].copy()
    s_hub = s_own[hub].copy()

    cum_regret = np.zeros((N, T))
    pulls_per_round = np.zeros(T, dtype=int)
    t = 0

    if t >= T:
        return cum_regret, np.zeros(T, dtype=int)

    committed_arm = -1
    committed_t = T + 1

    def _silent(i):
        cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0

    ul_inject = [[] for _ in range(N)]

    def _pull_and_inject(i):
        # Pull using only own local estimate
        n_i = np.maximum(n_own[i], 1e-9)
        arm = int(np.argmax(s_own[i] / n_i + sigma * np.sqrt(c * np.log(max(t, 2)) / n_i)))
        r = env.pull(arm)
        n_own[i, arm] += 1.0
        s_own[i, arm] += r
        cum_regret[i, t] = (cum_regret[i, t - 1] if t > 0 else 0.0) + env.gap(arm)
        pulls_per_round[t] += 1
        p = parent[i]
        if p >= 0:
            delta_n = n_own[i] - n_sent[i]
            delta_s = s_own[i] - s_sent[i]
            ul_inject[p].append({'n': delta_n.copy(), 's': delta_s.copy()})
            n_sent[i] = n_own[i].copy()
            s_sent[i] = s_own[i].copy()

    ul_buf = [[] for _ in range(N)]
    dl_buf = [None] * N
    wait_until = np.full(N, -1, dtype=int)

    phase = 'uplink'
    cycle_start_t = 0
    hub_last_t = 0
    learned_D = 0

    # Cycle 0 kick-off
    _silent(hub)
    for i in range(N):
        if i != hub:
            _pull_and_inject(i)
    for i in range(N):
        ul_buf[i].extend(ul_inject[i])
        ul_inject[i].clear()
    t += 1

    while t < T and committed_arm < 0:

        ul_next = [[] for _ in range(N)]
        dl_next = [None] * N

        # --- Hub ---
        if phase == 'uplink':
            if ul_buf[hub]:
                for pkt in ul_buf[hub]:
                    n_hub += pkt['n']
                    s_hub += pkt['s']
                hub_last_t = t

            if not any(len(ul_buf[i]) > 0 for i in range(N) if i != hub) and not ul_buf[hub]:
                learned_D = hub_last_t - cycle_start_t
                phase = 'pull'
                # Hub pulls on its own aggregated state
                log_t = np.log(max(t, 2))
                n_h = np.maximum(n_hub, 1e-9)
                arm = int(np.argmax(s_hub / n_h + sigma * np.sqrt(c * log_t / n_h)))
                r = env.pull(arm)
                n_hub[arm] += 1.0
                s_hub[arm] += r
                # Hub's own local accumulators also update
                n_own[hub, arm] += 1.0
                s_own[hub, arm] += r
                cum_regret[hub, t] = (cum_regret[hub, t - 1] if t > 0 else 0.0) + env.gap(arm)
                pulls_per_round[t] += 1

                n_h = np.maximum(n_hub, 1e-9)
                mu_h = s_hub / n_h
                cb = sigma * np.sqrt(c * log_t / n_h)
                lcb = mu_h - cb
                ucb_v = mu_h + cb
                best = int(np.argmax(lcb))
                if K == 1 or np.all(lcb[best] > ucb_v[np.arange(K) != best]):
                    committed_arm = best
                    committed_t = t
                    for i in range(N):
                        if i != hub:
                            _silent(i)
                    t += 1
                    break

                # Downlink: send only D (no aggregated statistics)
                phase = 'downlink'
                snap = {'D': learned_D, 'hops': 0}
                for child in children[hub]:
                    dl_next[child] = snap
                for i in range(N):
                    if i != hub:
                        _silent(i)
            else:
                _silent(hub)

        else:
            _silent(hub)

        # --- Non-hub nodes ---
        for i in range(N):
            if i == hub:
                continue

            if phase == 'uplink' and ul_buf[i]:
                p = parent[i]
                if p >= 0:
                    ul_next[p].extend(ul_buf[i])
                _silent(i)

            elif dl_buf[i] is not None:
                # Timing signal only — no estimate update
                pkt = dl_buf[i]
                hops = pkt['hops'] + 1
                D_pkt = pkt['D']
                wait = D_pkt - hops
                wait_until[i] = t + wait
                child_pkt = {'D': D_pkt, 'hops': hops}
                for child in children[i]:
                    dl_next[child] = child_pkt
                if wait == 0:
                    _pull_and_inject(i)
                    wait_until[i] = -1
                else:
                    _silent(i)

            elif wait_until[i] == t:
                _pull_and_inject(i)
                wait_until[i] = -1

            else:
                _silent(i)

        if phase == 'downlink' and not any(
            wait_until[i] > t for i in range(N) if i != hub
        ) and not any(dl_next[i] is not None for i in range(N) if i != hub):
            phase = 'uplink'
            cycle_start_t = t
            # hub_last_t intentionally NOT reset: updated when packets arrive

        for i in range(N):
            ul_next[i].extend(ul_inject[i])
            ul_inject[i].clear()

        ul_buf = ul_next
        dl_buf = dl_next
        t += 1

    if committed_arm < 0:
        committed_arm = int(np.argmax(s_hub / np.maximum(n_hub, 1e-9)))

    committed_at = np.array([committed_t + int(depth[i]) for i in range(N)])
    while t < T:
        for i in range(N):
            if committed_at[i] <= t:
                env.pull(committed_arm)
                cum_regret[i, t] = (cum_regret[i, t - 1] if t > 0 else 0.0) + env.gap(committed_arm)
                pulls_per_round[t] += 1
            else:
                cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0
        t += 1

    return cum_regret, np.cumsum(pulls_per_round)


# ============================================================
# Decentralized UCB1 (Zhu et al. 2024, Algorithm 2 / Theorem 2)
# ============================================================

def run_dpe2(env, T, G, N, c=2.0, sigma=1.0):
    """
    DPE2 from Wang et al. (2020), Section 3.

    Leader election: flood minimum node-id for D+1 rounds; node whose original
    id equals the global min becomes the leader.

    Exploration-exploitation phase:
      - Leader maintains D(t): arms whose KL-UCB index exceeds the best arm's.
        When D(t) is empty: leader exploits best empirical arm ell_1.
        When D(t) nonempty: leader opens exploration block, cycles through D(t).
      - Followers always play ell_1 (the best arm last announced by leader).
      - Leader communicates ell_1 to neighbours whenever it changes; followers
        forward it one hop per round (broadcast over the tree takes D rounds).

    KL-UCB index for arm k at the leader (Gaussian approximation):
      d_k(t) = mu_hat_k + sigma * sqrt(2 * log(t) / n_k)
    """
    K = env.K

    # --- Leader election: flood min random id for diameter+1 rounds ---
    diameter = nx.diameter(G)
    node_ids = np.random.permutation(N).astype(int)  # random unique ids
    min_seen = node_ids.copy()
    for _ in range(diameter + 1):
        new_min = min_seen.copy()
        for i in range(N):
            for j in G.neighbors(i):
                if min_seen[j] < new_min[i]:
                    new_min[i] = min_seen[j]
        min_seen = new_min
    leader = int(np.argmin(min_seen))  # node whose original id was the global min

    # Build BFS tree from leader for follower broadcast
    bfs_tree = nx.bfs_tree(G, leader)
    depth = nx.single_source_shortest_path_length(G, leader)

    # --- Initialization: all nodes pull each arm once (paper's orthogonalization phase) ---
    n_lead = np.zeros(K)
    s_lead = np.zeros(K)
    cum_regret = np.zeros((N, T))

    # Init regret accumulated before round 1 is not counted (pre-T rounds)
    for k in range(K):
        r = env.pull(k)
        n_lead[k] = 1.0
        s_lead[k] = r
    # Followers also pull each arm once (regret not counted — initialization phase)
    for i in range(N):
        if i == leader:
            continue
        for k in range(K):
            env.pull(k)

    ell1 = int(np.argmax(s_lead / n_lead))
    follower_arm = np.full(N, ell1, dtype=int)
    pending = {}   # {node: (arm, arrive_round)}

    def _klucb(mu, n, t, sigma):
        return mu + sigma * np.sqrt(2.0 * np.log(max(t, 2)) / max(n, 1e-9))

    for t in range(1, T + 1):

        # --- Deliver pending broadcasts ---
        delivered = [i for i, (arm, arr) in pending.items() if arr <= t]
        for i in delivered:
            follower_arm[i] = pending[i][0]
            del pending[i]

        # --- Leader: recompute D(t) every round ---
        mu_hat = s_lead / np.maximum(n_lead, 1e-9)
        klucb_vals = np.array([_klucb(mu_hat[k], n_lead[k], t, sigma) for k in range(K)])
        ell1_old = ell1
        ell1 = int(np.argmax(mu_hat))

        # D(t): arms whose KL-UCB index >= empirical mean of best arm (paper Algorithm 2)
        D_t = [k for k in range(K) if k != ell1 and klucb_vals[k] >= mu_hat[ell1]]
        leader_arm = D_t[0] if D_t else ell1

        r_lead = env.pull(leader_arm)
        n_lead[leader_arm] += 1.0
        s_lead[leader_arm] += r_lead
        cum_regret[leader, t - 1] = (cum_regret[leader, t - 2] if t > 1 else 0.0) + env.gap(leader_arm)

        # Broadcast new ell1 to followers if it changed
        mu_hat = s_lead / np.maximum(n_lead, 1e-9)
        new_ell1 = int(np.argmax(mu_hat))
        if new_ell1 != ell1_old:
            ell1 = new_ell1
            for i in range(N):
                if i != leader:
                    arrive = t + depth[i]
                    if arrive <= T:
                        pending[i] = (ell1, arrive)

        # --- Followers pull their current announced best arm ---
        for i in range(N):
            if i == leader:
                continue
            arm = follower_arm[i]
            env.pull(arm)
            cum_regret[i, t - 1] = (cum_regret[i, t - 2] if t > 1 else 0.0) + env.gap(arm)

    return cum_regret


# ============================================================
# Coop-UCB2 and UCB-Ind (copied from experiment.py for comparison)
# ============================================================

def run_ucb_ind(env, T, N, c=2.0, sigma=1.0):
    K = env.K
    n = np.ones((N, K))
    s = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s[i, k] = env.pull(k)
    cum_regret = np.zeros((N, T))
    for t in range(1, T):
        for i in range(N):
            mu_hat = s[i] / n[i]
            arm = int(np.argmax(mu_hat + np.sqrt(c * np.log(t) / n[i])))
            r = env.pull(arm)
            n[i, arm] += 1
            s[i, arm] += r
            cum_regret[i, t] = cum_regret[i, t - 1] + env.gap(arm)
    return cum_regret


def run_coop_ucb2(env, T, G, N, sigma=1.0, gamma=2.0, eta=0.5):
    K = env.K
    d_max = max(dict(G.degree()).values())
    L = np.array(nx.laplacian_matrix(G).todense(), dtype=float)
    kappa = d_max / (d_max - 1) if d_max > 1 else 1.0
    P = np.eye(N) - (kappa / d_max) * L
    G_eta = 1.0 - eta ** 2 / 16.0
    n_hat = np.zeros((N, K))
    s_hat = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            r = env.pull(k)
            n_hat[i, k] = 1.0
            s_hat[i, k] = r
    cum_regret = np.zeros((N, T))
    for t in range(1, T):
        n_hat = P @ n_hat
        s_hat = P @ s_hat
        arms = np.zeros(N, dtype=int)
        f_prev = np.sqrt(np.log(max(t, 2)))
        ln_t = np.log(max(t, 2))
        for i in range(N):
            n_i = np.maximum(n_hat[i], 1e-9)
            mu_hat = s_hat[i] / n_i
            bonus = sigma * np.sqrt((2.0 * gamma / G_eta) * ((n_i + f_prev) / (N * n_i)) * (ln_t / n_i))
            arms[i] = int(np.argmax(mu_hat + bonus))
        for i in range(N):
            r = env.pull(arms[i])
            n_hat[i, arms[i]] += 1.0
            s_hat[i, arms[i]] += r
            cum_regret[i, t] = cum_regret[i, t - 1] + env.gap(arms[i])
    return cum_regret


def run_central_ucb(env, T, N, c=2.0, sigma=1.0, hub=0):
    """Central-UCB: EigenTreeUCB with a star topology (D=1).

    Every non-hub node is directly connected to the hub (one hop).
    Cycle structure mirrors EigenTreeUCB with D=1:
      Uplink:   all non-hub nodes pull, send delta to hub (1 hop, instant).
      Hub pull: hub aggregates all deltas, pulls UCB once, commit check.
      Downlink: hub broadcasts (n_hat, s_hat); non-hub nodes receive and
                immediately pull (wait = D - hops = 1 - 1 = 0).

    Kick-off: all N nodes pull once, non-hub deltas go to hub.
    """
    K       = env.K
    non_hub = [i for i in range(N) if i != hub]

    # init: each node pulls each arm once (not counted in regret)
    n_own = np.ones((N, K))
    s_own = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s_own[i, k] = env.pull(k)
    n_hat  = n_own.copy()
    s_hat  = s_own.copy()
    n_sent = np.zeros((N, K))
    s_sent = np.zeros((N, K))

    cum_regret      = np.zeros((N, T))
    pulls_per_round = np.zeros(T, dtype=int)
    t = 1
    total_pulls = 0  # init pulls not counted

    def _pull_node(i, tp):
        n_i = np.maximum(n_hat[i], 1e-9)
        arm = int(np.argmax(s_hat[i] / n_i + sigma * np.sqrt(c * np.log(max(tp, 1)) / n_i)))
        r   = env.pull(arm)
        n_own[i, arm] += 1; s_own[i, arm] += r
        n_hat[i, arm] += 1; s_hat[i, arm] += r
        cum_regret[i, t] = cum_regret[i, t - 1] + env.gap(arm)
        pulls_per_round[t] += 1

    def _send_deltas():
        for i in non_hub:
            n_hat[hub] += n_own[i] - n_sent[i]
            s_hat[hub] += s_own[i] - s_sent[i]
            n_sent[i] = n_own[i].copy()
            s_sent[i] = s_own[i].copy()

    # kick-off: all N nodes pull, non-hub send deltas to hub
    total_pulls += N
    for i in range(N):
        _pull_node(i, total_pulls)
    _send_deltas()
    t += 1

    while t < T:
        # hub pull: hub now has all N kick-off deltas aggregated
        total_pulls += 1
        for i in range(N):
            cum_regret[i, t] = cum_regret[i, t - 1]
        _pull_node(hub, total_pulls)
        t += 1
        if t >= T:
            break

        # downlink + non-hub pull
        for i in non_hub:
            n_hat[i] = n_hat[hub].copy()
            s_hat[i] = s_hat[hub].copy()
        total_pulls += N
        cum_regret[hub, t] = cum_regret[hub, t - 1]
        for i in non_hub:
            _pull_node(i, total_pulls)
        _send_deltas()
        t += 1

    return cum_regret, np.cumsum(pulls_per_round)


# ============================================================
# Parallel worker
# ============================================================

def _worker(task):
    algo_name, run_seed, graph_seed, graph_type, N, _K, means, sigma, T, c, p = task
    np.random.seed(run_seed)
    G = make_graph(graph_type, N, seed=graph_seed, p=p)
    env = BanditEnv(means, sigma=sigma)

    psi, _ = merw_eigenvector(G)
    _, hubs, _, _, _, _, _ = build_routing_tree(G, psi)
    hub = hubs[0]

    if algo_name == "Coop-UCB2":
        cr = run_coop_ucb2(env, T, G, N, sigma=sigma)
        cum_pulls = np.arange(1, T + 1, dtype=float) * N
    elif algo_name == EIGENTREE_UCB_NAME:
        cr, cum_pulls = run_merw_ucb_relay(env, T, G, N, c=c, sigma=sigma)
    elif algo_name == "Central-UCB":
        cr, cum_pulls = run_central_ucb(env, T, N, c=c, sigma=sigma, hub=hub)
    else:
        raise ValueError(algo_name)

    return algo_name, cr.sum(axis=0), cr[hub], cum_pulls


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
    by_label = {name: ([], [], []) for name in algo_names}
    for name, group, hub_r, cum_pulls in results_list:
        by_label[name][0].append(group)
        by_label[name][1].append(hub_r)
        by_label[name][2].append(cum_pulls)
    return {name: (np.array(g).mean(axis=0), np.array(g).std(axis=0),
                   np.array(h).mean(axis=0), np.array(h).std(axis=0),
                   np.array(p).mean(axis=0))
            for name, (g, h, p) in by_label.items()}


# ============================================================
# Experiment
# ============================================================

ALGO_NAMES = ["Central-UCB", "Coop-UCB2", EIGENTREE_UCB_NAME]

STYLES = {
    "Central-UCB":      ("C0", ":",  "Central-UCB"),
    "Coop-UCB2":        ("C1", "--", "Coop-UCB2"),
    EIGENTREE_UCB_NAME: ("C2", "-",  EIGENTREE_UCB_NAME),
}


def _csv_path(graph_type, N, K, T):
    return DATA_DIR / f"relay_regret_{graph_type}_N{N}_K{K}_T{T}.csv"


def compute_regret_data(n_runs, T, N, K, graph_type, sigma, c, n_workers, seed=0, p=None):
    """Run the Monte Carlo experiment. Returns {algo: (mean_g, std_g, mean_h, std_h, mean_cp)}."""
    rng = np.random.RandomState(seed)
    means = np.linspace(0.0, 1.0, K)[::-1]

    graph_seed = int(rng.randint(0, 2**31))  # fixed across all runs
    tasks = []
    for _ in range(n_runs):
        run_seed = int(rng.randint(0, 2**31))
        for name in ALGO_NAMES:
            tasks.append((name, run_seed, graph_seed, graph_type,
                          N, K, means, sigma, T, c, p))

    return _run_parallel(tasks, ALGO_NAMES, n_workers, tag="relay")


def save_regret_csv(results, T, N, K, graph_type):
    """Write one row per (algo, t) to a long-format CSV."""
    out = _csv_path(graph_type, N, K, T)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "t", "mean_group", "std_group",
                          "mean_hub", "std_hub", "cum_pulls"])
        for name in ALGO_NAMES:
            if name not in results:
                continue
            mean_g, std_g, mean_h, std_h, mean_cp = results[name]
            for t in range(T):
                writer.writerow([name, t + 1, mean_g[t], std_g[t],
                                  mean_h[t], std_h[t], mean_cp[t]])
    print(f"  Saved {out}")
    return out


def load_regret_csv(graph_type, N, K, T):
    """Read back the long-format CSV into {algo: (mean_g, std_g, mean_h, std_h, mean_cp)}."""
    path = _csv_path(graph_type, N, K, T)
    rows_by_algo = {name: [] for name in ALGO_NAMES}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_algo[row["algo"]].append(row)
    results = {}
    for name, rows in rows_by_algo.items():
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["t"]))
        results[name] = (
            np.array([float(r["mean_group"]) for r in rows]),
            np.array([float(r["std_group"]) for r in rows]),
            np.array([float(r["mean_hub"]) for r in rows]),
            np.array([float(r["std_hub"]) for r in rows]),
            np.array([float(r["cum_pulls"]) for r in rows]),
        )
    return results


def plot_regret(results, T, N, K, graph_type, n_runs):
    """Render the group regret and regret-vs-pulls figure from precomputed results."""
    ts = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 4.1))

    max_common_pulls = min(results[n][4][-1] for n in ALGO_NAMES if n in results)
    pull_axis = np.linspace(0, max_common_pulls, 500)

    handles, labels = [], []
    for name, (color, ls, label) in STYLES.items():
        if name not in results:
            continue
        mean_g, std_g, mean_h, std_h, mean_cp = results[name]
        line, = axes[0].plot(ts, mean_g, label=label, color=color, linestyle=ls, linewidth=2.2)
        axes[0].fill_between(ts, mean_g - std_g, mean_g + std_g, color=color, alpha=0.15)
        reg_vs_pulls = np.interp(pull_axis, mean_cp, mean_g)
        std_vs_pulls = np.interp(pull_axis, mean_cp, std_g)
        axes[1].plot(pull_axis, reg_vs_pulls, label=label, color=color, linestyle=ls, linewidth=2.2)
        axes[1].fill_between(pull_axis, reg_vs_pulls - std_vs_pulls,
                             reg_vs_pulls + std_vs_pulls, color=color, alpha=0.15)
        handles.append(line)
        labels.append(label)

    axes[0].set_xlabel("Round $t$")
    axes[0].set_ylabel("$\\sum_i R_i(t)$")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Total arm pulls (all agents combined)")
    axes[1].set_ylabel("$\\sum_i R_i(t)$")
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(45)

    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=[0.02, 0.09, 0.98, 1])
    out1 = REGRET_DIR / f"relay_regret_{graph_type}_N{N}_K{K}_T{T}.pdf"
    fig.savefig(out1)
    plt.close(fig)
    print(f"  Saved {out1}")


def run_experiment(n_runs, T, N, K, graph_type, sigma, c, n_workers, mode="all", seed=0, p=None):
    if mode in ("compute", "all"):
        results = compute_regret_data(n_runs, T, N, K, graph_type, sigma, c, n_workers, seed=seed, p=p)
        save_regret_csv(results, T, N, K, graph_type)
    if mode == "compute":
        return
    if mode == "plot":
        results = load_regret_csv(graph_type, N, K, T)
    plot_regret(results, T, N, K, graph_type, n_runs)



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
    p.add_argument("--n-runs",    type=int,   default=100)
    p.add_argument("--T",         type=int,   default=5_000)
    p.add_argument("--N",         type=int,   default=20)
    p.add_argument("--K",         type=int,   default=20)
    p.add_argument("--graph",     choices=("ba", "er", "barbell", "grid", "cycle", "lollipop", "chain", "star"), default="ba")
    p.add_argument("--p-er",      type=float, default=None,
                   help="ER edge probability override (default: 2.5*ln(N)/N; ignored for other graph types)")
    p.add_argument("--sigma",     type=float, default=1.0)
    p.add_argument("--c",         type=float, default=2.0)
    p.add_argument("--n-workers", type=int,   default=None)
    p.add_argument("--mode", choices=("compute", "plot", "all"), default="all",
                   help="compute: run the experiment and save data/*.csv only; "
                        "plot: render the figure from an existing CSV; "
                        "all: compute then plot (default)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_workers = resolve_n_workers(args.n_workers)
    print(f"[config] graph={args.graph}, N={args.N}, K={args.K}, T={args.T}, "
          f"sigma={args.sigma}, c={args.c}, runs={args.n_runs}, workers={n_workers}, "
          f"mode={args.mode}, p_er={args.p_er}")
    start = time.time()
    run_experiment(args.n_runs, args.T, args.N, args.K,
                   args.graph, args.sigma, args.c, n_workers, mode=args.mode, p=args.p_er)
    print(f"\nTotal runtime: {time.time() - start:.2f}s")
