"""
Numerical experiments for EigenTreeUCB.

Compares on a Barabasi-Albert graph (heterogeneous topology, clear hub structure):
  - UCB-Ind          : independent UCB per agent, no communication
  - Coop-UCB2        : doubly-stochastic averaging (Landgren et al., 2021)
  - EigenTreeUCB    : directed consensus + absorbed-mass boost + commit-and-broadcast
  - Hillel           : distributed successive elimination (Hillel et al., NeurIPS 2013)

Rewards are drawn from Normal(mu_k, sigma^2) distributions (not Bernoulli losses).
Regret = sum_i sum_t E[mu_1 - mu_{a_i(t)}]  (reward-based, higher-is-better arms).

Two experiment types:
  1. Regret minimization: all algorithms tracked over T rounds.
  2. BAI (Best Arm Identification): Hillel-BAI vs EigenTree-BAI, fixed confidence,
     measuring total arm pulls to identify the best arm.

Usage:
  python experiment.py
  python experiment.py --graph ba --T 5000 --N 20 --K 5 --n-runs 50
  python experiment.py --n-workers 8

Output:
  results/merw_ucb_regret_*.pdf
  results/merw_ucb_bai_*.pdf
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
import numpy as np
import networkx as nx
warnings.filterwarnings("ignore", category=FutureWarning, module="networkx")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "paper" / "results"
BAI_DIR     = RESULTS_DIR / "BAI"
REGRET_DIR  = RESULTS_DIR / "Regret"
BAI_DIR.mkdir(parents=True, exist_ok=True)
REGRET_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Display names for the EigenTree-routed algorithms. Change these two
# constants to rename them everywhere (CSV "algo" column, plot legend, CLI).
EIGENTREE_UCB_NAME = "EigenTreeUCB"
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


def make_graph(graph_type, N, seed=None, p=None):
    if graph_type == "ba":
        return make_ba_graph(N, m=2, seed=seed)
    elif graph_type == "er":
        return make_er_graph(N, seed=seed, p=p)
    raise ValueError(f"Unknown graph type: {graph_type}")


# ============================================================
# MERW eigenvector (power iteration on adjacency matrix)
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

def run_ucb_ind(env, T, N, c=2.0, sigma=1.0):
    """
    Each agent runs UCB1 independently.
    UCB index: mu_hat_k + sqrt(c * log(t) / n_k)
    Returns cum_regret of shape (N, T).
    """
    K = env.K
    n = np.ones((N, K))       # init: each arm pulled once
    s = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s[i, k] = env.pull(k)

    cum_regret = np.zeros((N, T))
    for t in range(1, T + 1):
        for i in range(N):
            mu_hat = s[i] / n[i]
            ucb = mu_hat + np.sqrt(c * np.log(t) / n[i])
            arm = int(np.argmax(ucb))
            r = env.pull(arm)
            n[i, arm] += 1
            s[i, arm] += r
            cum_regret[i, t - 1] = (cum_regret[i, t - 2] if t > 1 else 0.0) + env.gap(arm)
    return cum_regret


# ============================================================
# Algorithm: Coop-UCB2 (Landgren et al., 2021)
# ============================================================

def run_coop_ucb2(env, T, G, N, sigma=1.0, gamma=2.0, eta=0.5):
    """
    Coop-UCB2 exactly as in Landgren et al. (2021), eqs. (7)-(10) and (15).

    Consensus matrix: P = I - (kappa/d_max) * L   (row-stochastic)
      with kappa = d_max/(d_max-1), so the self-weight is 1 - kappa*deg(i)/d_max.

    UCB index at time t (using accumulators from round t-1):
      Q_i^k(t-1) = mu_hat_i^k(t-1) + C_i^k(t-1)
      C_i^k(t-1) = sigma * sqrt( 2*gamma/G(eta)
                                 * (n_hat_i^k(t-1) + f(t-1)) / (M * n_hat_i^k(t-1))
                                 * ln(t-1) / n_hat_i^k(t-1) )
      with f(t) = sqrt(ln t),  G(eta) = 1 - eta^2/16,  M = N (number of agents).

    Round structure (matches the paper):
      1. Apply consensus to get n_hat(t-1), s_hat(t-1)  [eqs 7-8, step t]
      2. Each agent selects arm = argmax Q_i^k(t-1)
      3. Observe reward; add local observation to accumulators
    """
    K = env.K
    d_max = max(dict(G.degree()).values())
    L = np.array(nx.laplacian_matrix(G).todense(), dtype=float)
    kappa = d_max / (d_max - 1) if d_max > 1 else 1.0
    P = np.eye(N) - (kappa / d_max) * L   # row-stochastic

    G_eta = 1.0 - eta ** 2 / 16.0         # G(eta) from the paper

    # Initialisation: each agent pulls every arm once (round-robin, arm k by agent k mod N)
    n_hat = np.zeros((N, K))
    s_hat = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            r = env.pull(k)
            n_hat[i, k] = 1.0
            s_hat[i, k] = r

    cum_regret = np.zeros((N, T))
    for t in range(1, T + 1):
        # Step 1: consensus on previous accumulators
        n_hat = P @ n_hat
        s_hat = P @ s_hat

        # Step 2: arm selection using updated estimates
        arms = np.zeros(N, dtype=int)
        f_prev = np.sqrt(np.log(max(t, 2)))          # f(t) = sqrt(ln t)
        ln_t = np.log(max(t, 2))
        for i in range(N):
            n_i = np.maximum(n_hat[i], 1e-9)
            mu_hat = s_hat[i] / n_i
            bonus = sigma * np.sqrt(
                (2.0 * gamma / G_eta)
                * ((n_i + f_prev) / (N * n_i))
                * (ln_t / n_i)
            )
            arms[i] = int(np.argmax(mu_hat + bonus))

        # Step 3: pull and record
        for i in range(N):
            r = env.pull(arms[i])
            n_hat[i, arms[i]] += 1.0
            s_hat[i, arms[i]] += r
            cum_regret[i, t - 1] = (cum_regret[i, t - 2] if t > 1 else 0.0) + env.gap(arms[i])

    return cum_regret


# ============================================================
# Algorithm: EigenTreeUCB (localonly relay -- hub sends only D, not state)
# ============================================================

def run_merw_ucb(env, T, G, N, c=2.0, sigma=1.0, tau_init=None):
    """
    EigenTreeUCB with hub-only commit and D-only downlink.

    Same relay cycle structure as MaxENtUCB (new_vers), but the downlink
    carries only the timing signal D, not (n_hat, s_hat). Each non-hub node
    pulls using its own local estimate only. The hub aggregates all uplink
    deltas and runs the commit check on the global picture.
    """
    K = env.K
    psi, _ = merw_eigenvector(G, tau=tau_init)
    hub, parent, children, depth, _ = build_routing_tree(G, psi)

    n_own = np.ones((N, K))
    s_own = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            s_own[i, k] = env.pull(k)
    n_sent = np.zeros((N, K))
    s_sent = np.zeros((N, K))
    n_hub  = n_own[hub].copy()
    s_hub  = s_own[hub].copy()

    cum_regret = np.zeros((N, T))
    t = 0
    total_pulls = N * K  # init pulls

    committed_arm = -1
    committed_t   = T + 1

    def _silent(i):
        cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0

    ul_inject = [[] for _ in range(N)]

    def _pull_and_inject(i):
        nonlocal total_pulls
        n_i = np.maximum(n_own[i], 1e-9)
        own_pulls = max(n_own[i].sum(), 1)
        arm = int(np.argmax(s_own[i] / n_i + sigma * np.sqrt(c * np.log(own_pulls) / n_i)))
        r   = env.pull(arm)
        total_pulls += 1
        n_own[i, arm] += 1.0
        s_own[i, arm] += r
        cum_regret[i, t] = (cum_regret[i, t - 1] if t > 0 else 0.0) + env.gap(arm)
        p = parent[i]
        if p >= 0:
            delta_n = n_own[i] - n_sent[i]
            delta_s = s_own[i] - s_sent[i]
            ul_inject[p].append({'n': delta_n.copy(), 's': delta_s.copy()})
            n_sent[i] = n_own[i].copy()
            s_sent[i] = s_own[i].copy()

    ul_buf    = [[] for _ in range(N)]
    dl_buf    = [None] * N
    wait_until = np.full(N, -1, dtype=int)

    phase         = 'uplink'
    cycle_start_t = 0
    hub_last_t    = 0
    learned_D     = 0

    # Cycle 0 kick-off: all non-hub nodes pull and send deltas
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

        if phase == 'uplink':
            if ul_buf[hub]:
                for pkt in ul_buf[hub]:
                    n_hub += pkt['n']
                    s_hub += pkt['s']
                hub_last_t = t

            uplink_clear = (
                not ul_buf[hub] and
                not any(ul_buf[i] for i in range(N) if i != hub)
            )
            if uplink_clear:
                learned_D = hub_last_t - cycle_start_t
                log_P = np.log(max(total_pulls, 2))
                n_h   = np.maximum(n_hub, 1e-9)
                mu_h  = s_hub / n_h
                cb    = sigma * np.sqrt(c * log_P / n_h)
                lcb   = mu_h - cb
                ucb_v = mu_h + cb

                # LUCB: pull empirical best + challenger (highest UCB among the rest)
                best = int(np.argmax(lcb))
                others = [k for k in range(K) if k != best]
                challenger = int(others[np.argmax(ucb_v[others])]) if others else best

                for arm in ([best, challenger] if challenger != best else [best]):
                    r = env.pull(arm)
                    total_pulls += 1
                    n_hub[arm] += 1.0; s_hub[arm] += r
                    n_own[hub, arm] += 1.0; s_own[hub, arm] += r
                    cum_regret[hub, t] = (cum_regret[hub, t - 1] if t > 0 else 0.0) + env.gap(arm)

                # recompute after pulls
                n_h   = np.maximum(n_hub, 1e-9)
                mu_h  = s_hub / n_h
                cb    = sigma * np.sqrt(c * log_P / n_h)
                lcb   = mu_h - cb
                ucb_v = mu_h + cb
                best  = int(np.argmax(lcb))
                if K == 1 or np.all(lcb[best] > ucb_v[np.arange(K) != best]):
                    committed_arm = best
                    committed_t   = t
                    for i in range(N):
                        if i != hub:
                            _silent(i)
                    t += 1
                    break

                phase = 'downlink'
                snap  = {'D': learned_D, 'hops': 0}
                for child in children[hub]:
                    dl_next[child] = snap
                for i in range(N):
                    if i != hub:
                        _silent(i)
            else:
                _silent(hub)

        else:
            _silent(hub)

        for i in range(N):
            if i == hub:
                continue
            if phase == 'uplink' and ul_buf[i]:
                p = parent[i]
                if p >= 0:
                    ul_next[p].extend(ul_buf[i])
                _silent(i)
            elif dl_buf[i] is not None:
                pkt   = dl_buf[i]
                hops  = pkt['hops'] + 1
                D_pkt = pkt['D']
                wait  = D_pkt - hops
                wait_until[i] = t + wait
                for child in children[i]:
                    dl_next[child] = {'D': D_pkt, 'hops': hops}
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
            phase         = 'uplink'
            cycle_start_t = t

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
            else:
                cum_regret[i, t] = cum_regret[i, t - 1] if t > 0 else 0.0
        t += 1

    return cum_regret


# ============================================================
# Algorithm: Hillel et al. 2013 -- Distributed Successive Elimination
# ============================================================

def run_hillel(env, T, G, N, delta=0.05, sigma=1.0):
    """
    Hillel et al. (NeurIPS 2013) distributed successive elimination for regret tracking.

    Each communication round r:
      - Each agent uniformly explores all surviving arms for L_r pulls each.
      - Agents send local mean estimates to a central coordinator (hub).
      - Hub averages across agents, eliminates arms whose UCB < best LCB.
      - Hub broadcasts surviving set back to agents.

    L_r = ceil( (2 * sigma^2 / Delta_r^2) * log(4 * K * R / delta) )
    where Delta_r is halved each round and R = ceil(log2(K)) total rounds.

    Here we adapt this to a regret-tracking experiment: agents pull arms according
    to the Hillel schedule, and we record instantaneous regret at each pull.

    Communication is via the MERW hub (highest psi node) acting as coordinator.
    """
    K = env.K
    psi, _ = merw_eigenvector(G)
    hub = int(np.argmax(psi))

    # Number of elimination rounds: ceil(log2(K))
    R = int(np.ceil(np.log2(max(K, 2))))
    log_term = np.log(max(4.0 * K * R / delta, 2.0))

    cum_regret = np.zeros((N, T))
    t_global = 0  # current time step across all agents

    surviving = list(range(K))
    Delta = 1.0  # initial gap scale (halved each round)

    for r in range(R):
        if len(surviving) <= 1:
            break
        # pulls per arm per agent this round
        L = int(np.ceil(2.0 * sigma ** 2 / (Delta ** 2) * log_term))
        L = max(L, 1)

        # Each agent pulls each surviving arm L times
        local_sums = np.zeros((N, K))
        local_counts = np.zeros((N, K))

        for k in surviving:
            for i in range(N):
                for _ in range(L):
                    if t_global >= T:
                        break
                    r_val = env.pull(k)
                    local_sums[i, k] += r_val
                    local_counts[i, k] += 1
                    prev = cum_regret[i, t_global - 1] if t_global > 0 else 0.0
                    cum_regret[i, t_global] = prev + env.gap(k)
                    t_global += 1
                if t_global >= T:
                    break
            if t_global >= T:
                break

        # Hub aggregates: average across all agents
        global_mean = np.zeros(K)
        global_n = np.zeros(K)
        for k in surviving:
            global_n[k] = local_counts[:, k].sum()
            if global_n[k] > 0:
                global_mean[k] = local_sums[:, k].sum() / global_n[k]

        # Eliminate: drop arm k if UCB(k) < LCB(best)
        cb = np.where(global_n > 0,
                      sigma * np.sqrt(log_term / np.maximum(global_n, 1)),
                      np.inf)
        ucb_vals = {k: global_mean[k] + cb[k] for k in surviving}
        lcb_vals = {k: global_mean[k] - cb[k] for k in surviving}
        best_lcb_arm = max(surviving, key=lambda k: lcb_vals[k])
        surviving = [k for k in surviving
                     if ucb_vals[k] >= lcb_vals[best_lcb_arm]]

        Delta /= 2.0

    # Exploit the identified best arm for remaining rounds
    best_arm = surviving[0] if surviving else 0
    for i in range(N):
        for t in range(t_global, T):
            env.pull(best_arm)
            prev = cum_regret[i, t - 1] if t > 0 else 0.0
            cum_regret[i, t] = prev + env.gap(best_arm)

    return cum_regret


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
    EigenTree-BAI: Hillel successive elimination over the MERW spanning tree.

    Each elimination round r has three explicit phases simulated step-by-step:

      Pull phase (L_r steps):
        All nodes pull each surviving arm once per step (L_r steps total).
        No communication. Local sums accumulate in local_sums[i, k].

      Uplink phase (D steps):
        Local sums travel hop-by-hop up the tree to the hub.
        Each node sends its local_sums packet to its parent once per step.

      Hub step:
        Hub aggregates all received sums, computes global averages,
        eliminates arms where p_tilde_k < p_tilde_star - epsilon_r,
        produces new surviving set S_r.

      Downlink phase (D steps):
        Hub broadcasts (S_r, D) hop-by-hop down the tree.
        Nodes at depth d receive the packet at downlink step d.
        All nodes are synchronized: next pull phase starts together.

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
        local_sums = np.zeros((N, K))
        for _ in range(L):
            for i in range(N):
                for k in surviving:
                    local_sums[i, k] += env.pull(k)
                    total_pulls += 1

        # --- Uplink phase: packets travel hop-by-hop to hub ---
        # pending[i] holds the sum this node still needs to forward upstream.
        # Each step: every node with a pending packet forwards it one hop.
        # A node at depth d reaches the hub after d steps.
        pending = [local_sums[i].copy() for i in range(N)]
        hub_agg = np.zeros(K)

        for _ in range(tree_depth):
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
        # hub's own local sums (never relayed, just added directly)
        hub_agg += local_sums[hub]

        # --- Hub: compute global averages and eliminate ---
        global_mean = {k: hub_agg[k] / (N * L) for k in surviving}
        p_star = max(global_mean[k] for k in surviving)
        surviving = [k for k in surviving if global_mean[k] >= p_star - epsilon_r]

        # --- Downlink phase: hub broadcasts S_r + D hop-by-hop ---
        # Simulated as tree_depth steps; we only track pull count so
        # no extra pulls happen here. Nodes receive S_r and are synchronized
        # for the next round by the D signal.
        # (no pulls during downlink -- communication only)

    return total_pulls


# ============================================================
# Parallel worker
# ============================================================

def _worker(task):
    (algo_name, run_seed, graph_seed, graph_type, N, K,
     means, sigma, T, c, nu) = task
    np.random.seed(run_seed)
    G = make_graph(graph_type, N, seed=graph_seed)
    env = BanditEnv(means, sigma=sigma)

    if algo_name == "UCB-Ind":
        cr = run_ucb_ind(env, T, N, c=c, sigma=sigma)
    elif algo_name == "Coop-UCB2":
        cr = run_coop_ucb2(env, T, G, N, sigma=sigma)
    elif algo_name == EIGENTREE_UCB_NAME:
        cr = run_merw_ucb(env, T, G, N, c=c, sigma=sigma)
    elif algo_name == "Hillel":
        cr = run_hillel(env, T, G, N, sigma=sigma)
    else:
        raise ValueError(algo_name)

    psi, _ = merw_eigenvector(G)
    hub, _, _, depth, _ = build_routing_tree(G, psi)
    return algo_name, cr.sum(axis=0), cr[hub]


def _worker_bai(task):
    (algo_name, run_seed, graph_seed, graph_type, N, K,
     means, sigma, delta, c, nu, p) = task
    np.random.seed(run_seed)
    G = make_graph(graph_type, N, seed=graph_seed, p=p)
    env = BanditEnv(means, sigma=sigma)

    if algo_name == "Hillel-BAI":
        pulls = bai_hillel(env, N, delta=delta, sigma=sigma)
    elif algo_name == EIGENTREE_BAI_NAME:
        pulls = bai_merw(env, G, N, delta=delta, sigma=sigma, c=c)
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
# Experiment
# ============================================================

def run_experiment(n_runs, T, N, K, graph_type, sigma, c, n_workers, nu=1.0, seed=0):
    rng = np.random.RandomState(seed)

    # Arm means: equally spaced, best arm = 1.0
    means = np.linspace(0.0, 1.0, K)[::-1]   # means[0] = 1.0 is best

    algo_names = ["UCB-Ind", "Coop-UCB2", EIGENTREE_UCB_NAME, "Hillel"]

    tasks = []
    for _ in range(n_runs):
        run_seed = int(rng.randint(0, 2**31))
        graph_seed = int(rng.randint(0, 2**31))
        for name in algo_names:
            tasks.append((name, run_seed, graph_seed, graph_type,
                          N, K, means, sigma, T, c, nu))

    results = _run_parallel(tasks, algo_names, n_workers, tag="merw-ucb")

    # ---- Plot ----
    styles = {
        "UCB-Ind":          ("C0", "-",  "UCB-Ind (no comm.)"),
        "Coop-UCB2":        ("C1", "--", "Coop-UCB2"),
        EIGENTREE_UCB_NAME: ("C2", "-",  EIGENTREE_UCB_NAME),
        "Hillel":           ("C3", ":",  "Hillel (succ. elim.)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts = np.arange(1, T + 1)
    info = f"$N={N}$, $K={K}$, $T={T}$, {n_runs} runs"

    for name, (color, ls, label) in styles.items():
        if name not in results:
            continue
        mean_g, std_g, mean_h, std_h = results[name]
        axes[0].plot(ts, mean_g, label=label, color=color, linestyle=ls, linewidth=1.8)
        axes[0].fill_between(ts, mean_g - std_g, mean_g + std_g, color=color, alpha=0.15)
        if name == "UCB-Ind":
            continue
        axes[1].plot(ts, mean_h, label=label, color=color, linestyle=ls, linewidth=1.8)
        axes[1].fill_between(ts, mean_h - std_h, mean_h + std_h, color=color, alpha=0.15)

    axes[0].set_xlabel("Round $t$")
    axes[0].set_ylabel("Group cumulative regret $\\sum_i R_i(t)$")
    axes[0].set_title(f"Group regret — {graph_type.upper()} graph\n{info}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Round $t$")
    axes[1].set_ylabel("Hub cumulative regret $R_{i^\\star}(t)$")
    axes[1].set_title(f"Hub (best) agent regret — {graph_type.upper()} graph\n{info}")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = REGRET_DIR / f"merw_ucb_regret_{graph_type}_N{N}_K{K}_T{T}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Nu sweep experiment
# ============================================================

def _worker_nu(task):
    label, run_seed, graph_seed, graph_type, N, means, sigma, T, c, nu = task
    np.random.seed(run_seed)
    G = make_graph(graph_type, N, seed=graph_seed)
    env = BanditEnv(means, sigma=sigma)
    cr = run_merw_ucb(env, T, G, N, c=c, sigma=sigma, boost_type="D2", nu=nu)
    psi, _ = merw_eigenvector(G)
    hub = int(np.argmax(psi))
    return label, cr.sum(axis=0), cr[hub]


def run_nu_sweep(n_runs, T, N, K, graph_type, sigma, c, n_workers, seed=1):
    rng = np.random.RandomState(seed)
    means = np.linspace(0.0, 1.0, K)[::-1]
    nus = [0.1,0.25, 0.5, 1.0]
    labels = [f"nu={nu}" for nu in nus]

    tasks = []
    for _ in range(n_runs):
        run_seed = int(rng.randint(0, 2**31))
        graph_seed = int(rng.randint(0, 2**31))
        for nu, label in zip(nus, labels):
            tasks.append((label, run_seed, graph_seed, graph_type,
                          N, means, sigma, T, c, nu))

    by_nu = {label: ([], []) for label in labels}
    n_workers_ = min(n_workers, max(1, len(tasks)))
    report_every = max(1, len(tasks) // 20)
    print(f"[nu-sweep] dispatching {len(tasks)} tasks to {n_workers_} workers")
    t0 = time.time()
    with mp.Pool(processes=n_workers_) as pool:
        for done, (label, grp, hub_r) in enumerate(
                pool.imap_unordered(_worker_nu, tasks), start=1):
            by_nu[label][0].append(grp)
            by_nu[label][1].append(hub_r)
            if done % report_every == 0 or done == len(tasks):
                print(f"[nu-sweep] {done}/{len(tasks)} done ({time.time()-t0:.1f}s)")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(nus)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts = np.arange(1, T + 1)
    info = f"$N={N}$, $K={K}$, $T={T}$, {n_runs} runs"

    for (name, (g_list, h_list)), color in zip(by_nu.items(), colors):
        mean_g = np.array(g_list).mean(axis=0)
        std_g  = np.array(g_list).std(axis=0)
        mean_h = np.array(h_list).mean(axis=0)
        std_h  = np.array(h_list).std(axis=0)
        axes[0].plot(ts, mean_g, label=f"$\\nu={name.split('=')[1]}$",
                     color=color, linewidth=1.8)
        axes[0].fill_between(ts, mean_g - std_g, mean_g + std_g, color=color, alpha=0.15)
        axes[1].plot(ts, mean_h, label=f"$\\nu={name.split('=')[1]}$",
                     color=color, linewidth=1.8)
        axes[1].fill_between(ts, mean_h - std_h, mean_h + std_h, color=color, alpha=0.15)

    for ax, ylabel, title in zip(axes,
            ["Group cumulative regret $\\sum_i R_i(t)$",
             "Hub cumulative regret $R_{i^\\star}(t)$"],
            [f"EigenTreeUCB: group regret vs. $\\nu$ — {graph_type.upper()} graph\n{info}",
             f"EigenTreeUCB: hub regret vs. $\\nu$ — {graph_type.upper()} graph\n{info}"]):
        ax.set_xlabel("Round $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = REGRET_DIR / f"merw_ucb_nu_sweep_{graph_type}_N{N}_K{K}_T{T}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# BAI experiment: sample complexity vs N
# ============================================================

BAI_ALGO_NAMES = ["Hillel-BAI", EIGENTREE_BAI_NAME]
BAI_N_VALS = [10, 20, 30, 40, 50]

BAI_STYLES = {
    "Hillel-BAI":       ("C3", "o", "Hillel-BAI"),
    EIGENTREE_BAI_NAME: ("C2", "s", EIGENTREE_BAI_NAME),
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
            run_seed = int(sub_rng.randint(0, 2**31))
            for name in BAI_ALGO_NAMES:
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
            results[name]["std"].append(arr.std() / np.sqrt(len(arr)))
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
    fig, axes = plt.subplots(2, 1, figsize=(6, 7.5))
    info = f"$K={K}$, $\\delta={delta}$, {n_runs} runs (error bars: SEM)"

    handles, labels = [], []
    for name, (color, marker, label) in BAI_STYLES.items():
        if name not in results:
            continue
        means_arr = np.array(results[name]["mean"])
        stds_arr = np.array(results[name]["std"])
        line = axes[0].errorbar(N_vals, means_arr, yerr=stds_arr,
                                label=label, color=color, marker=marker,
                                linewidth=2.2, capsize=4, markersize=7)
        handles.append(line)
        labels.append(label)

    axes[0].set_xlabel("Number of agents $N$", fontsize=15)
    axes[0].set_ylabel("Total arm pulls", fontsize=15)
    axes[0].set_title(f"BAI sample complexity vs. $N$\n{info}", fontsize=16)
    axes[0].tick_params(labelsize=13)
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
                         linewidth=2.2, capsize=4, markersize=7)

    axes[1].set_xlabel("Number of agents $N$", fontsize=15)
    axes[1].set_ylabel("Pulls per agent", fontsize=15)
    axes[1].set_title(f"BAI per-agent sample complexity vs. $N$\n{info}", fontsize=16)
    axes[1].tick_params(labelsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=15,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
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
    p.add_argument("--graph",     choices=("ba", "er"), default="ba",
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
