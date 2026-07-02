"""
MERW Routing Tree Visualizer

Three panels, all using the same spring-layout node positions:

  Panel 1 — Undirected graph
    Nodes colored and sized by ψ² (MERW stationary mass).
    All edges drawn as plain undirected lines.

  Panel 2 — Routing tree overlay
    Same undirected graph faded in the background.
    Tree edges highlighted in a warm color (thick, no arrows).
    Hub node ringed in gold.

  Panel 3 — Induced directed graph
    Only the routing tree edges, now drawn as directed arrows
    (child → parent, i.e. toward the hub).
    Hub ringed in gold, depth annotations.

Usage:
    python visualize_merw.py                           # BA, N=20, seed=0
    python visualize_merw.py --graph er --N 30
    python visualize_merw.py --graph barbell --N 20
    python visualize_merw.py --graph clustered --N 24
    python visualize_merw.py --N 20 --seed 7
"""

import argparse
import csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
plt.style.use(Path(__file__).resolve().parent / "merw.mplstyle")

RESULTS_DIR  = Path(__file__).resolve().parent.parent / "paper" / "results"
MERW_VIZ_DIR = RESULTS_DIR / "MERW_visualization"
MERW_VIZ_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BG          = "#fafaf8"   # warm off-white background
EDGE_GREY   = "#c8c8d8"   # non-tree edges
TREE_ARROW  = "#333333"   # directed arrows (panel 3)
HUB_RING    = "#f4a261"   # hub ring

# One colormap per subtree — clipped to avoid very pale shades
SUBTREE_CMAPS = [
    matplotlib.colormaps["Blues"],
    matplotlib.colormaps["Oranges"],
    matplotlib.colormaps["Greens"],
    matplotlib.colormaps["Purples"],
    matplotlib.colormaps["Reds"],
]
# Panel-1 uses a single neutral colormap for global centrality
GLOBAL_CMAP = matplotlib.colormaps["YlOrRd"]


# ── helpers ───────────────────────────────────────────────────────────────────

def make_ba_graph(N, m=2, seed=None):
    for attempt in range(50):
        s = seed + attempt if seed is not None else None
        G = nx.barabasi_albert_graph(N, m, seed=s)
        if nx.is_connected(G):
            return G
    raise RuntimeError("Cannot make connected BA graph.")


def make_er_graph(N, seed=None, p=None):
    if p is None:
        p = 2.5 * np.log(N) / N
    rng = np.random.RandomState(seed)
    for _ in range(100):
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.randint(0, 2**31)))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Cannot make connected ER graph.")


def make_barbell_graph(N, seed=None):
    """Two cliques of size N//2 connected by a single bridge edge."""
    n1 = N // 2
    n2 = N - n1
    G = nx.complete_graph(n1)
    clique2 = nx.complete_graph(n2)
    # relabel second clique so nodes don't overlap
    clique2 = nx.relabel_nodes(clique2, {i: i + n1 for i in range(n2)})
    G = nx.compose(G, clique2)
    G.add_edge(n1 - 1, n1)  # single bridge
    return G


def make_grid_graph(N, seed=None):
    """2D grid graph with side length floor(sqrt(N)), relabeled 0..n-1."""
    k = int(np.floor(np.sqrt(N)))
    G = nx.grid_2d_graph(k, k)
    G = nx.convert_node_labels_to_integers(G)
    return G


def make_cycle_graph(N, seed=None):
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


def make_clustered_graph(N, seed=None):
    """Several dense clusters (ER internally) connected by sparse inter-cluster edges.
    Returns (G, clusters) where clusters is a list of node lists."""
    rng = np.random.RandomState(seed)
    n_clusters = max(2, N // 6)
    sizes = []
    remaining = N
    for i in range(n_clusters - 1):
        s = max(3, remaining // (n_clusters - i))
        sizes.append(s)
        remaining -= s
    sizes.append(max(3, remaining))

    clusters = []
    offset = 0
    G = nx.Graph()
    for s in sizes:
        p_in = 0.7
        for _ in range(100):
            C = nx.erdos_renyi_graph(s, p_in, seed=int(rng.randint(0, 2**31)))
            if nx.is_connected(C):
                break
        C = nx.relabel_nodes(C, {i: i + offset for i in range(s)})
        G = nx.compose(G, C)
        clusters.append(list(range(offset, offset + s)))
        offset += s

    # connect clusters in a ring with one inter-cluster edge each
    for i in range(n_clusters):
        u = rng.choice(clusters[i])
        v = rng.choice(clusters[(i + 1) % n_clusters])
        G.add_edge(u, v)

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        while len(components) > 1:
            u = rng.choice(list(components[0]))
            v = rng.choice(list(components[1]))
            G.add_edge(u, v)
            components = list(nx.connected_components(G))

    return G, clusters


def merw_eigenvector(G, tau=500, tol=1e-8, gossip_rounds=5):
    """Distributed power iteration with gossip-based normalization.

    Implements the algorithm from Jelasity, Canright, Engo-Monsen (EuroPar 2007):
      - Each node holds w_i (eigenvector component) and b_ki (buffered incoming).
      - Iteration: w_i <- sum_k b_ki  (Fig. 1, line 5)
      - Gossip normalization: each node tracks r_i = log(w_i_new / w_i_old),
        then gossips r_i via averaging to approximate the geometric mean growth
        rate across all nodes. Each node divides w_i by exp(r_i) to normalize.
      - No global norm, no knowledge of N required.

    Returns (psi, lambda_approx, history).
    """
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    # init: w_i = 1, b_ki = 1 for all (paper Sec 5.3)
    w = np.ones(N)
    b = np.ones((N, N))  # b[k, i] = buffered value from k at i

    r = np.zeros(N)  # local growth rate approximations
    history = []

    for _ in range(tau):
        w_old = w.copy()

        # iteration step: w_i <- sum_k A_ki * b_ki  (chaotic async iteration)
        w_new = A @ w_old

        # local growth rate: r_i = log(w_new_i / w_old_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_growth = np.where(w_old > 1e-15, np.log(np.abs(w_new) / np.abs(w_old)), 0.0)

        # gossip: average log_growth across all nodes (simulates distributed averaging)
        r = log_growth.copy()
        for _ in range(gossip_rounds):
            for i in range(N):
                nbrs = list(G.neighbors(i))
                if nbrs:
                    j = nbrs[np.random.randint(len(nbrs))]
                    r[i] = r[j] = (r[i] + r[j]) / 2

        # normalize each component by its local growth rate approximation
        w = w_new / np.exp(r)

        diff = np.max(np.abs(w - w_old) / (np.abs(w_old) + 1e-15))
        history.append(diff)
        if diff < tol:
            break

    w = np.abs(w)
    # lambda approximated as geometric mean of final growth rates
    lam = np.exp(np.mean(log_growth))
    # shape is correct up to scale; max-normalize so max component = 1
    w /= w.max()
    return w, lam, history


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

    # max-flooding: each node floods its psi outward one hop per round.
    # Stops as soon as no node updates its best value (local convergence check).
    # No diameter or global topology knowledge needed.
    m   = psi.copy()    # best psi seen so far at each node
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

    # Each local maximum re-routes toward the neighbor carrying a better psi.
    # The node whose psi is the global maximum sees no better neighbor and
    # keeps parent = -1, becoming the hub naturally.
    for i in local_maxima:
        if m[via[i]] > psi[i]:
            parent[i] = via[i]

    remaining = [i for i in range(N) if parent[i] < 0]
    hub = remaining[0]

    hubs = remaining
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
    return hub, hubs, parent, children, depth, int(depth.max())


def _subtree_layout(root, children, depth, n_leaves, x_offset):
    """Place one subtree rooted at `root`; returns a pos dict."""
    pos = {}

    def place(node, x_left):
        slot_width = n_leaves[node]
        pos[node]  = (x_offset + x_left + slot_width / 2.0, depth[node] * 2.0)
        cursor = x_left
        for c in children[node]:
            place(c, cursor)
            cursor += n_leaves[c]

    place(root, 0.0)
    return pos


def hierarchical_layout(hubs, children, depth, N):
    """Reingold-Tilford layout; each hub's subtree is placed side-by-side."""
    n_leaves = np.zeros(N, dtype=float)

    def count_leaves(node):
        if not children[node]:
            n_leaves[node] = 1.0
        else:
            for c in children[node]:
                count_leaves(c)
            n_leaves[node] = sum(n_leaves[c] for c in children[node])

    for h in hubs:
        count_leaves(h)

    pos = {}
    cursor = 0.0
    GAP = 2.0  # horizontal gap between separate subtrees
    for h in hubs:
        sub = _subtree_layout(h, children, depth, n_leaves, cursor)
        pos.update(sub)
        cursor += n_leaves[h] + GAP

    return pos


def global_node_style(pi, N):
    """Panel 1: single colormap over all nodes by global ψ²."""
    norm   = mcolors.Normalize(vmin=0, vmax=pi.max())
    colors = [GLOBAL_CMAP(0.25 + 0.70 * norm(pi[i])) for i in range(N)]
    sizes  = 300 + 1800 * (pi / pi.max())
    return colors, sizes, norm


def subtree_node_style(pi, N, hubs, subtree_of):
    """Panels 2 & 3: each subtree gets its own colormap, shaded by local ψ²."""
    colors = [None] * N
    sizes  = 300 + 1800 * (pi / pi.max())
    hub_cmap = {}
    for idx, h in enumerate(hubs):
        hub_cmap[h] = SUBTREE_CMAPS[idx % len(SUBTREE_CMAPS)]

    for h in hubs:
        cmap = hub_cmap[h]
        members = [i for i in range(N) if subtree_of[i] == h]
        pi_sub  = np.array([pi[i] for i in members])
        pi_max  = pi_sub.max() if pi_sub.max() > 0 else 1.0
        for i in members:
            t = 0.30 + 0.65 * (pi[i] / pi_max)   # clip to [0.30, 0.95]
            colors[i] = cmap(t)

    return colors, sizes, hub_cmap


# ── drawing primitives ────────────────────────────────────────────────────────

def draw_undirected(ax, G, pos, colors, sizes, alpha_edges=1.0):
    nx.draw_networkx_edges(G, pos, edge_color=EDGE_GREY,
                           width=1.2, alpha=alpha_edges, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors,
                           node_size=sizes, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={i: str(i) for i in G.nodes()},
                            font_size=7, font_color="#333333", ax=ax)


def draw_hub_ring(ax, pos, hub, sizes):
    nx.draw_networkx_nodes(nx.Graph(), pos, nodelist=[hub],
                           node_color="none",
                           node_size=sizes[hub] * 1.7,
                           edgecolors=HUB_RING, linewidths=3.5, ax=ax)


def _rgba_to_hex(rgba):
    return mcolors.to_hex(rgba)


def draw_tree_edges_undirected(ax, pos, tree_edges, subtree_of, hub_cmap):
    # group edges by hub so each call gets a single color string
    by_hub = {}
    for p, c in tree_edges:
        h = subtree_of[c]
        by_hub.setdefault(h, []).append((p, c))
    for h, edges in by_hub.items():
        color = _rgba_to_hex(hub_cmap[h](0.72))
        g = nx.Graph()
        g.add_edges_from(edges)
        nx.draw_networkx_edges(g, pos, edgelist=edges,
                               edge_color=color, width=3.0,
                               alpha=0.92, ax=ax)


def draw_tree_edges_directed(ax, pos, tree_edges, subtree_of, hub_cmap):
    # arrows point child → parent; group by hub for single-color calls
    by_hub = {}
    for p, c in tree_edges:
        h = subtree_of[c]
        by_hub.setdefault(h, []).append((c, p))  # reversed: child→parent
    for h, directed in by_hub.items():
        color = _rgba_to_hex(hub_cmap[h](0.72))
        dg = nx.DiGraph()
        dg.add_edges_from(directed)
        nx.draw_networkx_edges(dg, pos, edgelist=directed,
                               edge_color=color, width=2.0,
                               arrows=True, arrowstyle="-|>", arrowsize=18,
                               connectionstyle="arc3,rad=0.05",
                               ax=ax)


# ── layout helpers ────────────────────────────────────────────────────────────

def _graph_layout(G, graph_type, seed, clusters=None):
    """Return a pos dict suited to the graph's structure."""
    rng_seed = seed if seed is not None else 0
    if graph_type == "barbell":
        N = G.number_of_nodes()
        n1 = N // 2
        shell1 = list(range(n1))
        shell2 = list(range(n1, N))
        pos = nx.shell_layout(G.subgraph(shell1), nlist=[shell1])
        pos2 = nx.shell_layout(G.subgraph(shell2), nlist=[shell2])
        offset = 3.0
        for node, (x, y) in pos2.items():
            pos[node] = (x + offset, y)
        return pos
    if graph_type == "clustered" and clusters is not None:
        # place each cluster in a small circle, then arrange circles on a ring
        n_clusters = len(clusters)
        cluster_radius = 0.35          # radius of each per-cluster circle
        ring_radius    = 1.2 + 0.15 * n_clusters  # radius of the ring of clusters
        pos = {}
        for ci, nodes in enumerate(clusters):
            # center of this cluster on the outer ring
            angle_c = 2 * np.pi * ci / n_clusters
            cx = ring_radius * np.cos(angle_c)
            cy = ring_radius * np.sin(angle_c)
            # nodes arranged in a small circle around that center
            for ni, node in enumerate(nodes):
                angle_n = 2 * np.pi * ni / max(len(nodes), 1)
                pos[node] = (cx + cluster_radius * np.cos(angle_n),
                             cy + cluster_radius * np.sin(angle_n))
        return pos
    if graph_type == "grid":
        k = int(np.floor(np.sqrt(G.number_of_nodes())))
        return {i: (i % k, -(i // k)) for i in G.nodes()}
    if graph_type == "lollipop":
        N = G.number_of_nodes()
        n_clique = N // 2
        clique_nodes = list(range(n_clique))
        path_nodes   = list(range(n_clique, N))
        pos = nx.shell_layout(G.subgraph(clique_nodes), nlist=[clique_nodes])
        for idx, node in enumerate(path_nodes):
            pos[node] = (1.5 + idx * 0.6, 0.0)
        return pos
    if graph_type == "chain":
        return {i: (i, 0) for i in G.nodes()}
    if graph_type == "cycle":
        N = G.number_of_nodes()
        return {i: (np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)) for i in G.nodes()}
    if graph_type == "star":
        return nx.spring_layout(G, seed=rng_seed, k=2.5)
    # default: spring layout
    return nx.spring_layout(G, seed=rng_seed, k=1.8)


# ── main ──────────────────────────────────────────────────────────────────────

def _tag(graph_type, N, p=None):
    suffix = f"_p{p}" if p is not None else ""
    return f"{graph_type}_N{N}{suffix}"


def _meta_csv_path(graph_type, N, p=None):
    return DATA_DIR / f"merw_tree_{_tag(graph_type, N, p)}_meta.csv"


def _nodes_csv_path(graph_type, N, p=None):
    return DATA_DIR / f"merw_tree_{_tag(graph_type, N, p)}_nodes.csv"


def _edges_csv_path(graph_type, N, p=None):
    return DATA_DIR / f"merw_tree_{_tag(graph_type, N, p)}_edges.csv"


def compute_merw_data(G, graph_type, seed, clusters=None):
    """Run MERW eigenvector + routing-tree computation and the graph layout.

    Returns a dict with everything plot_merw() needs to draw the three
    panels, with no further numerics required.
    """
    N = G.number_of_nodes()
    d_max = max(dict(G.degree()).values())
    gossip_rounds = N if d_max == N - 1 else 5
    psi, lam, hist_gossip = merw_eigenvector(G, gossip_rounds=gossip_rounds)

    pi = psi ** 2
    hub, hubs, parent, children, depth, tree_depth = build_routing_tree(G, psi)

    # subtree membership
    subtree_of = np.full(N, -1, dtype=int)
    for h in hubs:
        subtree_of[h] = h
    queue = list(hubs)
    while queue:
        node = queue.pop(0)
        for c in children[node]:
            subtree_of[c] = subtree_of[node]
            queue.append(c)

    tree_edges     = [(parent[i], i) for i in range(N) if parent[i] >= 0]
    non_tree_edges = [e for e in G.edges()
                      if e not in tree_edges and (e[1], e[0]) not in tree_edges]

    pos = _graph_layout(G, graph_type, seed, clusters=clusters)
    pos_tree = hierarchical_layout(hubs, children, depth, N)

    return {
        "N": N, "graph_type": graph_type, "seed": seed,
        "psi": psi, "pi": pi, "lam": lam,
        "hub": hub, "hubs": hubs, "parent": parent, "children": children,
        "depth": depth, "tree_depth": tree_depth, "subtree_of": subtree_of,
        "tree_edges": tree_edges, "non_tree_edges": non_tree_edges,
        "degree": dict(G.degree()), "pos": pos, "pos_tree": pos_tree,
    }


def save_merw_csv(data, G, p=None):
    graph_type, N = data["graph_type"], data["N"]

    with open(_meta_csv_path(graph_type, N, p), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_type", "N", "seed", "hub", "lam", "tree_depth"])
        writer.writerow([graph_type, N, data["seed"], data["hub"],
                          data["lam"], data["tree_depth"]])

    with open(_nodes_csv_path(graph_type, N, p), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "psi", "pi", "parent", "depth", "subtree_hub",
                          "degree", "pos_x", "pos_y", "pos_tree_x", "pos_tree_y"])
        for i in range(N):
            px, py = data["pos"][i]
            ptx, pty = data["pos_tree"][i]
            writer.writerow([i, data["psi"][i], data["pi"][i], data["parent"][i],
                              data["depth"][i], data["subtree_of"][i],
                              data["degree"][i], px, py, ptx, pty])

    with open(_edges_csv_path(graph_type, N, p), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "is_tree_edge"])
        tree_edge_set = {tuple(e) for e in data["tree_edges"]}
        for u, v in G.edges():
            is_tree = (u, v) in tree_edge_set or (v, u) in tree_edge_set
            writer.writerow([u, v, int(is_tree)])

    print(f"  Saved {_meta_csv_path(graph_type, N, p)}, "
          f"{_nodes_csv_path(graph_type, N, p)}, {_edges_csv_path(graph_type, N, p)}")


def load_merw_data(graph_type, N, p=None):
    with open(_meta_csv_path(graph_type, N, p), newline="") as f:
        meta = next(csv.DictReader(f))
    seed = int(meta["seed"])
    hub = int(meta["hub"])
    lam = float(meta["lam"])
    tree_depth = int(meta["tree_depth"])

    with open(_nodes_csv_path(graph_type, N, p), newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["node"]))

    psi   = np.array([float(r["psi"]) for r in rows])
    pi    = np.array([float(r["pi"]) for r in rows])
    parent = np.array([int(r["parent"]) for r in rows], dtype=int)
    depth  = np.array([int(r["depth"]) for r in rows], dtype=int)
    subtree_of = np.array([int(r["subtree_hub"]) for r in rows], dtype=int)
    degree = {int(r["node"]): int(r["degree"]) for r in rows}
    pos      = {int(r["node"]): (float(r["pos_x"]), float(r["pos_y"])) for r in rows}
    pos_tree = {int(r["node"]): (float(r["pos_tree_x"]), float(r["pos_tree_y"])) for r in rows}

    children = [[] for _ in range(N)]
    for i in range(N):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    hubs = [i for i in range(N) if parent[i] < 0]

    with open(_edges_csv_path(graph_type, N, p), newline="") as f:
        edge_rows = list(csv.DictReader(f))
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for r in edge_rows:
        G.add_edge(int(r["src"]), int(r["dst"]))
    tree_edges = [(int(r["src"]), int(r["dst"])) for r in edge_rows if int(r["is_tree_edge"])]
    non_tree_edges = [(int(r["src"]), int(r["dst"])) for r in edge_rows if not int(r["is_tree_edge"])]

    return {
        "N": N, "graph_type": graph_type, "seed": seed,
        "psi": psi, "pi": pi, "lam": lam,
        "hub": hub, "hubs": hubs, "parent": parent, "children": children,
        "depth": depth, "tree_depth": tree_depth, "subtree_of": subtree_of,
        "tree_edges": tree_edges, "non_tree_edges": non_tree_edges,
        "degree": degree, "pos": pos, "pos_tree": pos_tree,
    }, G


def plot_merw(data, G, p=None):
    N, graph_type = data["N"], data["graph_type"]
    pi, lam, hub, hubs = data["pi"], data["lam"], data["hub"], data["hubs"]
    parent, children = data["parent"], data["children"]
    depth, tree_depth = data["depth"], data["tree_depth"]
    subtree_of = data["subtree_of"]
    tree_edges, non_tree_edges = data["tree_edges"], data["non_tree_edges"]
    pos, pos_tree = data["pos"], data["pos_tree"]
    degree = data["degree"]

    # per-panel node styles
    g_colors, g_sizes, g_norm = global_node_style(pi, N)
    s_colors, s_sizes, hub_cmap = subtree_node_style(pi, N, hubs, subtree_of)

    tree_node_size = max(150, min(500, 4000 // N))
    tree_sizes_flat = [tree_node_size] * N

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    hub_str = ", ".join(str(h) for h in hubs)

    # ── panel 1: graph colored by global ψ² centrality ───────────────────
    ax = axes[0]
    nx.draw_networkx_edges(G, pos, edge_color=EDGE_GREY,
                           width=1.2, alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=g_colors,
                           node_size=g_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={i: str(i) for i in G.nodes()},
                            font_size=7, font_color="#222222", ax=ax)
    # colorbar for panel 1
    cmap_p1 = mcolors.LinearSegmentedColormap.from_list(
        "YlOrRd_clip", [GLOBAL_CMAP(0.25 + 0.70 * x) for x in np.linspace(0, 1, 256)])
    sm1 = cm.ScalarMappable(cmap=cmap_p1, norm=g_norm)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax, orientation="vertical",
                         fraction=0.04, pad=0.02, shrink=0.8)
    cbar1.set_label(r"$\psi_i / \psi_{\max}$", fontsize=9)
    ax.set_title("Undirected graph\n"
                 r"(nodes colored by $\psi_i / \psi_{\max}$)",
                 fontsize=10, pad=8)

    # ── panel 2: subtree-colored overlay on graph ─────────────────────────
    ax = axes[1]
    nx.draw_networkx_edges(G, pos, edgelist=non_tree_edges,
                           edge_color=EDGE_GREY, width=0.8, alpha=0.25, ax=ax)
    draw_tree_edges_undirected(ax, pos, tree_edges, subtree_of, hub_cmap)
    nx.draw_networkx_nodes(G, pos, node_color=s_colors,
                           node_size=s_sizes, ax=ax)
    for h in hubs:
        draw_hub_ring(ax, pos, h, s_sizes)
    nx.draw_networkx_labels(G, pos,
                            labels={i: str(i) for i in G.nodes()},
                            font_size=7, font_color="#222222", ax=ax)
    ax.set_title(f"Routing subtrees (undirected)\nhub(s) = {{{hub_str}}}  ·  D = {tree_depth}",
                 fontsize=10, pad=8)

    # ── panel 3: directed hierarchical layout, same subtree colors ────────
    ax = axes[2]
    ax.invert_yaxis()

    draw_tree_edges_directed(ax, pos_tree, tree_edges, subtree_of, hub_cmap)
    nx.draw_networkx_nodes(G, pos_tree, node_color=s_colors,
                           node_size=tree_sizes_flat, ax=ax)
    for h in hubs:
        nx.draw_networkx_nodes(nx.Graph(), pos_tree, nodelist=[h],
                               node_color="none",
                               node_size=tree_sizes_flat[h] * 2.2,
                               edgecolors=HUB_RING, linewidths=2.5, ax=ax)

    depth_labels = {}
    for i in range(N):
        if children[i]:
            depth_labels[i] = r"$" + str(i) + r"$" + "\n" + r"$\psi^2=" + f"{pi[i]:.2f}" + r"$"
        else:
            depth_labels[i] = str(i)
    nx.draw_networkx_labels(G, pos_tree, labels=depth_labels,
                            font_size=6, font_color="#222222", ax=ax)

    for d in range(tree_depth + 1):
        nodes_at_d = [i for i in range(N) if depth[i] == d]
        if nodes_at_d:
            y     = np.mean([pos_tree[i][1] for i in nodes_at_d])
            x_min = min(pos_tree[i][0] for i in pos_tree) - 1.2
            ax.text(x_min, y, f"depth {d}", fontsize=8,
                    color="#555555", va="center", ha="right")

    hub_label = f"hub(s) = {{{hub_str}}}" if len(hubs) > 1 else f"hub = node {hub}"
    ax.set_title(f"Induced directed graph (hierarchical layout)\n{hub_label}  ·  arrows: child → parent",
                 fontsize=10, pad=8)

    fig.suptitle(
        f"MERW Centrality \\& Routing Tree  $\\cdot$  {graph_type.upper()} graph  $\\cdot$  "
        f"$N={N}$  $\\cdot$  $\\lambda_1={lam:.4f}$",
        fontsize=13, y=1.02)

    out = MERW_VIZ_DIR / f"merw_tree_{_tag(graph_type, N, p)}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    print(f"\nMERW summary  (N={N}, graph={graph_type})")
    print(f"  λ₁ = {lam:.6f}")
    for h in hubs:
        tag = " (primary)" if h == hub else ""
        print(f"  Hub{tag}: node {h}  ψ²={pi[h]:.4f}  degree={degree[h]}")
    print(f"  Tree depth D = {tree_depth}")
    top = np.argsort(pi)[::-1][:5]
    print("  Top-5 nodes by ψ²:")
    for r, n in enumerate(top):
        print(f"    {r+1}. node {n:3d}  ψ²={pi[n]:.4f}  "
              f"deg={degree[n]}  depth={depth[n]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=("ba", "er", "barbell", "clustered", "grid", "cycle", "lollipop", "chain", "star"), default="ba")
    p.add_argument("--N",     type=int,   default=20)
    p.add_argument("--p-er",  type=float, default=None,
                   help="ER edge probability override (default: 2.5*ln(N)/N; ignored for other graphs)")
    p.add_argument("--seed",  type=int,   default=0)
    p.add_argument("--mode", choices=("compute", "plot", "all"), default="all",
                   help="compute: run MERW + layout and save data/*.csv only; "
                        "plot: render the figure from existing CSVs; "
                        "all: compute then plot (default)")
    args = p.parse_args()

    if args.mode == "plot":
        data, G = load_merw_data(args.graph, args.N, p=args.p_er)
        plot_merw(data, G, p=args.p_er)
        return

    clusters = None
    if args.graph == "ba":
        G = make_ba_graph(args.N, m=2, seed=args.seed)
    elif args.graph == "er":
        G = make_er_graph(args.N, seed=args.seed, p=args.p_er)
    elif args.graph == "barbell":
        G = make_barbell_graph(args.N, seed=args.seed)
    elif args.graph == "grid":
        G = make_grid_graph(args.N, seed=args.seed)
    elif args.graph == "cycle":
        G = make_cycle_graph(args.N)
    elif args.graph == "lollipop":
        G = make_lollipop_graph(args.N, seed=args.seed)
    elif args.graph == "chain":
        G = make_chain_graph(args.N, seed=args.seed)
    elif args.graph == "star":
        G = make_star_graph(args.N, seed=args.seed)
    else:  # clustered
        G, clusters = make_clustered_graph(args.N, seed=args.seed)

    data = compute_merw_data(G, args.graph, args.seed, clusters=clusters)
    save_merw_csv(data, G, p=args.p_er)
    if args.mode == "compute":
        return
    plot_merw(data, G, p=args.p_er)


if __name__ == "__main__":
    main()
