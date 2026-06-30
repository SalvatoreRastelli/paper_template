"""
Relay cycle visualization.

Five sub-panels arranged as a single figure, one per round of the cycle
for a depth-2 tree (D=2, cycle = 2D+1 = 5 rounds):

  t=0  Pull:          every node pulls its arm locally. No messages yet.
  t=1  Uplink hop 1:  every node sends (a_i, r_i) one hop toward hub.
  t=2  Uplink hop 2:  leaves silent; mid-nodes relay packet to hub.
  t=3  Downlink hop 1: hub runs UCB, sends snapshot one hop down to mids.
  t=4  Downlink hop 2: mids forward snapshot to their leaves.

Topology: hub at top, two intermediate nodes m1 and m2 (depth 1),
each with two leaves (L1/L2 under m1, L3/L4 under m2).

Usage:
    python visualize_cycle.py
"""

import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
plt.style.use(Path(__file__).resolve().parent / "merw.mplstyle")

RESULTS_DIR  = Path(__file__).resolve().parent.parent / "paper" / "results"
MERW_VIZ_DIR = RESULTS_DIR / "MERW_visualization"
MERW_VIZ_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "paper" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
NODES_CSV = DATA_DIR / "relay_cycle_nodes.csv"
EDGES_CSV = DATA_DIR / "relay_cycle_edges.csv"

BG         = "#fafaf8"
HUB_COLOR  = "#2166ac"
MID_COLOR  = "#4393c3"
LEAF_COLOR = "#92c5de"
ARROW_UP   = "#d6604d"
ARROW_DOWN = "#4dac26"
SILENT_COL = "#dddddd"
HUB_RING   = "#f4a261"
PULL_COL   = "#9467bd"

# Symmetric tree: hub at top, m1 left, m2 right, two leaves each
NODES = {
    "hub": ( 0.0,  1.0),
    "m1":  (-0.9,  0.0),
    "m2":  ( 0.9,  0.0),
    "L1":  (-1.4, -1.0),
    "L2":  (-0.4, -1.0),
    "L3":  ( 0.4, -1.0),
    "L4":  ( 1.4, -1.0),
}

EDGES = [
    ("m1", "hub"),
    ("m2", "hub"),
    ("L1", "m1"),
    ("L2", "m1"),
    ("L3", "m2"),
    ("L4", "m2"),
]

LABELS = {
    "hub": "hub",
    "m1":  "$m_1$",
    "m2":  "$m_2$",
    "L1":  "$\\ell_1$",
    "L2":  "$\\ell_2$",
    "L3":  "$\\ell_3$",
    "L4":  "$\\ell_4$",
}

DEPTHS = {"hub": 0, "m1": 1, "m2": 1, "L1": 2, "L2": 2, "L3": 2, "L4": 2}


def save_cycle_csv():
    """Persist the fixed relay-cycle topology (node positions/labels/depths,
    edges) to CSV, so the figure can be redrawn from data alone."""
    with open(NODES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "x", "y", "label", "depth"])
        for name, (x, y) in NODES.items():
            writer.writerow([name, x, y, LABELS[name], DEPTHS[name]])

    with open(EDGES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["child", "parent"])
        for child, parent in EDGES:
            writer.writerow([child, parent])

    print(f"  Saved {NODES_CSV}, {EDGES_CSV}")


def load_cycle_csv():
    """Load the topology from CSV into the module-level NODES/EDGES/LABELS/DEPTHS."""
    global NODES, EDGES, LABELS, DEPTHS
    with open(NODES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    NODES  = {r["name"]: (float(r["x"]), float(r["y"])) for r in rows}
    LABELS = {r["name"]: r["label"] for r in rows}
    DEPTHS = {r["name"]: int(r["depth"]) for r in rows}

    with open(EDGES_CSV, newline="") as f:
        EDGES = [(r["child"], r["parent"]) for r in csv.DictReader(f)]


def draw_node(ax, name, state):
    """state: 'hub', 'active', 'silent', 'pulling'"""
    x, y = NODES[name]
    if state == "silent":
        color = SILENT_COL
        ec    = "#bbbbbb"
        ring  = False
    elif name == "hub":
        color = HUB_COLOR
        ec    = "#1a4d7a"
        ring  = (state != "silent")
    elif name in ("m1", "m2"):
        color = MID_COLOR
        ec    = "#2a6090"
        ring  = False
    else:
        color = LEAF_COLOR
        ec    = "#2a6090"
        ring  = False

    if state == "pulling":
        color = PULL_COL
        ec    = "#5a2d80"
        if name == "hub":
            ring = True

    circle = plt.Circle((x, y), 0.20, facecolor=color, edgecolor=ec,
                         linewidth=1.8, zorder=3)
    ax.add_patch(circle)
    if ring:
        r2 = plt.Circle((x, y), 0.30, facecolor="none", edgecolor=HUB_RING,
                         linewidth=2.5, zorder=2)
        ax.add_patch(r2)
    fc = "#333333" if state == "silent" else "white"
    ax.text(x, y, LABELS[name], ha="center", va="center",
            fontsize=9, fontweight="bold", color=fc, zorder=4)


def draw_edges(ax):
    for child, parent in EDGES:
        x0, y0 = NODES[child]
        x1, y1 = NODES[parent]
        ax.plot([x0, x1], [y0, y1], color="#e0e0e0", lw=1.2, zorder=1)


def _arrowprops(color, alpha=1.0, lw=1.8):
    return dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.18", mutation_scale=14)


def msg_arrow(ax, src, dst, color, label="", loffset=(0, 0), alpha=1.0, lw=1.8):
    x0, y0 = NODES[src]
    x1, y1 = NODES[dst]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=_arrowprops(color, alpha=alpha, lw=lw), zorder=5)
    if label and alpha > 0.3:
        mx = (x0 + x1) / 2 + loffset[0]
        my = (y0 + y1) / 2 + loffset[1]
        ax.text(mx, my, label, fontsize=7.5, color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=color,
                          alpha=0.88, lw=0.8))


def pull_label(ax, name, label, offset=(0, 0)):
    x, y = NODES[name]
    ax.text(x + offset[0], y + offset[1], label, fontsize=7.5,
            color=PULL_COL, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec=PULL_COL,
                      alpha=0.85, lw=0.7))


def setup_ax(ax, title, subtitle=""):
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.50, 1.50)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, pad=6, fontweight="bold", color="#222222")
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#555555", style="italic")


def plot_cycle():
    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 6, figsize=(28, 5.5))
    fig.patch.set_facecolor("white")

    # ── t=0: everyone pulls ──────────────────────────────────────────────
    ax = axes[0]
    setup_ax(ax, "t = 0  —  Pull",
             "Every node pulls arm $a_i$ locally. No messages.")
    draw_edges(ax)
    for name in NODES:
        draw_node(ax, name, "pulling")
    for name, off in [("hub",  ( 0.38,  0.0)),
                      ("m1",   (-0.30,  0.26)),
                      ("m2",   ( 0.30,  0.26)),
                      ("L1",   (-0.30,  0.26)),
                      ("L2",   ( 0.30,  0.26)),
                      ("L3",   (-0.30,  0.26)),
                      ("L4",   ( 0.30,  0.26))]:
        pull_label(ax, name, r"$\uparrow a_i$", offset=off)

    # ── t=1: uplink hop 1 (everyone sends one hop up) ───────────────────
    ax = axes[1]
    setup_ax(ax, "t = 1  —  Uplink hop 1",
             "Every node sends $(a_i, r_i)$ one hop toward hub.")
    draw_edges(ax)
    states = {"hub": "silent", "m1": "active", "m2": "active",
              "L1": "active", "L2": "active", "L3": "active", "L4": "active"}
    for name, state in states.items():
        draw_node(ax, name, state)

    msg_arrow(ax, "L1", "m1", ARROW_UP, r"$(a_1,r_1)$", loffset=(-0.32,  0.08))
    msg_arrow(ax, "L2", "m1", ARROW_UP, r"$(a_2,r_2)$", loffset=( 0.18,  0.10))
    msg_arrow(ax, "L3", "m2", ARROW_UP, r"$(a_3,r_3)$", loffset=(-0.18,  0.10))
    msg_arrow(ax, "L4", "m2", ARROW_UP, r"$(a_4,r_4)$", loffset=( 0.32,  0.08))
    msg_arrow(ax, "m1", "hub", ARROW_UP, r"$(a_{m_1},r_{m_1})$", loffset=(-0.42,  0.14))
    msg_arrow(ax, "m2", "hub", ARROW_UP, r"$(a_{m_2},r_{m_2})$", loffset=( 0.42,  0.14))

    # ── t=2: uplink hop 2 (leaves silent, mids relay) ───────────────────
    ax = axes[2]
    setup_ax(ax, "t = 2  —  Uplink hop 2",
             "Leaves silent. Mid-nodes relay collected packets to hub.")
    draw_edges(ax)
    states = {"hub": "silent", "m1": "active", "m2": "active",
              "L1": "silent", "L2": "silent", "L3": "silent", "L4": "silent"}
    for name, state in states.items():
        draw_node(ax, name, state)

    msg_arrow(ax, "m1", "hub", ARROW_UP,
              r"$(a_1{+}a_2,\;r_1{+}r_2)$",
              loffset=(-0.44, 0.12))
    msg_arrow(ax, "m2", "hub", ARROW_UP,
              r"$(a_3{+}a_4,\;r_3{+}r_4)$",
              loffset=( 0.44, 0.12))
    # faint ghost arrows from leaves to show packets already passed
    msg_arrow(ax, "L1", "m1", ARROW_UP, "", alpha=0.12, lw=1.0)
    msg_arrow(ax, "L2", "m1", ARROW_UP, "", alpha=0.12, lw=1.0)
    msg_arrow(ax, "L3", "m2", ARROW_UP, "", alpha=0.12, lw=1.0)
    msg_arrow(ax, "L4", "m2", ARROW_UP, "", alpha=0.12, lw=1.0)

    # ── t=3: hub UCB ──────────────────────────────────────────────────────
    ax = axes[3]
    setup_ax(ax, "t = 3  —  Hub UCB",
             r"Hub has all packets. Runs UCB on $(\hat{n},\hat{s})$. All others silent.")
    draw_edges(ax)
    states = {"hub": "hub", "m1": "silent", "m2": "silent",
              "L1": "silent", "L2": "silent", "L3": "silent", "L4": "silent"}
    for name, state in states.items():
        draw_node(ax, name, state)

    hx, hy = NODES["hub"]
    ax.text(hx, hy - 0.42,
            r"$a^\star = \arg\max_k \mathrm{UCB}_k(\hat{n},\hat{s})$",
            ha="center", va="top", fontsize=7.5, color=HUB_COLOR, zorder=6)
    # faint absorbed arrows
    for src in ["m1", "m2"]:
        msg_arrow(ax, src, "hub", ARROW_UP, "", alpha=0.12, lw=1.0)

    # ── t=4: downlink hop 1 (hub sends to mids) ─────────────────────────
    DL = r"$(\hat{n},\hat{s},D)$"
    ax = axes[4]
    setup_ax(ax, "t = 4  —  Downlink hop 1",
             r"Hub sends $(\hat{n},\hat{s},D)$ to mid-nodes.")
    draw_edges(ax)
    states = {"hub": "hub", "m1": "active", "m2": "active",
              "L1": "silent", "L2": "silent", "L3": "silent", "L4": "silent"}
    for name, state in states.items():
        draw_node(ax, name, state)

    msg_arrow(ax, "hub", "m1", ARROW_DOWN, DL, loffset=(-0.42, 0.14))
    msg_arrow(ax, "hub", "m2", ARROW_DOWN, DL, loffset=( 0.42, 0.14))

    # ── t=5: downlink hop 2 (mids forward to leaves) ────────────────────
    ax = axes[5]
    setup_ax(ax, "t = 5  —  Downlink hop 2",
             r"Mid-nodes forward snapshot to leaves. All nodes ready for next cycle.")
    draw_edges(ax)
    states = {"hub": "silent", "m1": "active", "m2": "active",
              "L1": "active", "L2": "active", "L3": "active", "L4": "active"}
    for name, state in states.items():
        draw_node(ax, name, state)

    msg_arrow(ax, "m1", "L1", ARROW_DOWN, DL, loffset=(-0.35,  0.08))
    msg_arrow(ax, "m1", "L2", ARROW_DOWN, DL, loffset=( 0.18,  0.10))
    msg_arrow(ax, "m2", "L3", ARROW_DOWN, DL, loffset=(-0.18,  0.10))
    msg_arrow(ax, "m2", "L4", ARROW_DOWN, DL, loffset=( 0.35,  0.08))
    # faint ghost arrows from hub to mids
    msg_arrow(ax, "hub", "m1", ARROW_DOWN, "", alpha=0.12, lw=1.0)
    msg_arrow(ax, "hub", "m2", ARROW_DOWN, "", alpha=0.12, lw=1.0)

    # ── shared legend ────────────────────────────────────────────────────
    pl   = mpatches.Patch(color=PULL_COL,   label="Pulling arm locally")
    ul   = mpatches.Patch(color=ARROW_UP,   label=r"Uplink packet $(a_i, r_i)$")
    dl   = mpatches.Patch(color=ARROW_DOWN, label=r"Downlink snapshot $(\hat{n},\hat{s},D)$")
    hp   = mpatches.Patch(color=HUB_COLOR,  label="Hub")
    mp   = mpatches.Patch(color=MID_COLOR,  label="Intermediate node")
    lp   = mpatches.Patch(color=LEAF_COLOR, label="Leaf node")
    sp   = mpatches.Patch(color=SILENT_COL, label="Silent this round")

    fig.legend(handles=[hp, mp, lp, sp, pl, ul, dl],
               loc="lower center", ncol=7, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Relay cycle on a depth-2 tree  ($D=2$,  cycle length $= 2D+2 = 6$ rounds)",
        fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    out = MERW_VIZ_DIR / "relay_cycle.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("compute", "plot", "all"), default="all",
                   help="compute: save the fixed topology to data/*.csv only; "
                        "plot: render the figure from existing CSVs; "
                        "all: compute then plot (default)")
    args = p.parse_args()

    if args.mode == "plot":
        load_cycle_csv()
        plot_cycle()
        return

    save_cycle_csv()
    if args.mode == "compute":
        return
    plot_cycle()


if __name__ == "__main__":
    main()
