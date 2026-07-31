# Reproducing the numerical experiments

Every figure and quoted number in the paper and the supplementary material
comes from the scripts in this directory. They are self-contained: each
depends only on NumPy, NetworkX and Matplotlib, imports nothing from the
others, and writes its output beside itself, so the directory can be unpacked
anywhere and run as-is.

```bash
pip install -r requirements.txt
python reproduce_paper.py
```

That reruns every experiment and redraws all five figures. Useful variants:

```bash
python reproduce_paper.py --plot-only    # redraw from existing CSVs
python reproduce_paper.py --only main    # just the three main-paper figures
python reproduce_paper.py --only supp    # just the two supplementary panels
```

## Figure → script → paper location

| Figure | Script | Paper location |
|---|---|---|
| `results/Regret/regret_min_er_N100_K5_T5000.pdf` | `regret_min.py` | main paper, Fig. 1 |
| `results/BAI/best_arm_id_er_K10.pdf` | `best_arm_id.py` | main paper, Fig. 2 |
| `results/FaultTolerance/fault_tolerance_er_N20_K10_T2000_at500-1000-1500.pdf` | `fault_tolerance.py` | main paper, Fig. 3 |
| `results/supplementary/bai_topologies.pdf` | `supplementary_figures.py` | supplementary |
| `results/supplementary/regret_topologies.pdf` | `supplementary_figures.py` | supplementary |

`supplementary_figures.py` computes nothing of its own; it reads the CSVs the
regret and BAI sweeps leave in `data/` and arranges them into per-topology
panels, which is why `reproduce_paper.py` runs those sweeps over all four
topologies before calling it.

## Running the experiments individually

Each experiment script has three modes:

| Mode | Effect |
|---|---|
| `--mode compute` | run the experiment, write `data/*.csv`, draw nothing |
| `--mode plot` | redraw the figure from an existing CSV, recompute nothing |
| `--mode all` | compute then plot (default) |

Separating them means the expensive Monte Carlo runs happen once; the figures
can then be restyled from the committed CSVs for free.

Run these from inside this directory:

```bash
python regret_min.py      --graph er --N 100 --K 5 --T 5000 --n-runs 50 --p-er 0.5
python best_arm_id.py     --graph er --K 10 --bai-runs 50 --delta 0.05 --p-er 0.5
python fault_tolerance.py --graph er --N 20 --p 0.5 --K 10 --T 2000 \
                          --fail-at 500 1000 1500 --seed 2
```

Swap `--graph er` for `ba`, `grid` or `barbell` to get the other panels of the
supplementary figures.

## Outputs

```
data/      one CSV per experiment: the raw means and spreads
results/   the figures, under BAI/, Regret/, FaultTolerance/ and supplementary/
```

Both directories are created on first run.

## Runtime

Measured on an Apple M1 Pro (8 cores, CPU only; no GPU is used anywhere).
The regret experiment parallelizes across cores, the others are fast enough
not to need it.

| Experiment | Wall clock |
|---|---|
| Best arm identification (5 sizes × 50 runs) | ~7 s |
| Fault tolerance | ~3 s |
| Regret minimization (N=100, T=5000, 50 runs) | ~70 s |

A full sweep over all four topologies is a few minutes.


