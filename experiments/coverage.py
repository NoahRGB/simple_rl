# NoahRGB 11/2026
#
# gridworld exploration experiments train one or more agents over several seeds,
# then record and compare how much of the gridworld each one covered
#
#   python experiments/coverage.py run --algos density disagreement --seeds 0 1 2 --title four_rooms \
#       env.kwargs.layout=four_rooms algo.total_timesteps=5000
#   python experiments/coverage.py compare results/logs/<run_a> results/logs/<run_b> ... --out results/experiments/mine
#
# anything containing "=" is passed through as a hydra override to every run
#


import sys, json, argparse, traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)) # lets simple_rl be imported

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from simple_rl.core.run import start


CONFIG_DIR = REPO / "simple_rl" / "configs"
LOG_DIR = REPO / "results" / "logs"
EXPERIMENT_DIR = REPO / "results" / "experiments"

# colours
BAR = "#2a78d6"
SEED_DOT = "#104281"
WALL = "#2b2b2b"
UNVISITED = "#a3a19b"
VISITS_CMAP = LinearSegmentedColormap.from_list("visits", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def coverage_stats(visits, walls):
    # compute the coverage and entropy based on visits/wlalls
    free = ~walls
    visited = (visits > 0) & free

    # entropy of the visit distribution over free cells, normalised so 1 = perfectly uniform
    p = visits[free] / max(visits[free].sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(free.sum()))

    return {
        "coverage": float(visited.sum() / free.sum()),
        "cells_visited": int(visited.sum()),
        "free_cells": int(free.sum()),
        "total_visits": int(visits.sum()),
        "visit_entropy": entropy,
    }


def save_coverage(run_dir, env, algo, seed):
    # save alg/seed details + coverage stats to rundir

    grids = env.unwrapped
    assert hasattr(grids[0], "visit_counts"), "coverage experiments need the gridworld env (env=gridworld)"

    visits = sum(grid.visit_counts for grid in grids) # summed over all sub envs in a vectorised envv
    walls = np.asarray(grids[0].walls, dtype=bool)

    np.save(run_dir / "visit_counts.npy", visits)
    np.save(run_dir / "walls.npy", walls)

    stats = {"algo": algo, "seed": seed, **coverage_stats(visits, walls)}
    (run_dir / "coverage.json").write_text(json.dumps(stats, indent=2))
    return stats


def make_cfg(algo, seed, title, notes, overrides):
    # create the hydra confgi based on the params

    defaults = []
    if not any(o.startswith("env=") for o in overrides):
        defaults.append("env=gridworld")
    if not any(o.startswith("notes=") for o in overrides):
        defaults.append(f"notes={notes}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose("config", overrides=[*defaults, f"algo={algo}", f"seed={seed}", f"title={title}", *overrides])


def run(args):
    exp_name = f"{args.title}_{timestamp()}"
    run_dirs = []

    # for every algo/seed combo
    for algo in args.algos:
        for seed in args.seeds:

            # make title, configg, rundir, etc. as normal
            title = f"{args.title}_{algo}_s{seed}"
            cfg = make_cfg(algo, seed, title, f"coverage experiment {exp_name}", args.overrides)

            run_dir = LOG_DIR / f"{title}_{timestamp()}"
            (run_dir / ".hydra").mkdir(parents=True)
            OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")

            # actually train the alg
            print(f"\n=== {algo} seed {seed} -> {run_dir}")
            try:
                agent, env = start(cfg, run_dir=run_dir)
            except Exception:
                traceback.print_exc()
                continue

            # save/print coverage stats
            stats = save_coverage(run_dir, env, algo, seed)
            print(f"coverage {stats['coverage']:.1%} ({stats['cells_visited']}/{stats['free_cells']} cells)")
            run_dirs.append(run_dir)

            del agent, env
            plt.close("all")

    if not run_dirs:
        print("no runs finished")
        return

    # make final dir, save completed runs, make comparison plots
    out_dir = EXPERIMENT_DIR / exp_name
    out_dir.mkdir(parents=True)
    (out_dir / "runs.txt").write_text("\n".join(str(d) for d in run_dirs) + "\n")
    compare(run_dirs, out_dir)


def load_run(run_dir):
    # read all details/save d stats for a runn

    run_dir = Path(run_dir)
    return {
        **json.loads((run_dir / "coverage.json").read_text()),
        "dir": str(run_dir),
        "visits": np.load(run_dir / "visit_counts.npy"),
        "walls": np.load(run_dir / "walls.npy"),
    }


def plot_coverage(groups):
    # plot barchart of coverage

    labels = list(groups)
    fig, ax = plt.subplots(figsize=(max(1.6 * len(labels) + 2, 4.5), 4))

    for i, label in enumerate(labels):
        values = [r["coverage"] for r in groups[label]]
        ax.bar(i, np.mean(values), width=0.6, color=BAR, zorder=2)
        # ax.scatter(np.full(len(values), i), values, s=28, color=SEED_DOT, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(f"{np.mean(values):.1%}", (i, np.mean(values)), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.8, len(labels) - 0.2)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("free cells visited")
    ax.set_title("mean coverage per agent", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_visits(groups):
    # plot coverage for every given run as a heatmap over the gridworld

    labels = list(groups)
    cols = max(len(runs) for runs in groups.values())
    vmax = max(np.log1p(r["visits"]).max() for runs in groups.values() for r in runs)

    cmap = VISITS_CMAP.copy()
    cmap.set_bad(UNVISITED)

    fig, axes = plt.subplots(len(labels), cols, figsize=(2.6 * cols + 1, 2.6 * len(labels)), squeeze=False)
    for row, label in enumerate(labels):
        runs = sorted(groups[label], key=lambda r: r["seed"])
        for col in range(cols):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col >= len(runs):
                ax.axis("off")
                continue

            r = runs[col]
            heat = np.ma.masked_where((r["visits"] == 0) | r["walls"], np.log1p(r["visits"]))
            im = ax.imshow(heat, cmap=cmap, vmin=0, vmax=vmax)
            ax.imshow(np.ma.masked_where(~r["walls"], r["walls"]), cmap=matplotlib.colors.ListedColormap([WALL]))
            ax.set_title(f"seed {r['seed']}  ({r['coverage']:.0%})", fontsize=9)
            if col == 0:
                ax.set_ylabel(label)

    fig.colorbar(im, ax=axes, label="log(1 + visits)", shrink=0.8)
    fig.suptitle("visit counts", fontsize=10)
    return fig


def compare(run_dirs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for d in run_dirs:
        r = load_run(d)
        groups.setdefault(r["algo"], []).append(r)

    summary = {}
    print(f"\n{'agent':<24}{'seeds':>6}{'coverage':>18}{'visit entropy':>18}")
    for label, runs in groups.items():
        cov = np.array([r["coverage"] for r in runs])
        ent = np.array([r["visit_entropy"] for r in runs])
        summary[label] = {
            "seeds": [r["seed"] for r in runs],
            "coverage_mean": float(cov.mean()), "coverage_std": float(cov.std()),
            "visit_entropy_mean": float(ent.mean()), "visit_entropy_std": float(ent.std()),
            "runs": [r["dir"] for r in runs],
        }
        print(f"{label:<24}{len(runs):>6}{f'{cov.mean():.1%} ± {cov.std():.1%}':>18}{f'{ent.mean():.3f} ± {ent.std():.3f}':>18}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_coverage(groups).savefig(out_dir / "coverage.png", dpi=150, bbox_inches="tight")
    plot_visits(groups).savefig(out_dir / "visits.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nwrote summary.json, coverage.png, visits.png to {out_dir}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="train agents, save coverage per run, then compare them")
    p_run.add_argument("--algos", nargs="+", required=True)
    p_run.add_argument("--seeds", nargs="+", type=int, default=[0])
    p_run.add_argument("--title", default="coverage")
    p_run.add_argument("overrides", nargs="*", help="hydra overrides applied to every run, e.g. algo.total_timesteps=5000")

    p_cmp = sub.add_parser("compare", help="compare finished runs from their saved coverage files")
    p_cmp.add_argument("run_dirs", nargs="+")
    p_cmp.add_argument("--out", default=None, help="output dir (default results/experiments/compare_<timestamp>)")

    args = parser.parse_args()

    if args.cmd == "run":
        args.overrides += [a for a in args.algos if "=" in a]
        args.algos = [a for a in args.algos if "=" not in a]
        run(args)
    else:
        compare(args.run_dirs, args.out or EXPERIMENT_DIR / f"compare_{timestamp()}")


if __name__ == "__main__":
    main()
