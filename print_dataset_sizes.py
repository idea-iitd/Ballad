"""Print train/val/test dataset sizes for each problem."""
import ast
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
PROBLEMS_DIR = ROOT / "problems"


def load_npy_size(path):
    return np.load(path).shape[0]


def load_npz_size_bpp(path):
    return np.load(path)["demands"].shape[0]


def load_pickle_size(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return sum(1 for k in data if k != "l1_bound")


def load_ged_size(path):
    with open(path, "r") as f:
        content = f.read()
    g_pairs_str = content.split("g_pairs: ")[1].split("node_features: ")[0].strip()
    return len(ast.literal_eval(g_pairs_str))


def scan_dir(dataset_dir, pattern, size_fn):
    """Return {mode: [(label, size)]} for files matching glob pattern."""
    results = {"train": [], "val": [], "test": []}
    for path in sorted(dataset_dir.glob(pattern)):
        name = path.stem  # e.g. train50_dataset
        for mode in results:
            if name.startswith(mode):
                label = name  # use full stem as label
                results[mode].append((label, size_fn(path)))
                break
    return results


def print_problem(problem_name, splits):
    print(f"\n{'=' * 60}")
    print(f"  {problem_name}")
    print(f"{'=' * 60}")
    for mode in ("train", "val", "test"):
        entries = splits.get(mode, [])
        if not entries:
            print(f"  {mode:<6}: (none)")
        else:
            for label, size in entries:
                print(f"  {mode:<6}: {label} — {size} instances")


# ── TSP constructive ──────────────────────────────────────────────────────────
tsp_dir = PROBLEMS_DIR / "tsp_constructive" / "dataset"
tsp_splits = scan_dir(tsp_dir, "*_dataset.npy", load_npy_size)
print_problem("tsp_constructive", tsp_splits)

# ── CVRP ACO ─────────────────────────────────────────────────────────────────
cvrp_dir = PROBLEMS_DIR / "cvrp_aco" / "dataset"
cvrp_splits = scan_dir(cvrp_dir, "*_dataset.npy", load_npy_size)
print_problem("cvrp_aco", cvrp_splits)

# ── BPP offline ACO ───────────────────────────────────────────────────────────
bpp_off_dir = PROBLEMS_DIR / "bpp_offline_aco" / "dataset"
bpp_off_splits = scan_dir(bpp_off_dir, "*_dataset.npz", load_npz_size_bpp)
print_problem("bpp_offline_aco", bpp_off_splits)

# ── BPP online ────────────────────────────────────────────────────────────────
bpp_on_dir = PROBLEMS_DIR / "bpp_online" / "dataset"
bpp_on_splits = {"train": [], "val": [], "test": []}
for path in sorted(bpp_on_dir.glob("weibull_*.pickle")):
    name = path.stem  # e.g. weibull_5k_train1
    for mode in bpp_on_splits:
        if f"_{mode}" in name:
            bpp_on_splits[mode].append((name, load_pickle_size(path)))
            break
print_problem("bpp_online", bpp_on_splits)

# ── GED ───────────────────────────────────────────────────────────────────────
ged_data_dir = PROBLEMS_DIR / "ged" / "dataset" / "data"
ged_splits = {"train": [], "val": [], "test": []}

for path in sorted((ged_data_dir / "train").glob("*.txt")):
    ged_splits["train"].append((path.stem, load_ged_size(path)))

for path in sorted((ged_data_dir / "val").glob("*.txt")):
    ged_splits["val"].append((path.stem, load_ged_size(path)))

for path in sorted((ged_data_dir / "test").glob("*.txt")):
    ged_splits["test"].append((path.stem, load_ged_size(path)))

print_problem("ged", ged_splits)

print()
