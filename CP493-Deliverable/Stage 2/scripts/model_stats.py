"""
model_stats.py  -  Static per-model resource stats (no GPU, no model run needed).

Reports the two resource metrics the professor asks for that can be measured
WITHOUT running the model:
    - parameter count   (summed from the checkpoint's tensors)
    - model size (MB)   (checkpoint file size on disk)

The other three the professor asks for - inference time, throughput, peak GPU
memory - are RUNTIME quantities and are captured in the notebook around the
actual eval run (see the timing + nvidia-smi cells). This script covers the two
static ones and appends them to a shared CSV so every model ends up in one table.

Parameter count is read straight from the checkpoint state_dict, so it works for
any model without needing its class definition. Common checkpoint wrappers
({'state_dict': ...}, {'model': ...}, {'params': ...}, {'net': ...}) are unwrapped.

USAGE:
    python model_stats.py --ckpt /path/RainDrop_DiT_ddpm.pth.tar --name DiT \
        --out_csv /kaggle/working/metrics/resource_static.csv
"""

import os
import csv
import argparse

import torch


def unwrap(obj):
    """Descend common wrapper dicts to reach the actual tensor state_dict."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "params", "net", "G", "generator"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def count_params(state):
    total = 0
    for v in state.values():
        if torch.is_tensor(v):
            total += v.numel()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the checkpoint file")
    ap.add_argument("--name", required=True, help="model label for the table")
    ap.add_argument("--out_csv", default="metrics/resource_static.csv")
    args = ap.parse_args()

    size_mb = os.path.getsize(args.ckpt) / (1024 * 1024)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = unwrap(ckpt)
    if not isinstance(state, dict):
        raise SystemExit(f"Could not find a state_dict inside {args.ckpt}")
    n_params = count_params(state)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    new = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "param_count", "param_count_millions", "model_size_MB"])
        w.writerow([args.name, n_params, round(n_params / 1e6, 3), round(size_mb, 2)])

    print(f"{args.name}: params={n_params:,} ({n_params/1e6:.2f}M)  size={size_mb:.2f} MB")
    print(f"appended to {args.out_csv}")


if __name__ == "__main__":
    main()
