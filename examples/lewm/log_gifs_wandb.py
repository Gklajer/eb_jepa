"""Upload LeWM rollout GIFs (+ validation metrics) to a single wandb run.

Scans <root>/*/viz/*.gif and <root>/logs/val_*.out, logs each GIF as a
wandb.Video and the parsed probing/rollout metrics as a wandb.Table, so all
variants (mtp 1/2/3/4, fixes) sit on one comparison dashboard.

    PYTHONPATH=$PWD python -m examples.lewm.log_gifs_wandb --root <lewm_train dir>
"""
from __future__ import annotations

import glob
import os
import re

import fire
import wandb


def parse_results(root):
    """ckpt-dir -> dict of val metrics, from the RESULT lines in logs/val_*.out."""
    res = {}
    for f in glob.glob(os.path.join(root, "logs", "val_*.out")):
        for line in open(f):
            if not line.startswith("RESULT"):
                continue
            kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
            m = re.search(r"run[\w]*", kv.get("ckpt", ""))
            d = os.path.basename(os.path.dirname(kv.get("ckpt", "").replace("/latest.pth.tar", "")))
            res[d] = kv
    return res


def run(root: str, project: str = "lewm", name: str = "lewm_gifs_eval"):
    wandb.init(project=project, name=name)
    results = parse_results(root)
    gifs = sorted(glob.glob(os.path.join(root, "*", "viz", "*.gif")))
    table = wandb.Table(columns=["variant", "gif", "feature", "lin_r", "mlp_r",
                                 "mlp_mse", "rollout_mse", "verdict"])
    for g in gifs:
        variant = g.split("/")[-3]  # run dir name
        wandb.log({f"rollout/{variant}": wandb.Video(g, format="gif", caption=variant)})
        r = results.get(variant, {})
        table.add_data(variant, g, r.get("feature", "-"), r.get("lin_r", r.get("pearson", "-")),
                       r.get("mlp_r", "-"), r.get("mlp_mse", r.get("pos_mse", "-")),
                       r.get("rollout_mse", "-"), r.get("verdict", "-"))
        print(f"logged {variant}: {g}")
    wandb.log({"eval/summary": table})
    wandb.finish()
    print(f"done: {len(gifs)} gifs logged to wandb project '{project}'")


if __name__ == "__main__":
    fire.Fire(run)
