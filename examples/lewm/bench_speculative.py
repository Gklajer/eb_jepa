"""Clean figure: self-speculative latent rollout accelerates vs vanilla AR rollout.

Vanilla does one predictor forward per step (nsteps forwards). Self-speculation
drafts up to `mtp` latent tokens in one forward and verifies them in one more, so
an accepted chunk advances several steps for ~2 forwards. We sweep the rollout
horizon, time both (warmed up, cuda-synced, averaged), and save a 2-panel figure:
  (left)  wall-clock vs horizon: vanilla vs speculative
  (right) speedup vs horizon
plus a printed summary line per checkpoint.

    PYTHONPATH=$PWD python -m examples.lewm.bench_speculative \
        --ckpt <mtp-run>/latest.pth.tar --out fig.pdf
"""
from __future__ import annotations

import time

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import init_data
from eb_jepa.lewm import LeWMPredictor, ViTTinyEncoder
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import setup_seed

logger = get_logger(__name__)


def _time(fn, repeats, device):
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def _build(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    cfg = OmegaConf.create(state["cfg"])
    enc = ViTTinyEncoder(in_channels=cfg.model.get("in_channels", 2),
                         img_size=cfg.model.get("img_size", 224), patch=cfg.model.get("patch", 14),
                         dim=cfg.model.get("enc_dim", 192), depth=cfg.model.get("enc_depth", 12),
                         heads=cfg.model.get("enc_heads", 3),
                         use_head=cfg.model.get("use_head", True)).to(device)
    pred = LeWMPredictor(latent_dim=enc.hidden_dim, dim=cfg.model.get("pred_dim", 384),
                         depth=cfg.model.get("pred_depth", 6), heads=cfg.model.get("pred_heads", 6),
                         action_dim=2, dropout=cfg.model.get("pred_dropout", 0.1),
                         mtp=cfg.model.get("mtp_horizon", 1), history=cfg.model.get("history", 0)).to(device)
    enc.load_state_dict(state["encoder"]); pred.load_state_dict(state["predictor"])
    enc.eval(); pred.eval()
    return enc, pred, cfg


@torch.no_grad()
def bench(ckpt: str, out: str = None, horizons=(8, 16, 24, 32, 48, 64),
          repeats: int = 30, threshold: float = 0.1, metric: str = "normalized_mse",
          batch: int = 16, seed: int = 0):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, pred, cfg = _build(ckpt, device)
    mtp = cfg.model.get("mtp_horizon", 1)
    out = out or str(__import__("pathlib").Path(ckpt).parent / "speculative_speedup.pdf")

    cfg_data = OmegaConf.to_container(cfg.data, resolve=True); cfg_data["pipeline"] = {"mode": "online"}
    _, val_loader, _, _ = init_data(env_name=cfg.data.env_name, cfg_data=cfg_data, device=device)
    x, *_ = next(iter(val_loader))
    first = enc(x.to(device)[:batch, :, :1]).float()             # [B,1,D]

    horizons = list(horizons)
    t_van, t_spec, speedup, accepted = [], [], [], []
    for n in horizons:
        actions = torch.randn(first.size(0), n, 2, device=device)
        tv = _time(lambda: pred.rollout(first, actions, n), repeats, device)
        ts = _time(lambda: pred.self_speculative_rollout(
            first, actions, n, threshold=threshold, distance_metric=metric), repeats, device)
        st = dict(pred.last_speculative_stats)
        t_van.append(tv * 1e3); t_spec.append(ts * 1e3); speedup.append(tv / max(ts, 1e-9))
        acc = st.get("mean_accepted", st.get("accepted", "-"))
        accepted.append(acc)
        logger.info(f"  n={n:3d}: vanilla {tv*1e3:7.2f}ms  spec {ts*1e3:7.2f}ms  "
                    f"speedup {tv/max(ts,1e-9):.2f}x  chunks={st.get('chunks','-')}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    ax1.plot(horizons, t_van, "o-", label="vanilla AR", color="#c0392b")
    ax1.plot(horizons, t_spec, "s--", label=f"self-speculative (mtp={mtp})", color="#2980b9")
    ax1.set_xlabel("rollout horizon (steps)"); ax1.set_ylabel("wall-clock (ms)")
    ax1.set_title("Latent rollout time"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(horizons, speedup, "^-", color="#27ae60")
    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("rollout horizon (steps)"); ax2.set_ylabel("speedup (×)")
    ax2.set_title(f"Speedup (mean {sum(speedup)/len(speedup):.2f}×)"); ax2.grid(alpha=0.3)
    fig.suptitle(f"MTP self-speculative acceleration — mtp={mtp}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    png = out.replace(".pdf", ".png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved figure: {out} / {png}")
    print(f"BENCH ckpt={ckpt} mtp={mtp} mean_speedup={sum(speedup)/len(speedup):.2f}x "
          f"max_speedup={max(speedup):.2f}x fig={png}")


if __name__ == "__main__":
    fire.Fire(bench)
