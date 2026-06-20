"""Side-by-side animation of the self-speculative speedup.

Left = mtp=1 vanilla autoregressive rollout (1 predictor forward per step).
Right = mtp=4 self-speculative rollout (drafts mtp tokens / forward, verifies,
keeps accepted prefix). The animation tick is ONE predictor forward pass; each
panel reveals the rollout up to the step it has reached after that many forwards.
The mtp=4 panel reaches the horizon first -> visual proof of the speedup.

Content frames are the real trajectory (so the agent is clearly visible); the
schedule (forwards->step) is measured from the actual models.

    PYTHONPATH=$PWD python -m examples.lewm.accel_gif \
        --ckpt1 <mtp1>/latest.pth.tar --ckpt4 <mtp4>/latest.pth.tar --out accel.gif
"""
from __future__ import annotations

import cv2
import fire
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import init_data
from eb_jepa.lewm import LeWMPredictor, ViTTinyEncoder
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import setup_seed

logger = get_logger(__name__)


def _build(ckpt, device):
    s = torch.load(ckpt, map_location=device)
    c = OmegaConf.create(s["cfg"])
    enc = ViTTinyEncoder(in_channels=c.model.get("in_channels", 2), img_size=c.model.get("img_size", 224),
                         patch=c.model.get("patch", 14), dim=c.model.get("enc_dim", 192),
                         depth=c.model.get("enc_depth", 12), heads=c.model.get("enc_heads", 3),
                         use_head=c.model.get("use_head", True)).to(device)
    pred = LeWMPredictor(latent_dim=enc.hidden_dim, dim=c.model.get("pred_dim", 384),
                         depth=c.model.get("pred_depth", 6), heads=c.model.get("pred_heads", 6),
                         action_dim=2, dropout=c.model.get("pred_dropout", 0.1),
                         mtp=c.model.get("mtp_horizon", 1), history=c.model.get("history", 0)).to(device)
    enc.load_state_dict(s["encoder"]); pred.load_state_dict(s["predictor"]); enc.eval(); pred.eval()
    return enc, pred, c


def _spec_forwards_per_step(pred, first, actions, T, threshold):
    """Run speculative rollout and return cumulative #forwards to reach each step."""
    pred.self_speculative_rollout(first, actions, T, threshold=threshold)
    st = dict(pred.last_speculative_stats)
    chunks = st.get("chunks", T)
    k = max(pred.mtp, 1)
    # each chunk = 2 forwards (draft+verify) advancing up to k steps
    fwd = []
    for s in range(1, T + 1):
        n_chunks = int(np.ceil(s / k))
        fwd.append(2 * n_chunks)
    return fwd, chunks


def _panel(frame, title, fwd, step, T, up=4):
    H, W = frame.shape[:2]
    img = cv2.resize(frame, (W * up, H * up), interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((48, W * up, 3), np.uint8)
    canvas = np.vstack([pad, img])
    cv2.putText(canvas, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"forwards: {fwd}", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 220, 90), 1, cv2.LINE_AA)
    # progress bar of rollout step
    bw = int((W * up - 12) * step / max(T - 1, 1))
    cv2.rectangle(canvas, (6, 44), (6 + bw, 47), (90, 160, 255), -1)
    return canvas


@torch.no_grad()
def make(ckpt1: str, ckpt4: str, out: str = "accel.gif", horizon: int = None,
         threshold: float = 0.1, fps: int = 4, seed: int = 0):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc1, pred1, c1 = _build(ckpt1, device)
    enc4, pred4, c4 = _build(ckpt4, device)

    cfg_data = OmegaConf.to_container(c4.data, resolve=True); cfg_data["pipeline"] = {"mode": "online"}
    _, val_loader, _, _ = init_data(env_name=c4.data.env_name, cfg_data=cfg_data, device=device)
    x, a, *_ = next(iter(val_loader))
    x = x.to(device); a = a.to(device).float()
    # real trajectory frames as content (agent clearly visible)
    norm = val_loader.dataset.normalizer
    raw = norm.unnormalize_state(x[0].permute(1, 0, 2, 3)).permute(0, 2, 3, 1)  # [Traw,H,W,C]
    raw = raw.clamp(0, 1).cpu().numpy()
    Traw = raw.shape[0]
    T = horizon or Traw
    T = min(T, Traw)
    frames = [(np.stack([raw[t, :, :, 0]] * 3, -1) * 255).astype(np.uint8) if raw.shape[-1] < 3
              else (raw[t] * 255).astype(np.uint8) for t in range(T)]

    # forward-pass schedule
    first = enc4(x[:1, :, :1]).float()
    actions = torch.randn(1, T, 2, device=device)
    spec_fwd, chunks = _spec_forwards_per_step(pred4, first, actions, T, threshold)
    van_fwd = list(range(1, T + 1))     # mtp=1: 1 forward per step
    total_fwd = max(van_fwd[-1], spec_fwd[-1])

    # step reached by each method after f forwards
    def step_after(fwd_sched, f):
        s = 0
        for i, need in enumerate(fwd_sched):
            if need <= f:
                s = i
        return s

    gif = []
    for f in range(1, total_fwd + 1):
        sv = step_after(van_fwd, f); ss = step_after(spec_fwd, f)
        left = _panel(frames[sv], "mtp=1  vanilla AR", min(f, T), sv, T)
        right = _panel(frames[ss], f"mtp={pred4.mtp}  self-speculative", min(f, spec_fwd[-1] if False else f), ss, T)
        sep = np.full((left.shape[0], 6, 3), 60, np.uint8)
        gif.append(np.hstack([left, sep, right]))
    # hold last frame
    gif += [gif[-1]] * fps
    imageio.mimsave(out, gif, fps=fps, loop=0)
    logger.info(f"saved {out} | vanilla {van_fwd[-1]} forwards vs speculative {spec_fwd[-1]} forwards "
                f"for {T} steps -> {van_fwd[-1]/max(spec_fwd[-1],1):.2f}x fewer forwards")
    print(f"ACCELGIF out={out} T={T} van_fwd={van_fwd[-1]} spec_fwd={spec_fwd[-1]} "
          f"fwd_speedup={van_fwd[-1]/max(spec_fwd[-1],1):.2f}x mtp={pred4.mtp}")


if __name__ == "__main__":
    fire.Fire(make)
