"""Validation for a trained LeWM checkpoint (no world-model retraining).

Quantifies what the FROZEN encoder/predictor learned, on a held-out val split:
  - Position probing (paper Tab. 3): fit an XY probe on TRAIN latents, report
    position MSE + Pearson r on VAL. r≈0 / MSE≈1 ⇒ latent ignores the agent
    (slow-feature collapse); r→1 / MSE→0 ⇒ position is encoded.
  - Latent rollout MSE on VAL (predictor accuracy). ≈0 together with probe MSE≈1
    is the collapse signature (trivial constant-latent prediction).

    PYTHONPATH=$PWD python -m examples.lewm.val_lewm --ckpt <run>/latest.pth.tar
"""

from __future__ import annotations

from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.optim import AdamW

from eb_jepa.datasets.utils import init_data
from eb_jepa.lewm import LeWMPredictor, ViTTinyEncoder
from eb_jepa.logging import get_logger
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_config, setup_seed

logger = get_logger(__name__)


def subtraj(x, a, loc, nf, skip):
    idx = [i * skip for i in range(nf)]
    frames = x[:, :, idx]
    blocks = torch.stack([a[:, :, i*skip:(i+1)*skip].sum(2) for i in range(nf-1)], 1)
    return frames, blocks, loc[:, :, idx]


@torch.no_grad()
def encode_set(enc, loader, device, nf, skip, n_batches):
    feats, locs = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, a, loc, *_ = next(it)
        except StopIteration:
            break
        f, _, lc = subtraj(x.to(device), a.to(device).float(), loc.to(device), nf, skip)
        with autocast("cuda", dtype=torch.bfloat16):
            z = enc(f)
        feats.append(z.float().cpu()); locs.append(lc.cpu())
    return torch.cat(feats), torch.cat(locs)  # [N,nf,D], [N,2,nf]


def pearson(pred, tgt):  # [M,2] each -> mean r over dims
    rs = []
    for d in range(pred.size(1)):
        p, t = pred[:, d], tgt[:, d]
        p, t = p - p.mean(), t - t.mean()
        rs.append((p * t).sum() / (p.norm() * t.norm() + 1e-8))
    return torch.stack(rs).mean().item()


def val(ckpt: str, nf: int = 4, skip: int = 5, train_batches: int = 40,
        val_batches: int = 20, probe_iters: int = 2000, seed: int = 0):
    setup_seed(seed)
    device = torch.device("cuda")
    state = torch.load(ckpt, map_location=device)
    cfg = OmegaConf.create(state["cfg"])
    enc = ViTTinyEncoder(in_channels=cfg.model.get("in_channels", 2),
                         img_size=cfg.model.get("img_size", 224), patch=cfg.model.get("patch", 14),
                         dim=cfg.model.get("enc_dim", 192), depth=cfg.model.get("enc_depth", 12),
                         heads=cfg.model.get("enc_heads", 3)).to(device)
    pred = LeWMPredictor(latent_dim=enc.hidden_dim, dim=cfg.model.get("pred_dim", 384),
                         depth=cfg.model.get("pred_depth", 6), heads=cfg.model.get("pred_heads", 6),
                         action_dim=2, dropout=cfg.model.get("pred_dropout", 0.1),
                         mtp=cfg.model.get("mtp_horizon", 1)).to(device)
    enc.load_state_dict(state["encoder"]); pred.load_state_dict(state["predictor"])
    enc.eval(); pred.eval()
    logger.info(f"== VAL {ckpt} (epoch {state.get('epoch')}, mtp={cfg.model.get('mtp_horizon',1)}) ==")

    cfg_data = OmegaConf.to_container(cfg.data, resolve=True); cfg_data["pipeline"] = {"mode": "online"}
    loader, val_loader, _, _ = init_data(env_name=cfg.data.env_name, cfg_data=cfg_data, device=device)

    # ---- probing: fit on train latents, evaluate on val ----
    ztr, ltr = encode_set(enc, loader, device, nf, skip, train_batches)
    zva, lva = encode_set(enc, val_loader, device, nf, skip, val_batches)
    probe = MLPXYHead(enc.hidden_dim).to(device)
    opt = AdamW(probe.parameters(), lr=1e-3); probe.train()
    N = ztr.size(0)
    for i in range(probe_iters):
        bi = torch.randint(0, N, (min(256, N),))
        z5 = ztr[bi].to(device).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(probe(z5), ltr[bi].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        z5 = zva.to(device).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        pva = probe(z5).permute(0, 2, 1).reshape(-1, 2).cpu()   # [M,2]
    tva = lva.permute(0, 2, 1).reshape(-1, 2)
    pos_mse = torch.nn.functional.mse_loss(pva, tva).item()
    r = pearson(pva, tva)

    # ---- latent rollout MSE on val ----
    with torch.no_grad():
        n = nf - 1
        roll_mse = []
        it = iter(val_loader)
        for _ in range(val_batches):
            try:
                x, a, loc, *_ = next(it)
            except StopIteration:
                break
            f, blocks, _ = subtraj(x.to(device), a.to(device).float(), loc.to(device), nf, skip)
            with autocast("cuda", dtype=torch.bfloat16):
                z = enc(f).float()
            pr = pred.rollout(z[:, :1], blocks, n)
            roll_mse.append(((pr - z[:, 1:]) ** 2).mean().item())
        roll = sum(roll_mse) / max(len(roll_mse), 1)

    logger.info("================ VALIDATION ================")
    logger.info(f"  position probe  MSE(val) = {pos_mse:.4f}   Pearson r = {r:.4f}")
    logger.info(f"  latent rollout  MSE(val) = {roll:.4f}")
    verdict = "COLLAPSED (latent ignores agent)" if (pos_mse > 0.6 or r < 0.4) else "OK (position encoded)"
    logger.info(f"  verdict: {verdict}")
    logger.info("============================================")
    print(f"RESULT ckpt={ckpt} pos_mse={pos_mse:.4f} pearson={r:.4f} rollout_mse={roll:.4f} verdict={verdict}")


if __name__ == "__main__":
    fire.Fire(val)
