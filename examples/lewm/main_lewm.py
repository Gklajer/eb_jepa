"""LeWorldModel (LeWM) training on two_rooms — arXiv:2603.19312 reproduction.

End-to-end JEPA world model from pixels with two loss terms (next-embedding MSE
+ λ·SIGReg), no EMA / stop-grad / pretrained encoder. TwoRoom recipe (Appendix D):
frame-skip 5, sub-trajectories of 4 frames, batch 128, 10 epochs.

Run (from the gklajer clone, with it on the path so eb_jepa.lewm resolves):
    cd /lustre/work/vivatech-yentlteam/gklajer/eb_jepa
    PYTHONPATH=$PWD python -m examples.lewm.main_lewm \
        --fname examples/lewm/cfgs/lewm_two_rooms.yaml --folder <out>
"""

from __future__ import annotations

import os
from pathlib import Path
from time import time

import fire
import torch
import wandb
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.datasets.utils import init_data
from eb_jepa.lewm import LeWMPredictor, ViTTinyEncoder, lewm_loss, lewm_mtp_loss
from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.training_utils import load_config, setup_seed

logger = get_logger(__name__)


def make_subtraj(x, a, num_frames=4, skip=5):
    """[B,C,T,H,W],[B,A,T] -> frames [B,C,nf,H,W] at t=0,skip,..; action blocks
    [B, nf-1, A] each summing `skip` consecutive actions between kept frames."""
    idx = [i * skip for i in range(num_frames)]
    assert idx[-1] < x.size(2), f"need T>{idx[-1]}, got {x.size(2)}"
    frames = x[:, :, idx]                                   # [B,C,nf,H,W]
    blocks = torch.stack(
        [a[:, :, i * skip:(i + 1) * skip].sum(dim=2) for i in range(num_frames - 1)],
        dim=1,
    )                                                       # [B, nf-1, A]
    return frames, blocks


def run(
    fname: str = "examples/lewm/cfgs/lewm_two_rooms.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)
    setup_seed(int(cfg.meta.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = Path(folder or cfg.meta.get("folder", "lewm_run"))
    folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {folder}")
    OmegaConf.save(cfg, folder / "config.yaml")

    wandb_run = None
    if cfg.logging.get("log_wandb", False):
        wandb_run = wandb.init(
            project=cfg.logging.get("wandb_project", "lewm"),
            name=cfg.logging.get("wandb_name", "lewm_two_rooms"),
            config=OmegaConf.to_container(cfg, resolve=True), dir=str(folder),
        )

    # ---- data (two_rooms, online) ----
    loader, val_loader, data_config, _ = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device,
    )

    # ---- model: trainable ViT-Tiny encoder + AdaLN causal predictor ----
    enc = ViTTinyEncoder(
        in_channels=cfg.model.get("in_channels", 2),
        img_size=cfg.model.get("img_size", 224),
        patch=cfg.model.get("patch", 14),
        dim=cfg.model.get("enc_dim", 192),
        depth=cfg.model.get("enc_depth", 12),
        heads=cfg.model.get("enc_heads", 3),
    ).to(device)
    mtp = cfg.model.get("mtp_horizon", 1)
    pred = LeWMPredictor(
        latent_dim=enc.hidden_dim,
        dim=cfg.model.get("pred_dim", 384),
        depth=cfg.model.get("pred_depth", 6),
        heads=cfg.model.get("pred_heads", 6),
        action_dim=2,
        dropout=cfg.model.get("pred_dropout", 0.1),
        mtp=mtp,
    ).to(device)
    logger.info(f"MTP heads: {mtp}")
    n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in pred.parameters())
    logger.info(f"LeWM params: {n_params/1e6:.1f}M (enc={sum(p.numel() for p in enc.parameters())/1e6:.1f}M)")

    num_frames = cfg.model.get("num_frames", 4)
    skip = cfg.model.get("frame_skip", 5)
    lmbd = cfg.optim.get("sigreg_lambda", 0.1)
    num_proj = cfg.optim.get("sigreg_proj", 1024)

    params = list(enc.parameters()) + list(pred.parameters())
    opt = AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.get("weight_decay", 1e-4),
                betas=(0.9, 0.95))
    steps_per_epoch = data_config.size // data_config.batch_size
    sched = CosineWithWarmup(opt, cfg.optim.epochs * steps_per_epoch,
                             warmup_ratio=cfg.optim.get("warmup_ratio", 0.05))

    gstep = 0
    for epoch in range(cfg.optim.epochs):
        enc.train(); pred.train()
        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f"Epoch {epoch}/{cfg.optim.epochs-1}",
                    disable=cfg.logging.get("tqdm_silent", False))
        for idx, batch in pbar:
            x, a = batch[0].to(device), batch[1].to(device)
            frames, blocks = make_subtraj(x, a.float(), num_frames, skip)

            # AMP bf16 for the heavy ViT/predictor forward; loss (complex-valued
            # SIGReg) computed in fp32 for numerical safety.
            with autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.get("use_amp", True)):
                z = enc(frames)                            # [B, nf, D]  (end-to-end)
                preds = pred(z[:, :-1], blocks)            # [B,nf-1,D] or [B,nf-1,K,D]
            z = z.float(); preds = preds.float()
            if mtp > 1:
                loss, parts = lewm_mtp_loss(
                    preds, z, lmbd=lmbd, num_proj=num_proj, step=gstep)
            else:
                loss, parts = lewm_loss(
                    preds, z[:, 1:], z.reshape(-1, z.size(-1)),
                    lmbd=lmbd, num_proj=num_proj, step=gstep)

            opt.zero_grad()
            loss.backward()
            if cfg.optim.get("grad_clip", 0):
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            opt.step(); sched.step()

            if not cfg.logging.get("tqdm_silent", False):
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "pred": f"{parts['loss_pred'].item():.4f}",
                                  "sig": f"{parts['loss_sigreg'].item():.4f}",
                                  "lr": f"{opt.param_groups[0]['lr']:.1e}"})
            if wandb_run and gstep % cfg.logging.get("log_every", 10) == 0:
                wandb.log({"train/loss": loss.item(),
                           "train/loss_pred": parts["loss_pred"].item(),
                           "train/loss_sigreg": parts["loss_sigreg"].item(),
                           "optim/lr": opt.param_groups[0]["lr"], "epoch": epoch}, step=gstep)
            gstep += 1

        ckpt = folder / f"e-{epoch}.pth.tar"
        torch.save({"epoch": epoch, "global_step": gstep, "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "encoder": enc.state_dict(), "predictor": pred.state_dict()}, ckpt)
        torch.save({"epoch": epoch, "global_step": gstep, "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "encoder": enc.state_dict(), "predictor": pred.state_dict()}, folder / "latest.pth.tar")
        logger.info(f"Saved {ckpt}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(run)
