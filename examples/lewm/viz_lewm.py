"""Visualize a trained LeWM checkpoint: latent rollout decoded to pixels.

LeWM predicts in latent space; two_rooms has no pixel decoder, so we fit a small
MLPXYHead probe (latent->XY) on frozen LeWM features, then render XY->frames with
env.coord_to_pixel. MTP checkpoints can use self-speculative latent rollout:
multi-token heads draft, the horizon-1 head verifies, accepted draft prefixes
are kept, and mismatches regenerate from the verifier token.

    cd /lustre/work/vivatech-yentlteam/gklajer/eb_jepa
    PYTHONPATH=$PWD python -m examples.lewm.viz_lewm \
        --ckpt <run>/latest.pth.tar --out_dir <out>
"""

from __future__ import annotations

from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.optim import AdamW

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.lewm import LeWMPredictor, ViTTinyEncoder
from eb_jepa.logging import get_logger
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import load_config, setup_seed
from eb_jepa.vis_utils import create_comparison_gif, plot_distances

logger = get_logger(__name__)


def subtraj(x, a, loc, nf=4, skip=5):
    idx = [i * skip for i in range(nf)]
    frames = x[:, :, idx]
    blocks = torch.stack(
        [a[:, :, i * skip:(i + 1) * skip].sum(dim=2) for i in range(nf - 1)], dim=1
    )  # [B, nf-1, A]
    locs = loc[:, :, idx]  # [B, 2, nf]
    return frames, blocks, locs


@torch.no_grad()
def decode(z, probe, env, normalizer, wall_x, door_y):
    z5 = z.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T, 1, 1]
    xy = probe(z5).permute(0, 2, 1)                      # [B, T, 2]
    xy = normalizer.unnormalize_location(xy)
    frames = env.coord_to_pixel(xy.to(env.device), wall_x.to(env.device), door_y.to(env.device))
    return frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # [B, T, H, W, C]


def viz(ckpt: str, fname: str = None, out_dir: str = None,
        nf: int = 4, skip: int = 5, probe_iters: int = 1500,
        cache_batches: int = 40, seed: int = 0,
        speculative: bool = True, speculative_threshold: float = 0.05,
        speculative_metric: str = "normalized_mse", max_draft_steps: int = None):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(ckpt)
    state = torch.load(ckpt, map_location=device)
    cfg = load_config(fname) if fname else OmegaConf.create(state["cfg"])
    out_dir = Path(out_dir) if out_dir else ckpt.parent / f"viz_{ckpt.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = ViTTinyEncoder(
        in_channels=cfg.model.get("in_channels", 2), img_size=cfg.model.get("img_size", 224),
        patch=cfg.model.get("patch", 14), dim=cfg.model.get("enc_dim", 192),
        depth=cfg.model.get("enc_depth", 12), heads=cfg.model.get("enc_heads", 3),
    ).to(device)
    pred = LeWMPredictor(
        latent_dim=enc.hidden_dim, dim=cfg.model.get("pred_dim", 384),
        depth=cfg.model.get("pred_depth", 6), heads=cfg.model.get("pred_heads", 6),
        action_dim=2, dropout=cfg.model.get("pred_dropout", 0.1),
        mtp=cfg.model.get("mtp_horizon", 1),
    ).to(device)
    enc.load_state_dict(state["encoder"])
    missing, unexpected = pred.load_state_dict(state["predictor"], strict=False)
    if missing or unexpected:
        logger.warning(
            f"Predictor checkpoint compatibility: missing={missing}, "
            f"unexpected={unexpected}"
        )
    enc.eval(); pred.eval()
    logger.info(f"Loaded LeWM ckpt {ckpt} (epoch {state.get('epoch')})")

    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    cfg_data["pipeline"] = {"mode": "online"}
    loader, val_loader, data_config, _ = init_data(
        env_name=cfg.data.env_name, cfg_data=cfg_data, device=device)
    normalizer = val_loader.dataset.normalizer
    data_config.device = str(device)
    env = create_env(cfg.data.env_name, config=data_config)

    # ---- fit XY probe on cached LeWM features ----
    feats, locs = [], []
    it = iter(loader)
    for _ in range(cache_batches):
        try:
            x, a, loc, wx, dy = next(it)
        except StopIteration:
            break
        f, _, lc = subtraj(x.to(device), a.to(device).float(), loc.to(device), nf, skip)
        with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
            z = enc(f)
        feats.append(z.float().cpu()); locs.append(lc.cpu())
    feats = torch.cat(feats); locs = torch.cat(locs)  # [N,nf,D], [N,2,nf]
    probe = MLPXYHead(enc.hidden_dim).to(device)
    opt = AdamW(probe.parameters(), lr=1e-3); probe.train()
    N = feats.size(0)
    for i in range(probe_iters):
        bi = torch.randint(0, N, (min(256, N),))
        z = feats[bi].to(device); tgt = locs[bi].to(device)
        z5 = z.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        loss = torch.nn.functional.mse_loss(probe(z5), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 300 == 0 or i == probe_iters - 1:
            logger.info(f"  probe iter {i}: mse={loss.item():.5f}")
    probe.eval()

    # ---- rollout on a val batch ----
    x, a, loc, wall_x, door_y = next(iter(val_loader))
    x = x.to(device); a = a.to(device).float(); loc = loc.to(device)
    frames, blocks, locs = subtraj(x, a, loc, nf, skip)
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        z = enc(frames).float()                          # [B, nf, D]
    n = nf - 1
    if speculative and pred.mtp > 1:
        pred_true = pred.self_speculative_rollout(
            z[:, :1], blocks, n, threshold=speculative_threshold,
            distance_metric=speculative_metric, max_draft_steps=max_draft_steps,
        )                                                 # [B, n, D]
        true_spec_stats = dict(pred.last_speculative_stats)
        logger.info(f"self-spec true actions: {true_spec_stats}")
    else:
        if speculative and pred.mtp <= 1:
            logger.warning("speculative=True ignored because checkpoint has mtp_horizon<=1")
        pred_true = pred.rollout(z[:, :1], blocks, n)     # [B, n, D]
    rand_blocks = torch.randn_like(blocks)
    if speculative and pred.mtp > 1:
        pred_rand = pred.self_speculative_rollout(
            z[:, :1], rand_blocks, n, threshold=speculative_threshold,
            distance_metric=speculative_metric, max_draft_steps=max_draft_steps,
        )
        logger.info(f"self-spec random actions: {pred.last_speculative_stats}")
    else:
        pred_rand = pred.rollout(z[:, :1], rand_blocks, n)

    # latent MSE curve
    mse_t = ((pred_true - z[:, 1:]) ** 2).mean(dim=(0, 2)).cpu().numpy()
    plot_distances({"true actions": mse_t}, str(out_dir / "latent_rollout_mse.pdf"),
                   xlabel="rollout step", ylabel="latent MSE")
    logger.info("latent MSE/step: " + " | ".join(f"t{i+1}={v:.3f}" for i, v in enumerate(mse_t)))

    # decode to pixels
    z0 = z[:, :1]
    gt_dec = decode(z, probe, env, normalizer, wall_x, door_y)
    pred_seq = decode(torch.cat([z0, pred_true], 1), probe, env, normalizer, wall_x, door_y)
    rand_seq = decode(torch.cat([z0, pred_rand], 1), probe, env, normalizer, wall_x, door_y)
    gt = normalizer.unnormalize_state(frames.permute(0, 2, 1, 3, 4)).permute(0, 1, 3, 4, 2)
    gt = gt.clamp(0, 1).cpu().numpy()

    save = str(out_dir / f"lewm_rollout_{nf}f.gif")
    create_comparison_gif(gt, pred_seq, rand_seq, gt_dec=gt_dec, save_path=save)
    logger.info(f"Saved {save}")


if __name__ == "__main__":
    fire.Fire(viz)
