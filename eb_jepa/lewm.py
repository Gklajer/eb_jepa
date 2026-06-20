"""LeWorldModel (LeWM) reproduction — Maes et al., arXiv:2603.19312.

"Stable End-to-End Joint-Embedding Predictive Architecture from Pixels": a JEPA
world model trained end-to-end from raw pixels with only TWO loss terms and no
EMA / stop-gradient / pretrained encoder:

    L = ||ẑ_{t+1} - z_{t+1}||²  +  λ · SIGReg(Z)

where the encoder + predictor are optimised jointly. SIGReg (the single
anti-collapse regularizer, replacing PLDM's ~6 VICReg/IDM terms) pushes the
marginal of every random 1-D projection of the latents toward N(0,1) via the
Epps–Pulley characteristic-function statistic (Cramér–Wold: matching all 1-D
marginals ⇔ matching the joint). Collapse is prevented purely by SIGReg + the
BatchNorm-terminated projection head, so no stop-grad is needed.

Faithful to Appendix D for the TwoRoom setting:
  - Encoder: ViT-Tiny (patch 14, 12 layers, 3 heads, dim 192), [CLS] → MLP+BN.
  - Predictor: ViT-S backbone, learned pos-emb, causal mask, action conditioning
    via AdaLN-Zero (DiT-style, zero-init so it starts as identity). History 1 for
    TwoRoom; the causal transformer generalises to longer history.
  - Two-term loss, end-to-end, no EMA/stop-grad.

Reuses eb_jepa.losses.epps_pulley for the SIGReg statistic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.losses import epps_pulley


# ---------------------------------------------------------------------------
# Encoder: ViT-Tiny
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=14, in_chans=3, dim=192):
        super().__init__()
        self.n = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # x: [B, C, H, W]
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]


class ViTTinyEncoder(nn.Module):
    """ViT-Tiny image encoder. [B, C, T, H, W] -> [B, T, D] (one z per frame).

    Each frame is encoded independently (no temporal mixing in the encoder). The
    [CLS] token from the last block is projected by a 1-layer MLP + BatchNorm —
    NOT LayerNorm, which would zero the per-sample variance SIGReg relies on.
    """

    def __init__(self, in_channels=2, img_size=224, patch=14,
                 dim=192, depth=12, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.hidden_dim = dim
        self.patch_embed = PatchEmbed(img_size, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.patch_embed.n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)  # no final LN
        # Projection head: MLP + BatchNorm (enables SIGReg's anti-collapse).
        self.head = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _adapt(self, x):  # [N, C, H, W] -> [N, 3, img, img], ImageNet-normalised
        N, C, H, W = x.shape
        if C < 3:
            x = torch.cat([x, x.new_zeros(N, 3 - C, H, W)], dim=1)
        elif C > 3:
            x = x[:, :3]
        if (H, W) != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x, return_cls=False):  # [B, C, T, H, W] -> [B, T, D]
        """Returns the prediction latent z = BN(head(CLS)). With return_cls=True,
        also returns the RAW last-layer [CLS] token (pre-head, pre-BN) — the
        representation the paper decodes for visualization/probing."""
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self._adapt(x)
        tok = self.patch_embed(x)
        cls = self.cls.expand(tok.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos
        tok = self.blocks(tok)
        cls_raw = tok[:, 0]               # raw last-layer CLS -> [B*T, D]
        z = self.bn(self.head(cls_raw))   # prediction latent (BN, not LayerNorm)
        z = z.reshape(B, T, self.hidden_dim)
        if return_cls:
            return z, cls_raw.reshape(B, T, self.hidden_dim)
        return z

    def encode_cls(self, x):  # [B,C,T,H,W] -> raw [CLS] [B,T,D]
        return self.forward(x, return_cls=True)[1]


# ---------------------------------------------------------------------------
# Predictor: causal transformer with AdaLN-Zero action conditioning
# ---------------------------------------------------------------------------


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNBlock(nn.Module):
    """DiT-style block: self-attention + MLP, both modulated by the action via
    AdaLN-Zero. The final modulation projection is zero-init so the block starts
    as identity and action conditioning ramps in during training."""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond, attn_mask):
        # cond: [B, L, dim] action embedding per position -> pool to [B, dim] via mean?
        # We keep it per-position: produce modulation per position.
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)  # each [B, L, dim]
        h = self.norm1(x) * (1 + sc1) + s1
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g1 * a
        h = self.norm2(x) * (1 + sc2) + s2
        x = x + g2 * self.mlp(h)
        return x


class LeWMPredictor(nn.Module):
    """Action-conditioned causal transformer (ViT-S backbone). Predicts ẑ_{t+1}
    from the history of state latents z_{≤t} conditioned on action a_t."""

    def __init__(self, latent_dim=192, dim=384, depth=6, heads=6,
                 action_dim=2, max_len=64, dropout=0.1, mtp=1):
        super().__init__()
        self.mtp = mtp
        self.in_proj = nn.Linear(latent_dim, dim)
        self.act_proj = nn.Linear(action_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(dim, heads, dropout=dropout) for _ in range(depth)]
        )
        # Output head(s): Linear + BatchNorm, same structure as the encoder head.
        # mtp==1: single next-step head (names kept for checkpoint compat).
        # mtp>1 : K Multi-Token-Prediction heads, head k -> z_{t+1+k}.
        if mtp == 1:
            self.out = nn.Linear(dim, latent_dim)
            self.bn = nn.BatchNorm1d(latent_dim)
        else:
            self.heads = nn.ModuleList([nn.Linear(dim, latent_dim) for _ in range(mtp)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(latent_dim) for _ in range(mtp)])

    def _trunk(self, states, actions):
        B, L, _ = states.shape
        x = self.in_proj(states) + self.pos[:, :L]
        cond = self.act_proj(actions)
        mask = torch.triu(torch.ones(L, L, device=states.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, cond, mask)
        return x, B, L

    def forward(self, states, actions):
        # states: [B,L,latent], actions: [B,L,A]
        # -> [B,L,latent] (mtp==1) or [B,L,K,latent] (mtp>1)
        x, B, L = self._trunk(states, actions)
        if self.mtp == 1:
            out = self.bn(self.out(x).reshape(B * L, -1)).reshape(B, L, -1)
            return out
        outs = [self.bns[k](self.heads[k](x).reshape(B * L, -1)).reshape(B, L, -1)
                for k in range(self.mtp)]
        return torch.stack(outs, dim=2)  # [B, L, K, latent]

    @torch.no_grad()
    def rollout(self, first_state, actions, nsteps):
        """Autoregressive 1-step rollout (uses head k=0 if MTP). first_state
        [B,1,D], actions [B,>=nsteps,A]."""
        seq = first_state
        preds = []
        for i in range(nsteps):
            out = self.forward(seq, actions[:, : seq.size(1)])
            if self.mtp > 1:
                out = out[:, :, 0]               # k=0 (next-step) head -> [B,L,D]
            nxt = out[:, -1:, :]
            preds.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
        return torch.cat(preds, dim=1)


# ---------------------------------------------------------------------------
# SIGReg + total loss
# ---------------------------------------------------------------------------


def sigreg(z, num_proj=1024, step=0):
    """SIGReg(Z): mean Epps–Pulley statistic over `num_proj` random unit
    projections of the latents. z: [N, D] (flatten batch×time). Returns scalar."""
    D = z.size(1)
    g = torch.Generator(device=z.device)
    g.manual_seed(int(step))
    A = torch.randn(D, num_proj, device=z.device, generator=g)
    A = A / A.norm(p=2, dim=0, keepdim=True)
    proj = z @ A                       # [N, num_proj]
    return epps_pulley(proj).mean()


def lewm_loss(pred, target, all_z, lmbd=0.1, num_proj=1024, step=0):
    """Two-term LeWM loss. No stop-gradient on `target` (end-to-end; SIGReg is
    the sole anti-collapse mechanism)."""
    l_pred = F.mse_loss(pred, target)
    l_reg = sigreg(all_z, num_proj=num_proj, step=step)
    total = l_pred + lmbd * l_reg
    return total, {"loss_pred": l_pred.detach(), "loss_sigreg": l_reg.detach()}


def lewm_mtp_loss(preds, z, lmbd=0.1, num_proj=1024, step=0):
    """Multi-Token-Prediction LeWM loss. preds: [B, L, K, D] where head k at
    position t predicts z_{t+1+k}; z: [B, nf, D] (L = nf-1). The prediction term
    averages MSE over all valid (position, head) pairs; SIGReg unchanged."""
    B, L, K, D = preds.shape
    nf = z.size(1)
    l_pred, cnt = 0.0, 0
    for k in range(K):
        tmax = nf - 2 - k                       # last position with a z_{t+1+k} target
        if tmax < 0:
            break
        p = preds[:, : tmax + 1, k]             # [B, tmax+1, D]
        tg = z[:, 1 + k: 1 + k + tmax + 1]      # [B, tmax+1, D]
        l_pred = l_pred + F.mse_loss(p, tg)
        cnt += 1
    l_pred = l_pred / max(cnt, 1)
    l_reg = sigreg(z.reshape(-1, D), num_proj=num_proj, step=step)
    total = l_pred + lmbd * l_reg
    return total, {"loss_pred": l_pred.detach(), "loss_sigreg": l_reg.detach()}
