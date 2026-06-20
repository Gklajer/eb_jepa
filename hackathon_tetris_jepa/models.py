from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from eb_jepa.architectures import (
    CausalMultiHorizonPredictor,
    ImpalaEncoder,
    InverseDynamicsModel,
    RNNPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import MultiHorizonLoss, SquareLossSeq, VC_IDM_Sim_Regularizer


ModelKind = Literal["standard", "msp"]


@dataclass
class TetrisModelConfig:
    obs_channels: int = 2
    action_dim: int = 4
    img_size: int = 64
    encoder_dim: int = 256
    stack_sizes: tuple[int, int, int] = (16, 32, 32)
    horizon: int = 8
    context_length: int = 1
    predictor_depth: int = 2
    predictor_heads: int = 4
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    horizon_loss_gamma: float = 1.0
    horizon_loss_type: str = "smooth_l1"
    cov_coeff: float = 1.0
    std_coeff: float = 1.0
    sim_coeff_t: float = 0.25
    idm_coeff: float = 0.1


class BoardDecoder(nn.Module):
    """Small CNN decoder from JEPA latent vectors to 2x64x64 Tetris observations."""

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 2,
        base_channels: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (base_channels, 4, 4)),
            nn.ConvTranspose2d(base_channels, 96, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 5:
            batch, channels, time, height, width = latent.shape
            if height != 1 or width != 1:
                raise ValueError(
                    "BoardDecoder expects latent maps with spatial shape 1x1"
                )
            flat = latent.permute(0, 2, 1, 3, 4).reshape(batch * time, channels)
            decoded = self.net(flat)
            return decoded.view(batch, time, self.out_channels, 64, 64).permute(
                0, 2, 1, 3, 4
            )
        if latent.dim() == 4:
            if latent.shape[-2:] != (1, 1):
                raise ValueError(
                    "BoardDecoder expects latent maps with spatial shape 1x1"
                )
            latent = latent.flatten(1)
        elif latent.dim() != 2:
            raise ValueError(f"Unsupported latent shape {tuple(latent.shape)}")
        return self.net(latent)


def _make_encoder(config: TetrisModelConfig) -> ImpalaEncoder:
    return ImpalaEncoder(
        width=1,
        stack_sizes=config.stack_sizes,
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=config.obs_channels,
        final_ln=True,
        mlp_output_dim=config.encoder_dim,
        input_shape=(config.obs_channels, config.img_size, config.img_size),
    )


def build_tetris_jepa(
    model_kind: ModelKind,
    config: TetrisModelConfig | None = None,
) -> tuple[JEPA, BoardDecoder, str]:
    config = TetrisModelConfig() if config is None else config
    encoder = _make_encoder(config)

    if model_kind == "standard":
        predictor = RNNPredictor(
            hidden_size=config.encoder_dim,
            action_dim=config.action_dim,
            final_ln=nn.LayerNorm(config.encoder_dim),
        )
        pred_loss = SquareLossSeq()
        unroll_mode = "autoregressive"
    elif model_kind == "msp":
        predictor = CausalMultiHorizonPredictor(
            encoder_dim=config.encoder_dim,
            action_dim=config.action_dim,
            num_patches=1,
            horizon=config.horizon,
            context_length=config.context_length,
            pred_dim=config.encoder_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            mlp_ratio=config.predictor_mlp_ratio,
            dropout=config.predictor_dropout,
        )
        pred_loss = MultiHorizonLoss(
            gamma=config.horizon_loss_gamma,
            loss_type=config.horizon_loss_type,
        )
        unroll_mode = "direct_multi_horizon"
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    idm = InverseDynamicsModel(
        state_dim=config.encoder_dim,
        hidden_dim=256,
        action_dim=config.action_dim,
    )
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=config.cov_coeff,
        std_coeff=config.std_coeff,
        sim_coeff_t=config.sim_coeff_t,
        idm_coeff=config.idm_coeff,
        idm=idm,
        first_t_only=False,
        spatial_as_samples=False,
        idm_after_proj=False,
        sim_t_after_proj=False,
    )
    decoder = BoardDecoder(
        latent_dim=config.encoder_dim,
        out_channels=config.obs_channels,
    )
    jepa = JEPA(
        encoder=encoder,
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=regularizer,
        predcost=pred_loss,
    )
    return jepa, decoder, unroll_mode


def config_from_checkpoint_dict(raw_config: dict | None) -> TetrisModelConfig:
    if raw_config is None:
        return TetrisModelConfig()
    allowed = set(TetrisModelConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw_config.items() if key in allowed}
    if isinstance(filtered.get("stack_sizes"), list):
        filtered["stack_sizes"] = tuple(filtered["stack_sizes"])
    return TetrisModelConfig(**filtered)


def save_tetris_checkpoint(
    path: str | Path,
    model_kind: ModelKind,
    config: TetrisModelConfig,
    jepa: JEPA,
    decoder: BoardDecoder,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": model_kind,
        "config": asdict(config),
        "jepa_state_dict": jepa.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_tetris_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[JEPA, BoardDecoder, str, TetrisModelConfig, dict]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_kind = checkpoint["model_kind"]
    config = config_from_checkpoint_dict(checkpoint.get("config"))
    jepa, decoder, unroll_mode = build_tetris_jepa(model_kind, config)
    jepa.load_state_dict(checkpoint["jepa_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    jepa.to(device).eval()
    decoder.to(device).eval()
    return jepa, decoder, unroll_mode, config, checkpoint
