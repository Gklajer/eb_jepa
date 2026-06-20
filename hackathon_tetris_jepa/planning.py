from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from eb_jepa.datasets.tetris.tetris_env import (
    ACTION_DIM,
    heuristic_from_tensor,
    one_hot_action,
)


@dataclass
class MPCConfig:
    num_candidates: int = 64
    horizon: int = 8
    threshold: float = 0.35
    seed: int = 0


@dataclass
class MPCResult:
    action_index: int
    action: torch.Tensor
    latency_ms: float
    best_score: float
    all_scores: np.ndarray


class TetrisMPCPlanner:
    def __init__(
        self,
        jepa,
        decoder,
        unroll_mode: str,
        config: MPCConfig | None = None,
        device: torch.device | str | None = None,
    ):
        self.jepa = jepa
        self.decoder = decoder
        self.unroll_mode = unroll_mode
        self.config = MPCConfig() if config is None else config
        self.device = (
            torch.device(device) if device is not None else next(jepa.parameters()).device
        )
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.config.seed)

    @torch.no_grad()
    def plan(self, observation: torch.Tensor) -> MPCResult:
        start = perf_counter()
        obs = observation.to(self.device, dtype=torch.float32)
        if obs.dim() != 3:
            raise ValueError(f"Expected observation [2,64,64], got {tuple(obs.shape)}")

        action_indices = torch.randint(
            low=0,
            high=ACTION_DIM,
            size=(self.config.num_candidates, self.config.horizon),
            generator=self.generator,
        )
        action_indices = action_indices.to(self.device)
        for action_idx in range(min(ACTION_DIM, self.config.num_candidates)):
            action_indices[action_idx, 0] = action_idx
        actions = F.one_hot(action_indices, num_classes=ACTION_DIM).float()
        actions = actions.permute(0, 2, 1).contiguous()

        obs_batch = obs.unsqueeze(0).unsqueeze(2).expand(
            self.config.num_candidates, -1, -1, -1, -1
        ).contiguous()
        predicted_latents, _ = self.jepa.unroll(
            obs_batch,
            actions,
            nsteps=self.config.horizon,
            unroll_mode=self.unroll_mode,
            ctxt_window_time=getattr(self.jepa.predictor, "context_length", 1),
            compute_loss=False,
            return_all_steps=False,
        )
        final_latent = predicted_latents[:, :, -1]
        decoded_logits = self.decoder(final_latent)
        decoded = torch.sigmoid(decoded_logits).detach().cpu()
        scores = np.asarray(
            [
                heuristic_from_tensor(frame, threshold=self.config.threshold)
                for frame in decoded
            ],
            dtype=np.float32,
        )
        best_idx = int(scores.argmax())
        latency_ms = (perf_counter() - start) * 1000.0
        action_index = int(action_indices[best_idx, 0].detach().cpu().item())
        return MPCResult(
            action_index=action_index,
            action=one_hot_action(action_index),
            latency_ms=latency_ms,
            best_score=float(scores[best_idx]),
            all_scores=scores,
        )
