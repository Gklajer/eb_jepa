import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.logging import get_logger

logging = get_logger(__name__)


def spatial_mean_pool_latents(state):
    """Pool spatial latent maps to one global token per frame."""
    if state.dim() != 5:
        return state
    return state.mean(dim=(-2, -1), keepdim=True)


def latent_step_distance(draft, verifier, metric="normalized_mse"):
    """Distance between draft and verifier latent sequences, returned as [B, T]."""
    if draft.shape != verifier.shape:
        raise ValueError(
            "Draft and verifier latent shapes must match: "
            f"{tuple(draft.shape)} != {tuple(verifier.shape)}"
        )
    draft_flat = draft.permute(0, 2, 1, 3, 4).flatten(2)
    verifier_flat = verifier.permute(0, 2, 1, 3, 4).flatten(2)
    if metric == "mse":
        return (draft_flat - verifier_flat).pow(2).mean(dim=2)
    if metric == "normalized_mse":
        mse = (draft_flat - verifier_flat).pow(2).mean(dim=2)
        denom = verifier_flat.pow(2).mean(dim=2).clamp_min(1e-6)
        return mse / denom
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(draft_flat, verifier_flat, dim=2)
    raise ValueError(f"Unknown speculative latent distance metric: {metric}")


def accepted_prefix_lengths(distances, threshold):
    """Count accepted draft steps before the first verifier mismatch."""
    accepted = distances <= threshold
    return accepted.cumprod(dim=1).sum(dim=1)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, aencoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Action Encoder
        self.action_encoder = aencoder
        # Predictor
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file), weights_only=False)

    @torch.no_grad()
    def encode(self, observations):
        """Encode a sequence of observations and return the encoder output."""
        return self.encoder(observations)

    def prediction_state(self, state):
        """Return the latent representation expected by the predictor."""
        if getattr(self.predictor, "spatial_pool", False):
            return spatial_mean_pool_latents(state)
        return state


class JEPA(JEPAbase):
    """Trainable JEPA with prediction loss and anti-collapse regularizer."""

    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        """Initialize JEPA with regularizer and prediction cost in addition to base components."""
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.ploss = 0
        self.rloss = 0

    @torch.no_grad()
    def infer(self, observations, actions):
        """Produce single-step predictions over all sequence elements in parallel."""
        preds, _ = self.unroll(
            observations,
            actions,
            nsteps=1,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        return preds[0]

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
        speculative_threshold=0.05,
        speculative_distance_metric="normalized_mse",
    ):
        """Unified multi-step prediction with optional loss computation.

        This function supports both training (with loss computation) and planning/inference
        (without loss, just state prediction).

        Usage examples:
        - Training video_jepa: unroll(x, None, nsteps, unroll_mode="parallel", compute_loss=True)
        - Training ac_video_jepa with RNN: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=1, compute_loss=True)
        - Planning with ac_video_jepa: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=k, compute_loss=False)
        - Inference like infern(): unroll(x, a, nsteps, unroll_mode="parallel",
          compute_loss=False, return_all_steps=True)

        Predictor behavior:
        - unroll_mode="parallel" (Conv predictor, is_rnn=False):
          Processes all timesteps in parallel. Uses predictor.context_length to
          determine how many ground truth frames to re-feed at each iteration.
          Output: [B, D, T, H', W'] (same length as input, predictions replace non-context).
          Best for training with full ground truth trajectory available.

        - unroll_mode="autoregressive":
          Step-by-step prediction with sliding window of ctxt_window_time states.
          Each step: takes last ctxt_window_time states, predicts next, appends to sequence.
          Output: [B, D, T_context + nsteps, H', W'] (context + predictions appended).
          Best for planning/inference where future ground truth is not available.
          Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1.

        Args:
            observations: [B, C, T, H, W] - observation sequence
                For training (compute_loss=True): full trajectory with ground truth
                For planning (compute_loss=False): context frames only
            actions: [B, A, T_actions] - action sequence, or None for state-only prediction
                T_actions >= nsteps required for autoregressive mode
            nsteps: number of prediction steps
            unroll_mode: "parallel" or "autoregressive"
                - "parallel": Process all timesteps, refeed GT context on left
                - "autoregressive": Step-by-step, append predictions on right
            ctxt_window_time: Context window size for autoregressive mode.
                For RNN predictors (is_rnn=True), this is effectively 1.
            compute_loss: Whether to compute losses (requires ground truth observations)
            return_all_steps: If True, return list of predictions at each step (like infern).
                If False, return only the final predicted states.
            speculative_threshold: Latent distance threshold for self_speculative mode.
            speculative_distance_metric: "normalized_mse", "mse", or "cosine".

        Returns:
            Tuple of (predicted_states, losses) where:
            - If return_all_steps=False:
              predicted_states: [B, D, T_out, H', W'] - final predicted state sequence
            - If return_all_steps=True:
              predicted_states: List[Tensor] of length nsteps, each [B, D, T_out, H', W']
            - losses: None if compute_loss=False, otherwise tuple of 5 elements:
              (total_loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss)
        """
        state = self.encoder(observations)
        context_length = getattr(self.predictor, "context_length", 0)

        # Compute regularization loss if needed
        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        # Encode actions
        if actions is not None:
            actions_encoded = self.action_encoder(actions)
        else:
            actions_encoded = None

        # Collect all steps if requested
        all_steps = [] if return_all_steps else None

        # Parallel mode: process all timesteps at once, refeed GT context
        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                # Predict all timesteps, discard last (no target for it)
                predicted_states = self.predictor(predicted_states, actions_encoded)[
                    :, :, :-1
                ]
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states)
                # Refeed ground truth context on the left
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        # Direct multi-horizon mode: predict K future states from true latent context
        # and an action sequence in one forward pass with causal masking inside the predictor.
        elif unroll_mode == "direct_multi_horizon":
            if actions_encoded is None:
                raise ValueError("direct_multi_horizon requires action inputs")
            if nsteps > actions_encoded.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length "
                    f"({actions_encoded.size(2)})"
                )

            effective_ctxt_window = getattr(
                self.predictor, "context_length", ctxt_window_time
            )
            max_horizon = getattr(self.predictor, "horizon", nsteps)
            pred_state = self.prediction_state(state)

            if compute_loss:
                num_windows = min(
                    pred_state.size(2) - effective_ctxt_window - nsteps + 1,
                    actions_encoded.size(2) - effective_ctxt_window - nsteps + 2,
                )
                if num_windows <= 0:
                    raise ValueError(
                        "Not enough timesteps for direct_multi_horizon training: "
                        f"T_state={pred_state.size(2)}, T_actions={actions_encoded.size(2)}, "
                        f"context={effective_ctxt_window}, nsteps={nsteps}"
                    )

                context_batches = []
                action_batches = []
                target_batches = []
                for start in range(num_windows):
                    action_start = start + effective_ctxt_window - 1
                    context_batches.append(
                        pred_state[:, :, start : start + effective_ctxt_window]
                    )
                    action_batches.append(
                        actions_encoded[:, :, action_start : action_start + nsteps]
                    )
                    target_batches.append(
                        pred_state[
                            :,
                            :,
                            start
                            + effective_ctxt_window : start
                            + effective_ctxt_window
                            + nsteps,
                        ]
                    )

                context_states = torch.cat(context_batches, dim=0)
                context_actions = torch.cat(action_batches, dim=0)
                target_states = torch.cat(target_batches, dim=0)
                predicted_future = self.predictor(context_states, context_actions)
                ploss = self.predcost(target_states, predicted_future)

                # Return the first window in the usual [B, D, T, H, W] style.
                predicted_states = torch.cat(
                    [
                        pred_state[:, :, :effective_ctxt_window],
                        predicted_future[: state.size(0)],
                    ],
                    dim=2,
                )
                if return_all_steps:
                    all_steps.extend(
                        [
                            torch.cat(
                                [
                                    pred_state[:, :, :effective_ctxt_window],
                                    predicted_future[: state.size(0), :, : h + 1],
                                ],
                                dim=2,
                            )
                            for h in range(predicted_future.size(2))
                        ]
                    )
            else:
                if pred_state.size(2) < effective_ctxt_window:
                    pad_count = effective_ctxt_window - pred_state.size(2)
                    pad = pred_state[:, :, :1].expand(-1, -1, pad_count, -1, -1)
                    context_states = torch.cat([pad, pred_state], dim=2)
                else:
                    context_states = pred_state[:, :, -effective_ctxt_window:]

                predicted_states = context_states
                steps_done = 0
                while steps_done < nsteps:
                    chunk = min(max_horizon, nsteps - steps_done)
                    context_states = predicted_states[:, :, -effective_ctxt_window:]
                    context_actions = actions_encoded[
                        :, :, steps_done : steps_done + chunk
                    ]
                    predicted_future = self.predictor(context_states, context_actions)
                    predicted_states = torch.cat(
                        [predicted_states, predicted_future], dim=2
                    )
                    if return_all_steps:
                        all_steps.extend(
                            [
                                torch.cat(
                                    [
                                        predicted_states[:, :, : -chunk],
                                        predicted_future[:, :, : h + 1],
                                    ],
                                    dim=2,
                                )
                                for h in range(chunk)
                            ]
                        )
                    steps_done += chunk

        # Self-speculative latent rollout: draft K future latents directly, verify
        # the draft in latent space with one-step predictor calls, keep the valid
        # prefix, and regenerate after the first mismatch.
        elif unroll_mode == "self_speculative":
            if compute_loss:
                raise ValueError("self_speculative is an inference/planning mode only")
            if actions_encoded is None:
                raise ValueError("self_speculative requires action inputs")
            if nsteps > actions_encoded.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length "
                    f"({actions_encoded.size(2)})"
                )
            if not getattr(self.predictor, "direct_multi_horizon", False):
                raise ValueError(
                    "self_speculative requires a direct_multi_horizon predictor"
                )

            predicted_states, step_outputs = self._self_speculative_unroll(
                state=state,
                actions_encoded=actions_encoded,
                nsteps=nsteps,
                ctxt_window_time=ctxt_window_time,
                threshold=speculative_threshold,
                distance_metric=speculative_distance_metric,
                collect_steps=return_all_steps,
            )
            if return_all_steps:
                all_steps.extend(step_outputs)

        # Autoregressive mode: step-by-step with sliding window
        # Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1
        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length ({actions.size(2)})"
                )
            # For RNN predictors, force ctxt_window_time=1
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time

            predicted_states = state[:, :, :effective_ctxt_window]
            for i in range(nsteps):
                # Take last ctxt_window_time states
                context_states = predicted_states[:, :, -effective_ctxt_window:]
                # Take corresponding actions
                if actions_encoded is not None:
                    context_actions = actions_encoded[
                        :, :, max(0, i + 1 - effective_ctxt_window) : i + 1
                    ]
                else:
                    context_actions = None
                # Predict and take only last timestep
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                # Append prediction to sequence
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += (
                        self.predcost(pred_step, state[:, :, i + 1 : i + 2]) / nsteps
                    )
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        # Compute total loss and return
        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        # Return all steps or just final state
        if return_all_steps:
            return all_steps, losses
        else:
            return predicted_states, losses

    def _self_speculative_unroll(
        self,
        state,
        actions_encoded,
        nsteps,
        ctxt_window_time,
        threshold,
        distance_metric,
        collect_steps=False,
    ):
        effective_ctxt_window = getattr(
            self.predictor, "context_length", ctxt_window_time
        )
        max_horizon = getattr(self.predictor, "horizon", nsteps)
        pred_state = self.prediction_state(state)

        if pred_state.size(2) < effective_ctxt_window:
            pad_count = effective_ctxt_window - pred_state.size(2)
            pad = pred_state[:, :, :1].expand(-1, -1, pad_count, -1, -1)
            predicted_states = torch.cat([pad, pred_state], dim=2)
        else:
            predicted_states = pred_state[:, :, -effective_ctxt_window:]

        step_outputs = []
        accepted_counts = []
        distance_means = []
        chunks = 0
        steps_done = 0
        while steps_done < nsteps:
            chunk = min(max_horizon, nsteps - steps_done)
            context_states = predicted_states[:, :, -effective_ctxt_window:]
            chunk_actions = actions_encoded[:, :, steps_done : steps_done + chunk]

            draft_future = self.predictor(context_states, chunk_actions)
            verifier_future = self._verify_draft_future(
                context_states=context_states,
                draft_future=draft_future,
                chunk_actions=chunk_actions,
                context_length=effective_ctxt_window,
            )
            distances = latent_step_distance(
                draft_future, verifier_future, metric=distance_metric
            )
            accepted_prefix = accepted_prefix_lengths(distances, threshold)
            advance_per_item = torch.clamp(accepted_prefix + 1, max=chunk)
            advance = int(advance_per_item.min().item())
            advance = max(1, advance)

            accepted_counts.append(accepted_prefix.float().mean().detach())
            distance_means.append(distances.mean().detach())
            chunks += 1

            for local_step in range(advance):
                use_draft = (local_step < accepted_prefix).view(-1, 1, 1, 1, 1)
                next_state = torch.where(
                    use_draft,
                    draft_future[:, :, local_step : local_step + 1],
                    verifier_future[:, :, local_step : local_step + 1],
                )
                predicted_states = torch.cat([predicted_states, next_state], dim=2)
                if collect_steps:
                    step_outputs.append(predicted_states.clone())

            steps_done += advance

        if accepted_counts:
            mean_accepted_prefix = torch.stack(accepted_counts).mean().item()
            mean_verify_distance = torch.stack(distance_means).mean().item()
        else:
            mean_accepted_prefix = 0.0
            mean_verify_distance = 0.0
        self.last_speculative_stats = {
            "chunks": chunks,
            "mean_accepted_prefix": mean_accepted_prefix,
            "mean_verify_distance": mean_verify_distance,
            "threshold": float(threshold),
            "distance_metric": distance_metric,
        }
        return predicted_states, step_outputs

    def _verify_draft_future(
        self,
        context_states,
        draft_future,
        chunk_actions,
        context_length,
    ):
        batch_size, dim, chunk, height, width = draft_future.shape
        tentative = torch.cat([context_states, draft_future], dim=2)
        verifier_contexts = []
        verifier_actions = []
        for step in range(chunk):
            verifier_contexts.append(tentative[:, :, step : step + context_length])
            verifier_actions.append(chunk_actions[:, :, step : step + 1])

        flat_contexts = torch.cat(verifier_contexts, dim=0)
        flat_actions = torch.cat(verifier_actions, dim=0)
        verifier_flat = self.predictor(flat_contexts, flat_actions)[:, :, :1]
        verifier_future = (
            verifier_flat.reshape(chunk, batch_size, dim, 1, height, width)
            .permute(1, 2, 0, 3, 4, 5)
            .reshape(batch_size, dim, chunk, height, width)
        )
        return verifier_future


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed."""

    def __init__(self, jepa, head, hcost):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    @torch.no_grad()
    def infer(self, observations):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.encode(observations)
        return self.head(state)

    @torch.no_grad()
    def apply_head(self, embeddings):
        """
        Decode embeddings using the head.
        This is useful for generating predictions from an unrolling of the predictor, for example.
        """
        return self.head(embeddings)

    def forward(self, observations, targets):
        """Forward pass for training the head (JEPA encoder gradients are detached)."""
        with torch.no_grad():
            state = self.jepa.encode(observations)
        output = self.head(state.detach())
        return self.hcost(output, targets)
