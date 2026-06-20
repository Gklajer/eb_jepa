import math

import torch

from eb_jepa.lewm import (
    LeWMPredictor,
    accepted_prefix_lengths,
    lewm_latent_step_distance,
)


def _tiny_predictor(mtp=3):
    torch.manual_seed(7)
    model = LeWMPredictor(
        latent_dim=8,
        dim=16,
        depth=1,
        heads=4,
        action_dim=2,
        max_len=16,
        dropout=0.0,
        mtp=mtp,
    )
    model.eval()
    return model


def test_latent_distance_and_prefix_acceptance():
    draft = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    verifier = torch.tensor([[[1.0, 2.0], [5.0, 4.0]]])

    distances = lewm_latent_step_distance(draft, verifier, metric="mse")
    torch.testing.assert_close(distances, torch.tensor([[0.0, 2.0]]))

    prefixes = accepted_prefix_lengths(
        torch.tensor([[0.01, 0.20, 0.01], [0.01, 0.02, 0.03]]),
        threshold=0.05,
    )
    torch.testing.assert_close(prefixes, torch.tensor([1, 3]))


def test_rejecting_all_drafts_matches_autoregressive_rollout():
    torch.manual_seed(11)
    model = _tiny_predictor(mtp=3)
    first_state = torch.randn(4, 1, 8)
    actions = torch.randn(4, 5, 2)

    autoregressive = model.rollout(first_state, actions, nsteps=5)
    speculative = model.self_speculative_rollout(
        first_state,
        actions,
        nsteps=5,
        threshold=-1.0,
        distance_metric="mse",
    )

    torch.testing.assert_close(speculative, autoregressive, atol=1e-5, rtol=1e-5)
    assert model.last_speculative_stats["accepted_token_rate"] == 0.0


def test_accepting_all_drafts_advances_by_mtp_chunks():
    torch.manual_seed(13)
    model = _tiny_predictor(mtp=3)
    first_state = torch.randn(4, 1, 8)
    actions = torch.randn(4, 7, 2)

    speculative = model.self_speculative_rollout(
        first_state,
        actions,
        nsteps=7,
        threshold=1e9,
        distance_metric="mse",
    )

    assert speculative.shape == (4, 7, 8)
    stats = model.last_speculative_stats
    assert stats["enabled"] is True
    assert stats["chunks"] == math.ceil(7 / 3)
    assert stats["accepted_token_rate"] == 1.0
