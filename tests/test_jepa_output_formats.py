"""
Test script to verify output formats of JEPA's unroll() function.

This tests:
1. Output format of unroll() function in parallel and autoregressive modes
2. Correct behavior with both RNN and Conv predictors
3. return_all_steps functionality

Usage patterns:
- unroll(parallel, return_all_steps=True): Multi-step inference (formerly infern())
- unroll(parallel, compute_loss=True): Training with full GT trajectory
- unroll(autoregressive, compute_loss=False): Planning/MPC (formerly unrolln())
"""

import torch
import torch.nn as nn

from eb_jepa.architectures import (
    CausalMultiHorizonPredictor,
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    ResNet5,
    ResUNet,
    RNNPredictor,
    StateOnlyPredictor,
    build_multi_horizon_mask,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import (
    MultiHorizonLoss,
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
    VCLoss,
)


# ============================================================================
# Helper function to set random seed for reproducibility
# ============================================================================
def set_seed(seed=42):
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_video_jepa_model(device="cpu"):
    """Create a Video JEPA model matching the default config."""
    # Config values from examples/video_jepa/cfgs/default.yaml
    dobs = 1  # Input channels (grayscale)
    henc = 32  # Hidden dimension in encoder
    dstc = 16  # Output representation dimension
    hpre = 32  # Hidden dimension in predictor

    encoder = ResNet5(dobs, henc, dstc)
    predictor_model = ResUNet(2 * dstc, hpre, dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}")
    regularizer = VCLoss(std_coeff=10.0, cov_coeff=100.0, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    return jepa, dstc


def create_ac_video_jepa_model(device="cpu", img_size=65):
    """
    Create an Action-Conditioned Video JEPA model matching the
    architecture from examples/ac_video_jepa/main.py with default
    config from examples/ac_video_jepa/cfgs/train.yaml.
    """
    # Config values from examples/ac_video_jepa/cfgs/train.yaml
    dobs = 2  # Input channels (RGB + position = 2 channels for two_rooms)
    henc = 32  # Hidden dimension in encoder
    dstc = 32  # Output representation dimension
    nsteps = 8  # Number of prediction steps
    action_dim = 2  # Action dimension for two_rooms

    # Regularizer config
    cov_coeff = 8
    std_coeff = 16
    sim_coeff_t = 12
    idm_coeff = 1
    first_t_only = False
    spatial_as_samples = False
    use_proj = False
    idm_after_proj = False
    sim_t_after_proj = False

    # Create encoder (ImpalaEncoder as in main.py)
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, henc, dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(dobs, img_size, img_size),
    )

    # Test encoder to get output dimensions
    test_input = torch.rand((1, dobs, 1, img_size, img_size))
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    # Create predictor (RNNPredictor as in main.py)
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        action_dim=action_dim,
        final_ln=nn.LayerNorm(encoder.mlp_output_dim) if encoder.final_ln else None,
    )

    # Action encoder is identity
    aencoder = nn.Identity()

    # Projector (only if use_proj=True in config)
    if use_proj:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None

    # Create IDM (InverseDynamicsModel)
    idm = InverseDynamicsModel(
        state_dim=h * w * (projector.out_dim if idm_after_proj and projector else f),
        hidden_dim=256,
        action_dim=action_dim,
    ).to(device)

    # Create regularizer (VC_IDM_Sim_Regularizer as in main.py)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cov_coeff,
        std_coeff=std_coeff,
        sim_coeff_t=sim_coeff_t,
        idm_coeff=idm_coeff,
        idm=idm,
        first_t_only=first_t_only,
        projector=projector,
        spatial_as_samples=spatial_as_samples,
        idm_after_proj=idm_after_proj,
        sim_t_after_proj=sim_t_after_proj,
    )

    # Prediction loss
    ploss = SquareLossSeq()

    # Create JEPA model
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    config = {
        "dobs": dobs,
        "henc": henc,
        "dstc": dstc,
        "nsteps": nsteps,
        "action_dim": action_dim,
        "img_size": img_size,
        "mlp_output_dim": encoder.mlp_output_dim,
        "encoder_spatial_h": h,
        "encoder_spatial_w": w,
    }

    return jepa, config


class ZeroRegularizer(nn.Module):
    """Regularizer stub for output-format tests."""

    def forward(self, x, actions=None):
        loss = x.sum() * 0.0
        return loss, loss, {}


# ============================================================================
# Tests for unroll() function in parallel mode
# ============================================================================


def test_unroll_parallel_mode_output_format():
    """
    Test unroll() output format in parallel mode.

    Usage pattern:
        preds, losses = jepa.unroll(x, actions=None, nsteps=nsteps,
                                    unroll_mode="parallel", compute_loss=False,
                                    return_all_steps=True)
    """
    print("=" * 60)
    print("Testing unroll() parallel mode output format")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    jepa, dstc = create_video_jepa_model(device)
    jepa.eval()

    # Create fake video input matching Moving MNIST format
    # Shape: [B, C, T, H, W] = [batch, channels, time, height, width]
    B, C, T, H, W = 4, 1, 10, 64, 64
    x = torch.randn(B, C, T, H, W, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"  B (batch) = {B}")
    print(f"  C (channels) = {C}")
    print(f"  T (time steps) = {T}")
    print(f"  H (height) = {H}")
    print(f"  W (width) = {W}")

    # Call unroll with return_all_steps=True (like former infern())
    nsteps = T - 2
    print(f"\nCalling: jepa.unroll(x, actions=None, nsteps={nsteps}, ...")
    print(
        f"         unroll_mode='parallel', compute_loss=False, return_all_steps=True)"
    )

    with torch.no_grad():
        preds, losses = jepa.unroll(
            x,
            actions=None,
            nsteps=nsteps,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )

    print(f"\n--- unroll() Output Analysis ---")
    print(f"Return type of preds: {type(preds)}")
    print(f"Length of preds list: {len(preds)}")
    print(f"Expected length (nsteps): {nsteps}")
    print(f"losses: {losses} (expected: None when compute_loss=False)")

    # Analyze each prediction step
    print(f"\nPer-step shapes:")
    for i, pred in enumerate(preds):
        print(f"  preds[{i}] shape: {pred.shape}")

    # Verify the expected format
    first_pred = preds[0]
    context_length = jepa.predictor.context_length
    print(f"\n--- Shape Breakdown for preds[0] ---")
    print(f"  Dimension 0 (batch): {first_pred.shape[0]} (expected: {B})")
    print(f"  Dimension 1 (embedding dim): {first_pred.shape[1]} (expected: {dstc})")
    print(
        f"  Dimension 2 (time): {first_pred.shape[2]} (expected: T-context_length = {T}-{context_length} = {T-context_length})"
    )
    print(f"  Dimension 3 (height): {first_pred.shape[3]}")
    print(f"  Dimension 4 (width): {first_pred.shape[4]}")

    # Assertions
    assert isinstance(preds, list), f"Expected list, got {type(preds)}"
    assert len(preds) == nsteps, f"Expected {nsteps} steps, got {len(preds)}"
    assert losses is None, f"Expected losses=None, got {losses}"
    print("\n  ✓ All assertions passed!")

    print("=" * 60)

    return preds


def test_unroll_parallel_mode_with_loss():
    """
    Test unroll() output format in parallel mode with loss computation.

    Usage pattern:
        _, losses = jepa.unroll(x, actions=None, nsteps=cfg.model.steps,
                                unroll_mode="parallel", compute_loss=True)
        loss, rloss, rloss_unweight, rloss_dict, ploss = losses
    """
    print("\n" + "=" * 60)
    print("Testing unroll() parallel mode with loss computation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    jepa, dstc = create_video_jepa_model(device)
    jepa.train()

    # Create fake video input matching Moving MNIST format
    B, C, T, H, W = 4, 1, 10, 64, 64
    x = torch.randn(B, C, T, H, W, device=device)

    print(f"\nInput shape: {x.shape}")

    # Call unroll with compute_loss=True (for training)
    nsteps = 4
    print(f"\nCalling: jepa.unroll(x, actions=None, nsteps={nsteps}, ...")
    print(f"         unroll_mode='parallel', compute_loss=True)")

    predicted_states, losses = jepa.unroll(
        x, actions=None, nsteps=nsteps, unroll_mode="parallel", compute_loss=True
    )
    loss, rloss, rloss_unweight, rloss_dict, ploss = losses

    print(f"\n--- unroll() Output Analysis ---")
    print(f"Output is a tuple of (predicted_states, losses):")
    print(f"  predicted_states shape: {predicted_states.shape}")
    print(f"\nlosses tuple contains 5 elements:")
    print(
        f"  1. total_loss (loss):      {type(loss).__name__}, shape: {loss.shape}, value: {loss.item():.6f}"
    )
    print(
        f"  2. reg_loss (rloss):       {type(rloss).__name__}, shape: {rloss.shape}, value: {rloss.item():.6f}"
    )
    print(
        f"  3. reg_loss_unweighted:    {type(rloss_unweight).__name__}, shape: {rloss_unweight.shape}, value: {rloss_unweight.item():.6f}"
    )
    print(f"  4. reg_loss_dict:          {type(rloss_dict).__name__}")
    for k, v in rloss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"       - '{k}': {v.item():.6f}")
        else:
            print(f"       - '{k}': {v}")
    print(
        f"  5. pred_loss (ploss):      {type(ploss).__name__}, shape: {ploss.shape}, value: {ploss.item():.6f}"
    )

    # Assertions
    assert loss.shape == torch.Size(
        []
    ), f"total_loss should be scalar, got {loss.shape}"
    assert rloss.shape == torch.Size(
        []
    ), f"reg_loss should be scalar, got {rloss.shape}"
    assert ploss.shape == torch.Size(
        []
    ), f"pred_loss should be scalar, got {ploss.shape}"
    print("\n  ✓ All assertions passed!")

    print("=" * 60)

    return loss, rloss, rloss_unweight, rloss_dict, ploss


def test_infer_method():
    """Test the infer() method which uses unroll() internally."""
    print("\n" + "=" * 60)
    print("Testing infer() method")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa, _ = create_video_jepa_model(device)
    jepa.eval()

    B, C, T, H, W = 2, 1, 8, 64, 64
    x = torch.randn(B, C, T, H, W, device=device)

    with torch.no_grad():
        # infer() is defined as: unroll(..., nsteps=1, return_all_steps=True)[0]
        infer_result = jepa.infer(x, actions=None)

    print(f"infer() output shape: {infer_result.shape}")
    print(f"  ✓ infer() returns single tensor (first step from unroll)")

    print("=" * 60)


# ============================================================================
# Tests for unroll() function in autoregressive mode
# ============================================================================


def test_unroll_autoregressive_mode_shapes():
    """
    Test unroll() input and output tensor shapes in autoregressive mode.

    This tests the autoregressive mode as used in planning/MPC.

    Usage pattern:
        predicted_states, _ = jepa.unroll(obs_init, actions, nsteps,
                                          unroll_mode="autoregressive",
                                          ctxt_window_time=1, compute_loss=False)
    """
    print("\n" + "=" * 60)
    print("Testing AC Video JEPA unroll() autoregressive mode shapes")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create AC Video JEPA model
    img_size = 65  # Default for two_rooms
    jepa, config = create_ac_video_jepa_model(device, img_size=img_size)
    jepa.eval()

    print("\n--- Model Configuration (from train.yaml) ---")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Input dimensions matching two_rooms format
    B = 4  # Batch size
    C = config["dobs"]  # Input channels
    H = W = config["img_size"]
    A = config["action_dim"]
    D = config["mlp_output_dim"]  # Encoder output dimension

    # Test case 1: Single initial frame (as used in planning)
    print("\n" + "-" * 60)
    print("Test Case 1: Single initial frame (planning pattern)")
    print("-" * 60)

    T_context = 1  # Single context frame
    nsteps = 10  # Number of prediction steps
    T_actions = nsteps  # Actions for all prediction steps

    # Input: single initial observation
    obs_init = torch.randn(B, C, T_context, H, W, device=device)
    # Input: action sequence for unrolling
    actions = torch.randn(B, A, T_actions, device=device)

    print(f"\n--- Input Shapes ---")
    print(f"  obs_init:  [{B}, {C}, {T_context}, {H}, {W}]  (initial observation)")
    print(f"  actions:   [{B}, {A}, {T_actions}]  (action sequence)")
    print(f"  nsteps:    {nsteps}")

    print(f"\nCalling: jepa.unroll(obs_init, actions, nsteps={nsteps}, ...")
    print(
        f"         unroll_mode='autoregressive', ctxt_window_time=1, compute_loss=False)"
    )

    with torch.no_grad():
        predicted_states, losses = jepa.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=False,
        )

    print(f"\n--- unroll() Output Analysis ---")
    print(f"  predicted_states shape: {predicted_states.shape}")
    print(f"  losses: {losses} (expected: None when compute_loss=False)")

    # For RNN predictor (single_unroll=True), output is [B, D, 1 + nsteps, H', W']
    # The RNN predictor only uses the first frame (state[:, :, :1]) as initial state
    expected_T_out = 1 + nsteps  # First encoded frame + nsteps predictions
    expected_shape = (B, D, expected_T_out, 1, 1)

    print(f"\n--- Shape Assertions ---")
    assert (
        predicted_states.shape[0] == B
    ), f"Batch dim mismatch: {predicted_states.shape[0]} vs {B}"
    print(f"  ✓ Batch dimension: {B}")

    assert (
        predicted_states.shape[1] == D
    ), f"Feature dim mismatch: {predicted_states.shape[1]} vs {D}"
    print(f"  ✓ Feature dimension: {D}")

    assert (
        predicted_states.shape[2] == expected_T_out
    ), f"Time dim mismatch: {predicted_states.shape[2]} vs {expected_T_out}"
    print(f"  ✓ Time dimension: {expected_T_out} (1 + nsteps={nsteps})")

    assert losses is None, f"Expected losses=None, got {losses}"
    print(f"  ✓ losses is None when compute_loss=False")

    # Test case 2: Verify encoder output is preserved in first timestep(s)
    print("\n" + "-" * 60)
    print("Test Case 2: Verify encoder output preserved at t=0")
    print("-" * 60)

    with torch.no_grad():
        # Encode the initial observation
        encoded_init = jepa.encoder(obs_init)  # [B, D, T_context, 1, 1]

    print(f"  Encoded initial obs shape: {encoded_init.shape}")
    print(
        f"  First timestep of unroll output shape: {predicted_states[:, :, :T_context].shape}"
    )

    # The first timestep(s) should match the encoded initial observation
    first_timesteps = predicted_states[:, :, :T_context]
    assert torch.allclose(
        first_timesteps, encoded_init, atol=1e-5
    ), "First timestep(s) of unroll should match encoded initial observation"
    print(f"  ✓ First timestep matches encoded initial observation")

    # Test case 3: Verify error when nsteps > action sequence length
    print("\n" + "-" * 60)
    print("Test Case 3: Error handling for nsteps > action length")
    print("-" * 60)

    short_actions = torch.randn(B, A, 5, device=device)  # Only 5 actions
    try:
        with torch.no_grad():
            _ = jepa.unroll(
                obs_init,
                short_actions,
                nsteps=10,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
            )  # Request 10 steps
        print("  ✗ Should have raised an error!")
        assert False, "Expected ValueError for nsteps > action sequence length"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    print("\n" + "=" * 60)
    print("AC Video JEPA unroll() autoregressive mode Shape Test Summary:")
    print("=" * 60)
    print(f"  Input observations: [B, C, T_context, H, W]")
    print(f"  Input actions:      [B, A, T_actions] where T_actions >= nsteps")
    print(f"  Output:             [B, D, 1 + nsteps, H', W'] for RNN predictor")
    print(
        f"                      (only first frame of context is used as initial state)"
    )
    print(f"  Where:")
    print(f"    - B = batch size")
    print(f"    - D = encoder output dim ({D})")
    print(f"    - H', W' = spatial dims after encoder (1, 1 for ImpalaEncoder)")
    print("=" * 60)
    print("  ✓ All shape assertions passed!")
    print("=" * 60)

    return predicted_states


def test_unroll_autoregressive_with_loss():
    """
    Test unroll() autoregressive mode with loss computation for training.

    Usage pattern:
        _, losses = jepa.unroll(x, actions, nsteps,
                                unroll_mode="autoregressive", ctxt_window_time=1,
                                compute_loss=True)
    """
    print("\n" + "=" * 60)
    print("Testing AC Video JEPA unroll() autoregressive mode with loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    img_size = 65
    jepa, config = create_ac_video_jepa_model(device, img_size=img_size)
    jepa.train()

    # Verify this is an RNN predictor
    assert (
        jepa.single_unroll
    ), "AC Video JEPA should have single_unroll=True (RNN predictor)"
    print("  ✓ Confirmed RNN predictor (single_unroll=True)")

    # Create test input
    B = 2
    C = config["dobs"]
    T = 12
    H = W = config["img_size"]
    A = config["action_dim"]

    set_seed(42)
    x = torch.randn(B, C, T, H, W, device=device)
    actions = torch.randn(B, A, T, device=device)

    nsteps = 6
    print(f"\nInput shapes: observations={x.shape}, actions={actions.shape}")
    print(f"nsteps: {nsteps}")

    # Call unroll with compute_loss=True
    print(f"\nCalling: jepa.unroll(x, actions, nsteps={nsteps}, ...")
    print(
        f"         unroll_mode='autoregressive', ctxt_window_time=1, compute_loss=True)"
    )

    predicted_states, losses = jepa.unroll(
        x,
        actions,
        nsteps=nsteps,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=True,
    )
    loss, rloss, rloss_unweight, rloss_dict, ploss = losses

    print(f"\n--- unroll() Output Analysis ---")
    print(f"  predicted_states shape: {predicted_states.shape}")
    print(f"\nlosses tuple contains 5 elements:")
    print(f"  1. total_loss:         shape={loss.shape}, dtype={loss.dtype}")
    print(f"  2. reg_loss:           shape={rloss.shape}, dtype={rloss.dtype}")
    print(
        f"  3. reg_loss_unweight:  shape={rloss_unweight.shape}, dtype={rloss_unweight.dtype}"
    )
    print(f"  4. reg_loss_dict:      keys={list(rloss_dict.keys())}")
    print(f"  5. pred_loss:          shape={ploss.shape}, dtype={ploss.dtype}")

    # Assertions
    assert loss.shape == torch.Size(
        []
    ), f"total_loss should be scalar, got {loss.shape}"
    assert rloss.shape == torch.Size(
        []
    ), f"reg_loss should be scalar, got {rloss.shape}"
    assert ploss.shape == torch.Size(
        []
    ), f"pred_loss should be scalar, got {ploss.shape}"

    # reg_loss_dict should contain expected keys for VC_IDM_Sim_Regularizer
    expected_keys = {"std_loss", "cov_loss", "sim_loss_t", "idm_loss"}
    assert (
        set(rloss_dict.keys()) == expected_keys
    ), f"Expected keys {expected_keys}, got {set(rloss_dict.keys())}"
    print(f"\n  ✓ reg_loss_dict contains expected keys: {expected_keys}")
    print("  ✓ All assertions passed!")

    print("=" * 60)

    return loss, rloss, rloss_unweight, rloss_dict, ploss


def test_unroll_autoregressive_with_conv_predictor():
    """
    Test unroll() autoregressive mode with Conv predictor (non-RNN).

    This tests the sliding window behavior with the Video JEPA model,
    which uses a ResUNet predictor that processes sliding windows.
    """
    print("\n" + "=" * 60)
    print("Testing unroll() autoregressive mode (Conv predictor)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    jepa, dstc = create_video_jepa_model(device)
    jepa.eval()

    # Verify this is NOT an RNN predictor
    assert (
        not jepa.single_unroll
    ), "Video JEPA should have single_unroll=False (Conv predictor)"
    print("  ✓ Confirmed Conv predictor (single_unroll=False)")

    # Create test input
    B, C, T_context, H, W = 2, 1, 3, 64, 64
    nsteps = 5
    ctxt_window_time = 2

    set_seed(42)
    obs = torch.randn(B, C, T_context, H, W, device=device)

    print(f"\nInput shape: {obs.shape}")
    print(f"nsteps: {nsteps}")
    print(f"ctxt_window_time: {ctxt_window_time}")

    print(f"\nCalling: jepa.unroll(obs, actions=None, nsteps={nsteps}, ...")
    print(
        f"         unroll_mode='autoregressive', ctxt_window_time={ctxt_window_time})"
    )

    with torch.no_grad():
        unroll_result, unroll_losses = jepa.unroll(
            obs,
            actions=None,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=ctxt_window_time,
            compute_loss=False,
            return_all_steps=False,
        )

    expected_T_out = ctxt_window_time + nsteps
    print(f"\n  Output shape: {unroll_result.shape}")
    print(f"  Expected time dimension: {expected_T_out} (ctxt_window_time + nsteps)")

    assert (
        unroll_result.shape[2] == expected_T_out
    ), f"Time dim mismatch: got {unroll_result.shape[2]}, expected {expected_T_out}"
    print(f"  ✓ Time dimension correct: {unroll_result.shape[2]}")

    print("\n" + "=" * 60)
    print("unroll() autoregressive mode (Conv predictor) Test: PASSED")
    print("=" * 60)

    return True


def test_unroll_return_all_steps_format():
    """
    Test that return_all_steps=True returns the correct format for both modes.
    """
    print("\n" + "=" * 60)
    print("Testing unroll() return_all_steps format")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with Video JEPA (parallel mode)
    print("\n--- Parallel mode (Video JEPA) ---")
    set_seed(42)
    jepa, dstc = create_video_jepa_model(device)
    jepa.eval()

    B, C, T, H, W = 2, 1, 8, 64, 64
    x = torch.randn(B, C, T, H, W, device=device)
    nsteps = 3

    with torch.no_grad():
        all_steps, _ = jepa.unroll(
            x,
            actions=None,
            nsteps=nsteps,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )

    assert isinstance(all_steps, list), f"Expected list, got {type(all_steps)}"
    assert len(all_steps) == nsteps, f"Expected {nsteps} steps, got {len(all_steps)}"
    print(f"  ✓ Parallel mode returns list of {len(all_steps)} tensors")
    for i, step in enumerate(all_steps):
        print(f"    Step {i}: shape={step.shape}")

    # Test with AC Video JEPA (autoregressive mode)
    print("\n--- Autoregressive mode (AC Video JEPA) ---")
    set_seed(42)
    jepa_ac, config = create_ac_video_jepa_model(device)
    jepa_ac.eval()

    B = 2
    C = config["dobs"]
    H = W = config["img_size"]
    A = config["action_dim"]

    obs = torch.randn(B, C, 1, H, W, device=device)
    actions = torch.randn(B, A, nsteps, device=device)

    with torch.no_grad():
        all_steps_ac, _ = jepa_ac.unroll(
            obs,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=False,
            return_all_steps=True,
        )

    assert isinstance(all_steps_ac, list), f"Expected list, got {type(all_steps_ac)}"
    assert (
        len(all_steps_ac) == nsteps
    ), f"Expected {nsteps} steps, got {len(all_steps_ac)}"
    print(f"  ✓ Autoregressive mode returns list of {len(all_steps_ac)} tensors")
    for i, step in enumerate(all_steps_ac):
        print(f"    Step {i}: shape={step.shape}")

    # Verify autoregressive steps grow in time dimension
    for i in range(1, len(all_steps_ac)):
        assert (
            all_steps_ac[i].shape[2] == all_steps_ac[i - 1].shape[2] + 1
        ), f"Autoregressive steps should grow by 1: step {i-1}={all_steps_ac[i-1].shape[2]}, step {i}={all_steps_ac[i].shape[2]}"
    print("  ✓ Autoregressive steps correctly grow in time dimension")

    print("\n" + "=" * 60)
    print("unroll() return_all_steps format Test: PASSED")
    print("=" * 60)

    return True


def test_direct_multi_horizon_mask():
    """Verify causal access rules for the direct multi-horizon predictor mask."""
    num_context_tokens = 2 * 3
    num_patches = 3
    horizon = 4
    mask = build_multi_horizon_mask(
        num_context_tokens=num_context_tokens,
        num_patches=num_patches,
        horizon=horizon,
        device="cpu",
    )

    action_start = num_context_tokens
    query_start = action_start + horizon

    # Action token j can attend action prefix only.
    for j in range(horizon):
        row = action_start + j
        assert not mask[row, :num_context_tokens].any()
        assert not mask[row, action_start : action_start + j + 1].any()
        assert mask[row, action_start + j + 1 : action_start + horizon].all()

    # Query h can attend context and actions a_0...a_h, but no future actions.
    for h in range(horizon):
        for p in range(num_patches):
            row = query_start + h * num_patches + p
            assert not mask[row, :num_context_tokens].any()
            assert not mask[row, action_start : action_start + h + 1].any()
            assert mask[row, action_start + h + 1 : action_start + horizon].all()

            query_cols = mask[row, query_start:]
            self_col = h * num_patches + p
            assert not query_cols[self_col]
            query_cols_without_self = torch.cat(
                [query_cols[:self_col], query_cols[self_col + 1 :]]
            )
            assert query_cols_without_self.all()


def test_direct_multi_horizon_unroll_with_loss():
    """Test JEPA direct multi-horizon mode with sliding training windows."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, d, t, h, w = 2, 8, 7, 2, 2
    action_dim = 2
    horizon = 3
    context_length = 2

    predictor = CausalMultiHorizonPredictor(
        encoder_dim=d,
        action_dim=action_dim,
        num_patches=1,
        horizon=horizon,
        context_length=context_length,
        pred_dim=16,
        depth=1,
        num_heads=4,
    ).to(device)
    model = JEPA(
        encoder=nn.Identity(),
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=ZeroRegularizer(),
        predcost=MultiHorizonLoss(gamma=1.0),
    ).to(device)

    observations = torch.randn(b, d, t, h, w, device=device)
    actions = torch.randn(b, action_dim, t, device=device)
    predicted, losses = model.unroll(
        observations,
        actions,
        nsteps=horizon,
        unroll_mode="direct_multi_horizon",
        ctxt_window_time=context_length,
        compute_loss=True,
        return_all_steps=False,
    )

    assert predicted.shape == (b, d, context_length + horizon, 1, 1)
    assert losses is not None
    total_loss, reg_loss, _, _, pred_loss = losses
    assert torch.isfinite(total_loss)
    assert torch.isfinite(reg_loss)
    assert torch.isfinite(pred_loss)

    planned, plan_losses = model.unroll(
        observations[:, :, :1],
        actions[:, :, :horizon],
        nsteps=horizon,
        unroll_mode="direct_multi_horizon",
        ctxt_window_time=context_length,
        compute_loss=False,
        return_all_steps=False,
    )
    assert plan_losses is None
    assert planned.shape == (b, d, context_length + horizon, 1, 1)


def test_self_speculative_unroll_latent_verification():
    """Test self-speculative rollout uses latent draft/verify shapes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, d, h, w = 2, 8, 2, 2
    action_dim = 2
    horizon = 3
    context_length = 2

    predictor = CausalMultiHorizonPredictor(
        encoder_dim=d,
        action_dim=action_dim,
        num_patches=1,
        horizon=horizon,
        context_length=context_length,
        pred_dim=16,
        depth=1,
        num_heads=4,
    ).to(device)
    model = JEPA(
        encoder=nn.Identity(),
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=ZeroRegularizer(),
        predcost=MultiHorizonLoss(gamma=1.0),
    ).to(device)

    observations = torch.randn(b, d, 1, h, w, device=device)
    actions = torch.randn(b, action_dim, horizon, device=device)
    predicted, losses = model.unroll(
        observations,
        actions,
        nsteps=horizon,
        unroll_mode="self_speculative",
        ctxt_window_time=context_length,
        compute_loss=False,
        return_all_steps=False,
        speculative_threshold=0.05,
        speculative_distance_metric="normalized_mse",
    )

    assert losses is None
    assert predicted.shape == (b, d, context_length + horizon, 1, 1)
    assert model.last_speculative_stats["distance_metric"] == "normalized_mse"
    assert model.last_speculative_stats["threshold"] == 0.05

    all_steps, _ = model.unroll(
        observations,
        actions,
        nsteps=horizon,
        unroll_mode="self_speculative",
        ctxt_window_time=context_length,
        compute_loss=False,
        return_all_steps=True,
    )
    assert len(all_steps) == horizon
    assert all_steps[-1].shape == (b, d, context_length + horizon, 1, 1)


def run_all_tests():
    """Run all tests for unroll() function."""
    print("\n" + "#" * 60)
    print("# UNROLL() FUNCTION TEST SUITE")
    print("#" * 60)

    results = {}

    # Parallel mode tests
    try:
        test_unroll_parallel_mode_output_format()
        results["unroll parallel mode output"] = "PASSED"
    except AssertionError as e:
        results["unroll parallel mode output"] = f"FAILED: {e}"

    try:
        test_unroll_parallel_mode_with_loss()
        results["unroll parallel mode with loss"] = "PASSED"
    except AssertionError as e:
        results["unroll parallel mode with loss"] = f"FAILED: {e}"

    try:
        test_infer_method()
        results["infer method"] = "PASSED"
    except AssertionError as e:
        results["infer method"] = f"FAILED: {e}"

    # Autoregressive mode tests
    try:
        test_unroll_autoregressive_mode_shapes()
        results["unroll autoregressive mode shapes"] = "PASSED"
    except AssertionError as e:
        results["unroll autoregressive mode shapes"] = f"FAILED: {e}"

    try:
        test_unroll_autoregressive_with_loss()
        results["unroll autoregressive with loss"] = "PASSED"
    except AssertionError as e:
        results["unroll autoregressive with loss"] = f"FAILED: {e}"

    try:
        test_unroll_autoregressive_with_conv_predictor()
        results["unroll autoregressive (Conv)"] = "PASSED"
    except AssertionError as e:
        results["unroll autoregressive (Conv)"] = f"FAILED: {e}"

    try:
        test_unroll_return_all_steps_format()
        results["return_all_steps format"] = "PASSED"
    except AssertionError as e:
        results["return_all_steps format"] = f"FAILED: {e}"

    try:
        test_direct_multi_horizon_mask()
        results["direct multi-horizon mask"] = "PASSED"
    except AssertionError as e:
        results["direct multi-horizon mask"] = f"FAILED: {e}"

    try:
        test_direct_multi_horizon_unroll_with_loss()
        results["direct multi-horizon unroll"] = "PASSED"
    except AssertionError as e:
        results["direct multi-horizon unroll"] = f"FAILED: {e}"

    try:
        test_self_speculative_unroll_latent_verification()
        results["self-speculative latent unroll"] = "PASSED"
    except AssertionError as e:
        results["self-speculative latent unroll"] = f"FAILED: {e}"

    # Summary
    print("\n" + "#" * 60)
    print("# TEST SUMMARY")
    print("#" * 60)
    all_passed = True
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {test_name}: {result}")
        if result != "PASSED":
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# JEPA Output Format Test Suite")
    print("#" * 60 + "\n")

    all_passed = run_all_tests()

    print("\n" + "#" * 60)
    if all_passed:
        print("# All tests completed successfully!")
    else:
        print("# Some tests FAILED - see details above")
    print("#" * 60)
