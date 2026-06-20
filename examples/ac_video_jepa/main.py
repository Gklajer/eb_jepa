import copy
import os
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
import wandb
import yaml
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    MultiHorizonRNNPredictor,
    Projector,
    RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import GoalValueHead, MLPXYHead
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.ac_video_jepa.eval import launch_plan_eval, launch_unroll_eval

logger = get_logger(__name__)


def run(
    fname: str = "examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.henc=64).
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            folder_name = folder.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("ac_video_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="ac_video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]

    os.makedirs(folder, exist_ok=True)

    # -- SETUP (device must be ready before init_data for stream pipeline mode)
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, val_loader, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True),
        device=device,
    )
    if data_pipeline is not None:
        logger.info("Warming up data pipeline (stream mode: loading chunks 0 and 1)...")
        data_pipeline.warm_up()
        logger.info("Pipeline warm-up complete.")

    # -- WANDB
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "ac_video_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "ac_video_jepa"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config.batch_size,
        train_samples=data_config.size,
        val_samples=data_config.val_size,
    )

    # Mixed precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    # -- ENV (for plan/unroll eval)
    enable_eval = cfg.meta.get("enable_plan_eval", False)
    env_creator = None
    plan_cfg = None
    num_eval_episodes = 10
    n_parallel_eval = 1

    if enable_eval:
        if cfg.meta.eval_every_itr <= 0:
            cfg.meta.eval_every_itr = len(loader)
        with open(cfg.eval.plan_cfg_path, "r") as f:
            plan_cfg = yaml.load(f, Loader=yaml.FullLoader)
        # Merge training logging into plan_cfg so eval-specific keys
        # (save_gif, optional_plots) from planning_mppi.yaml are preserved.
        plan_cfg_logging = plan_cfg.get("logging") or {}
        plan_cfg_logging.update(dict(cfg.logging))
        plan_cfg["logging"] = plan_cfg_logging
        with open(cfg.eval.eval_cfg_path, "r") as f:
            eval_cfg_dict = yaml.safe_load(f)
        # The eval env must share the model's geometry. The model is built from
        # cfg.data (train config), so start from it and let the eval yaml's data
        # section override — otherwise a non-default img_size/maze size (e.g.
        # train_maze_small.yaml) wouldn't reach the env and the encoder built for
        # the trained img_size would receive mismatched frames.
        merged_eval_data = {
            **OmegaConf.to_container(cfg.data, resolve=True),
            **(eval_cfg_dict.get("data") or {}),
        }
        # The eval env only needs the dataset geometry (img_size/maze size); it
        # never streams data. Drop any training data pipeline (stream/offline)
        # inherited from cfg.data — it would require a device here and is
        # irrelevant to building the planning env config.
        merged_eval_data.pop("pipeline", None)
        _, _, env_config, _ = init_data(
            env_name=cfg.data.env_name, cfg_data=merged_eval_data
        )
        num_eval_episodes = eval_cfg_dict.get("meta", {}).get("num_eval_episodes", 10)
        n_parallel_eval = eval_cfg_dict.get("meta", {}).get("n_parallel", 1)

        def env_creator():
            from eb_jepa.datasets.utils import create_env

            cfg_eval_env = eval_cfg_dict.get("env") or {}
            return create_env(
                cfg.data.env_name,
                config=env_config,
                **cfg_eval_env,
            )

    # -- SAVE CONFIG
    latest_ckpt_path = folder / "latest.pth.tar"
    steps_per_epoch = data_config.size // data_config.batch_size
    total_steps = cfg.optim.epochs * steps_per_epoch
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    print(f"Saved complete config to {config_path}")

    # -- MODEL
    test_input = torch.rand(
        (
            1,
            cfg.model.dobs,
            1,
            data_config.img_size,
            data_config.img_size,
        )
    )
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape
    num_horizons = cfg.model.get("num_horizons", 1)
    if num_horizons > 1:
        predictor = MultiHorizonRNNPredictor(
            hidden_size=encoder.mlp_output_dim,
            num_horizons=num_horizons,
            final_ln=encoder.final_ln,
        )
    else:
        predictor = RNNPredictor(
            hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln
        )
    aencoder = nn.Identity()
    if cfg.model.regularizer.use_proj:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None
    logger.info(f"Encoder output: {tuple(test_output.shape)}")
    idm = InverseDynamicsModel(
        state_dim=h
        * w
        * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),
        hidden_dim=256,
        action_dim=2,
    ).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    log_config(cfg)

    # -- PROBER
    xy_head = MLPXYHead(
        input_shape=test_output.shape[1],
        normalizer=loader.dataset.normalizer,
    ).to(device)
    xy_prober = JEPAProbe(
        jepa=jepa,
        head=xy_head,
        hcost=nn.MSELoss(),
    )

    # Optional: freeze the encoder. The 4 MTP predictors (and the IDM in the
    # regularizer) keep training on the fixed latent space. Only params with
    # requires_grad=True are passed to AdamW so the encoder stays put.
    if cfg.model.get("freeze_encoder", False):
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
        logger.info("🥶 Encoder frozen — training predictor / IDM / probe only")

    jepa_optimizer = AdamW(
        filter(lambda p: p.requires_grad, jepa.parameters()),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    # -- VALUE HEAD (TD-MPC style learned planning cost)
    # A goal-conditioned scalar value V(z, z_goal) ≈ discounted return-to-goal,
    # trained by TD on the world model's own rollouts (gated by value_coeff>0).
    # At planning the `learned_value` objective maximises V instead of minimising
    # raw latent/position distance. `value_target` is an EMA copy (TD target net).
    value_coeff = cfg.model.get("value_coeff", 0.0)
    value_gamma = cfg.model.get("value_gamma", 0.95)
    value_head = GoalValueHead(test_output.shape[1]).to(device)
    value_target = copy.deepcopy(value_head)
    for p in value_target.parameters():
        p.requires_grad_(False)
    value_optimizer = AdamW(value_head.parameters(), lr=cfg.model.get("value_lr", 1e-3),
                            weight_decay=1e-5)
    value_scheduler = CosineWithWarmup(value_optimizer, total_steps, warmup_ratio=0.1)

    # -- LOAD CKPT
    start_epoch = 0
    ckpt_info = {}
    if cfg.meta.get("init_from"):
        # Fine-tune: load WEIGHTS ONLY from another checkpoint (fresh optimizer,
        # scheduler and epoch=0) so a new auxiliary objective can reshape the
        # model at a useful LR. Saves to this run's own folder, leaving the
        # source checkpoint untouched.
        from pathlib import Path as _P
        init_path = _P(cfg.meta.init_from)
        ckpt_info = load_checkpoint(
            init_path, jepa, optimizer=None, scheduler=None, device=device,
            strict=False,
        )
        start_epoch = 0
        if "xy_head_state_dict" in ckpt_info:
            xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])
        if "value_head_state_dict" in ckpt_info:
            value_head.load_state_dict(ckpt_info["value_head_state_dict"])
            value_target.load_state_dict(ckpt_info["value_head_state_dict"])
            logger.info("Loaded value_head from init checkpoint")
        logger.info(f"Fine-tune init from {init_path}: weights only, start_epoch=0")
    elif cfg.meta.load_model:
        checkpoint_path = folder / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(
            checkpoint_path, jepa, jepa_optimizer, jepa_scheduler, device=device
        )
        start_epoch = ckpt_info.get("epoch", 0)
        if "xy_head_state_dict" in ckpt_info:
            xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])
        if "value_head_state_dict" in ckpt_info:
            value_head.load_state_dict(ckpt_info["value_head_state_dict"])
            value_target.load_state_dict(ckpt_info["value_head_state_dict"])

    # Compile
    if torch.cuda.is_available() and cfg.model.compile:
        compile_mode = cfg.model.get("compile_mode")
        compile_kwargs = {"mode": compile_mode} if compile_mode else {}
        logger.info(f"✅ Compiling model with torch.compile (mode={compile_mode or 'default'})")
        jepa = torch.compile(jepa, **compile_kwargs)

    # -- EVAL ONLY MODE
    if cfg.meta.get("eval_only_mode", False):
        if not enable_eval:
            raise ValueError("eval_only_mode requires enable_plan_eval=True")
        logger.info("Running evaluation only (no training)")
        # Skip the (slow) unroll/rollout eval during planning tuning — it only
        # produces prediction-quality GIFs and is irrelevant to the success rate.
        if cfg.meta.get("skip_unroll_eval", False):
            logger.info("skip_unroll_eval=True — skipping rollout eval")
            eval_results = {}
        else:
            eval_results = launch_unroll_eval(
                jepa,
                env_creator,
                folder,
                start_epoch,
                ckpt_info.get("step", 0),
                "_eval_only",
                val_loader,
                xy_prober,
                cfg,
            )
        eval_results.update(
            launch_plan_eval(
                jepa,
                env_creator,
                folder,
                start_epoch,
                global_step=ckpt_info.get("step", 0),
                suffix="_eval_only",
                num_eval_episodes=num_eval_episodes,
                n_parallel=n_parallel_eval,
                loader=val_loader,
                prober=xy_prober,
                plan_cfg=plan_cfg,
                value_head=value_head,
            )
        )
        logger.info(
            f"Evaluation complete. Success rate: {eval_results['success_rate']:.2%}"
        )
        return eval_results

    # -- TRAINING LOOP
    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start_time = time()
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        for idx, (x, a, loc, _, _) in pbar:
            itr_start_time = time()
            global_step = epoch * len(loader) + idx

            x = x.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True)
            total_loss = torch.tensor(0.0, device=device)

            # When freeze_world_model=True, the encoder/predictor/probe are kept
            # FIXED (loaded from a proven checkpoint) and only the value head is
            # trained — the faithful TD-MPC setup (value learned on a fixed world
            # model's rollouts) and ~2x faster (no JEPA backward).
            freeze_wm = cfg.model.get("freeze_world_model", False)
            if not freeze_wm:
                # Calculate JEPA loss
                jepa_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    if num_horizons > 1:
                        _, (jepa_loss, regl, regl_unweight, regldict, pl) = (
                            jepa.unroll_multihorizon(x, a)
                        )
                    else:
                        _, (jepa_loss, regl, regl_unweight, regldict, pl) = (
                            jepa.unroll(
                                x,
                                a,
                                nsteps=cfg.model.nsteps,
                                unroll_mode="autoregressive",
                                ctxt_window_time=1,
                                compute_loss=True,
                                return_all_steps=False,
                            )
                        )
                    total_loss += jepa_loss

                # Auxiliary position loss: shape the ENCODER so its latent is
                # linearly position-decodable (the probe head is detached and does
                # NOT shape the encoder; planning in position space then suffers
                # from a noisy latent→position map → planner stalls). This term
                # flows into the encoder via jepa_optimizer. Gated by aux_pos_coeff.
                aux_pos_coeff = cfg.model.get("aux_pos_coeff", 0.0)
                if aux_pos_coeff:
                    with autocast(device.type, enabled=use_amp, dtype=dtype):
                        enc_state = jepa.encode(x[:, :, :1])  # [B, C, 1, H, W]
                        aux_pred = xy_head(enc_state)  # NO detach → grad to encoder
                        aux_loss = aux_pos_coeff * torch.nn.functional.mse_loss(
                            aux_pred, loc[:, :, :1]
                        )
                    jepa_loss = jepa_loss + aux_loss

                # Mixed precision backward pass
                scaler.scale(jepa_loss).backward()
                if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                    scaler.unscale_(jepa_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        jepa.encoder.parameters(), cfg.optim.grad_clip_enc
                    )
                    torch.nn.utils.clip_grad_norm_(
                        jepa.predictor.parameters(), cfg.optim.grad_clip_pred
                    )
                scaler.step(jepa_optimizer)
                scaler.update()
                jepa_scheduler.step()

                # Calculate probe loss
                probe_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    xy_loss = xy_prober(
                        observations=x[:, :, :1],
                        targets=loc[:, :, :1],
                    )
                    xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
                    total_loss += xy_loss

                scaler.scale(xy_loss).backward()
                scaler.step(probe_optimizer)
                scaler.update()
                probe_scheduler.step()
            else:
                # Frozen world model: placeholders so logging/epoch-summary work.
                jepa_loss = regl = regl_unweight = pl = torch.tensor(0.0, device=device)
                xy_loss = torch.tensor(0.0, device=device)
                regldict = {}

            # -- VALUE HEAD: TD(0) on the world model's OWN rollouts (TD-MPC style)
            # Goal = the trajectory-window endpoint latent (hindsight goal). The
            # value V(z_t, z_goal) is trained to predict the discounted
            # return-to-goal: reward 1 on reaching the endpoint, else 0; bootstrap
            # with an EMA target net. We regress BOTH the encoded real latents AND
            # the model's autoregressive rollout latents (what the planner actually
            # sees) onto the same TD target — so V is calibrated on imagined states.
            value_loss = torch.tensor(0.0, device=device)
            if value_coeff:
                with torch.no_grad(), autocast(device.type, enabled=use_amp, dtype=dtype):
                    z_gt = jepa.encode(x).float()  # [B, C, T, h, w]
                    Tw = z_gt.shape[2]
                    z_roll, _ = jepa.unroll(
                        x[:, :, :1], a, nsteps=Tw - 1,
                        unroll_mode="autoregressive", ctxt_window_time=1,
                        compute_loss=False, return_all_steps=False,
                    )  # [B, C, T, h, w] — model rollout from frame 0 with true actions
                    z_roll = z_roll.float()
                    g = z_gt[:, :, -1:].detach()  # goal latent
                    v_next = value_target(z_gt[:, :, 1:].detach(), g)  # [B, T-1]
                    done = torch.zeros_like(v_next)
                    done[:, -1] = 1.0  # endpoint reached at the last transition
                    td_target = done + value_gamma * (1.0 - done) * v_next  # [B, T-1]
                value_optimizer.zero_grad()
                with autocast(device.type, enabled=use_amp, dtype=dtype):
                    v_real = value_head(z_gt[:, :, :-1].detach(), g)   # [B, T-1]
                    v_roll = value_head(z_roll[:, :, :-1].detach(), g)  # [B, T-1]
                    value_loss = value_coeff * (
                        torch.nn.functional.mse_loss(v_real, td_target)
                        + torch.nn.functional.mse_loss(v_roll, td_target)
                    )
                scaler.scale(value_loss).backward()
                scaler.step(value_optimizer)
                scaler.update()
                value_scheduler.step()
                with torch.no_grad():  # EMA update of the TD target net
                    for pt, p in zip(value_target.parameters(), value_head.parameters()):
                        pt.mul_(0.99).add_(p.detach(), alpha=0.01)
                total_loss = total_loss + value_loss.detach()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "reg": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                    "val": f"{value_loss.item():.4f}",
                }
            )

            itr_time = time() - itr_start_time
            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/reg_loss_unweight": regl_unweight.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": xy_loss.item(),
                    "train/value_loss": value_loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "itr_time": itr_time,
                    "optim/jepa_lr": jepa_optimizer.param_groups[0]["lr"],
                    "optim/probe_lr": probe_optimizer.param_groups[0]["lr"],
                }
                for loss_name, loss_value in regldict.items():
                    log_data[f"train/regl/{loss_name}"] = loss_value

                if cfg.logging.get("log_wandb"):
                    wandb.log(log_data, step=global_step)

            # Planning eval (only if eval is enabled)
            if (
                enable_eval
                and (global_step + 1) % cfg.meta.eval_every_itr == 0
                and global_step > 0
            ):
                eval_results = launch_plan_eval(
                    jepa,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    num_eval_episodes=num_eval_episodes,
                    n_parallel=n_parallel_eval,
                    loader=val_loader,
                    prober=xy_prober,
                    plan_cfg=plan_cfg,
                    value_head=value_head,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

            # Light eval (only if eval is enabled)
            if (
                enable_eval
                and (global_step + 1) % cfg.meta.light_eval_freq == 0
                and global_step > 0
            ):
                eval_results = launch_unroll_eval(
                    jepa,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    loader=val_loader,
                    prober=xy_prober,
                    cfg=cfg,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

        epoch_time = time() - epoch_start_time

        # Log epoch summary
        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "reg": regl.item(),
                "pred": pl.item(),
                "probe": xy_loss.item(),
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch, "epoch_time": epoch_time},
                step=epoch * len(loader),
            )

        # Save checkpoint
        save_checkpoint(
            latest_ckpt_path,
            model=jepa,
            optimizer=jepa_optimizer,
            scheduler=jepa_scheduler,
            epoch=epoch,
            step=global_step,
            xy_head_state_dict=xy_head.state_dict(),
            value_head_state_dict=value_head.state_dict(),
            probe_optimizer_state_dict=probe_optimizer.state_dict(),
            probe_scheduler_state_dict=probe_scheduler.state_dict(),
        )
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                folder / f"e-{epoch}.pth.tar",
                model=jepa,
                optimizer=jepa_optimizer,
                scheduler=jepa_scheduler,
                epoch=epoch,
                step=global_step,
                xy_head_state_dict=xy_head.state_dict(),
                value_head_state_dict=value_head.state_dict(),
                probe_optimizer_state_dict=probe_optimizer.state_dict(),
                probe_scheduler_state_dict=probe_scheduler.state_dict(),
            )

    if data_pipeline is not None:
        data_pipeline.shutdown()


if __name__ == "__main__":
    fire.Fire(run)
