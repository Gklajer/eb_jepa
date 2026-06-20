from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eb_jepa.datasets.tetris.tetris_env import (
    ACTION_NAMES,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    TetrisEnv,
    generate_piece_sequence,
)
from hackathon_tetris_jepa.models import load_tetris_checkpoint
from hackathon_tetris_jepa.planning import MPCConfig, TetrisMPCPlanner


def _draw_text(surface, font, text, x, y, color=(235, 235, 235)):
    surface.blit(font.render(text, True, color), (x, y))


def _draw_board(pygame, surface, env: TetrisEnv, x: int, y: int, cell: int) -> None:
    locked_color = (45, 170, 210)
    active_color = (245, 196, 77)
    grid_color = (50, 54, 60)
    bg_color = (17, 19, 23)
    pygame.draw.rect(surface, bg_color, (x, y, BOARD_WIDTH * cell, BOARD_HEIGHT * cell))

    active = set(env.active_cells())
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            rect = pygame.Rect(x + col * cell, y + row * cell, cell, cell)
            if env.board[row, col] > 0:
                pygame.draw.rect(surface, locked_color, rect)
            elif (row, col) in active:
                pygame.draw.rect(surface, active_color, rect)
            pygame.draw.rect(surface, grid_color, rect, 1)

    pygame.draw.rect(
        surface,
        (160, 166, 176),
        (x, y, BOARD_WIDTH * cell, BOARD_HEIGHT * cell),
        2,
    )


def _load_planner(
    path: Path,
    device: torch.device,
    num_candidates: int,
    horizon: int,
    seed: int,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}. Train first with "
            "`python3 examples/ac_video_jepa/train_tetris.py`."
        )
    jepa, decoder, unroll_mode, _, _ = load_tetris_checkpoint(path, device=device)
    return TetrisMPCPlanner(
        jepa=jepa,
        decoder=decoder,
        unroll_mode=unroll_mode,
        config=MPCConfig(num_candidates=num_candidates, horizon=horizon, seed=seed),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side Tetris JEPA MPC demo")
    parser.add_argument("--standard", default="examples/ac_video_jepa/tetris_checkpoints/latest_standard.pth")
    parser.add_argument("--msp", default="examples/ac_video_jepa/tetris_checkpoints/latest_msp.pth")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cell", type=int, default=28)
    parser.add_argument("--speedup-every", type=float, default=12.0)
    args = parser.parse_args()

    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required for visualization: pip install pygame") from exc

    device = torch.device(args.device)
    try:
        standard_planner = _load_planner(
            Path(args.standard), device, args.num_candidates, args.horizon, args.seed
        )
        msp_planner = _load_planner(
            Path(args.msp), device, args.num_candidates, args.horizon, args.seed + 1
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    piece_sequence = generate_piece_sequence(10000, seed=args.seed)
    standard_env = TetrisEnv(seed=args.seed, piece_sequence=piece_sequence)
    msp_env = TetrisEnv(seed=args.seed, piece_sequence=piece_sequence)

    pygame.init()
    font = pygame.font.SysFont("Menlo", 18)
    title_font = pygame.font.SysFont("Menlo", 24, bold=True)
    cell = args.cell
    board_w = BOARD_WIDTH * cell
    board_h = BOARD_HEIGHT * cell
    margin = 42
    panel_gap = 80
    width = margin * 2 + board_w * 2 + panel_gap
    height = board_h + 170
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tetris JEPA MPC: Standard vs MSP")
    clock = pygame.time.Clock()

    standard_latency = 0.0
    msp_latency = 0.0
    standard_score = 0.0
    msp_score = 0.0
    started_ticks = pygame.time.get_ticks()
    next_step_ms = 0
    running = True

    while running:
        now_ms = pygame.time.get_ticks()
        elapsed_s = (now_ms - started_ticks) / 1000.0
        speed_level = int(elapsed_s // args.speedup_every)
        step_interval_ms = max(80, int(700 * (0.84 ** speed_level)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if now_ms >= next_step_ms:
            if not standard_env.game_over:
                result = standard_planner.plan(standard_env.observation())
                standard_latency = result.latency_ms
                standard_score = result.best_score
                standard_env.step(result.action_index)
            if not msp_env.game_over:
                result = msp_planner.plan(msp_env.observation())
                msp_latency = result.latency_ms
                msp_score = result.best_score
                msp_env.step(result.action_index)
            next_step_ms = pygame.time.get_ticks() + step_interval_ms

        screen.fill((11, 12, 15))
        left_x = margin
        right_x = margin + board_w + panel_gap
        board_y = 92

        _draw_text(screen, title_font, "Standard JEPA", left_x, 28)
        _draw_text(screen, title_font, "MSP JEPA", right_x, 28)
        _draw_text(screen, font, f"RNN autoregressive K={args.horizon}", left_x, 58)
        _draw_text(screen, font, f"Direct multi-horizon K={args.horizon}", right_x, 58)

        _draw_board(pygame, screen, standard_env, left_x, board_y, cell)
        _draw_board(pygame, screen, msp_env, right_x, board_y, cell)

        metric_y = board_y + board_h + 20
        budget_color_standard = (
            (245, 92, 92) if standard_latency > step_interval_ms else (145, 220, 160)
        )
        budget_color_msp = (
            (245, 92, 92) if msp_latency > step_interval_ms else (145, 220, 160)
        )
        _draw_text(
            screen,
            font,
            f"planning {standard_latency:7.1f} ms",
            left_x,
            metric_y,
            budget_color_standard,
        )
        _draw_text(
            screen,
            font,
            f"planning {msp_latency:7.1f} ms",
            right_x,
            metric_y,
            budget_color_msp,
        )
        _draw_text(
            screen,
            font,
            f"lines {standard_env.lines_cleared:3d}  h {standard_score:7.2f}",
            left_x,
            metric_y + 26,
        )
        _draw_text(
            screen,
            font,
            f"lines {msp_env.lines_cleared:3d}  h {msp_score:7.2f}",
            right_x,
            metric_y + 26,
        )
        _draw_text(
            screen,
            font,
            f"speed level {speed_level}  tick budget {step_interval_ms} ms",
            margin,
            metric_y + 62,
        )
        _draw_text(
            screen,
            font,
            f"actions: {', '.join(ACTION_NAMES)}",
            margin,
            metric_y + 88,
            (170, 176, 186),
        )

        if standard_env.game_over:
            _draw_text(
                screen,
                title_font,
                "GAME OVER",
                left_x + 46,
                board_y + board_h // 2,
                (245, 92, 92),
            )
        if msp_env.game_over:
            _draw_text(
                screen,
                title_font,
                "GAME OVER",
                right_x + 46,
                board_y + board_h // 2,
                (245, 92, 92),
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
