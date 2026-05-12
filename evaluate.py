from __future__ import annotations

import argparse
import time
from pathlib import Path

import pygame
import torch

from environment import TetrisEnv
from models import DQN, DuelingDQN


def _checkpoint_action_size(checkpoint) -> int | None:
    if isinstance(checkpoint, dict) and "action_size" in checkpoint:
        return int(checkpoint["action_size"])
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    for key in ("net.6.weight", "advantage.2.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def load_model(checkpoint, env: TetrisEnv, device: torch.device):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        dueling = bool(checkpoint.get("dueling", False))
        model = (DuelingDQN if dueling else DQN)(env.state_size, env.action_space_n).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = DQN(env.state_size, env.action_space_n).to(device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.model, map_location=device) if Path(args.model).exists() else None
    action_size = _checkpoint_action_size(checkpoint) if checkpoint is not None else None
    action_mode = checkpoint.get("action_mode", args.action_mode) if isinstance(checkpoint, dict) else args.action_mode
    use_hold = bool(checkpoint.get("use_hold", args.use_hold)) if isinstance(checkpoint, dict) else args.use_hold
    if checkpoint is not None and "action_mode" not in checkpoint and action_size is not None:
        action_mode = "primitive"
        use_hold = action_size == 5
    if checkpoint is not None and args.use_hold != use_hold:
        print(f"Checkpoint uses {action_size} actions, so evaluation use_hold={use_hold}.")

    env = TetrisEnv(use_hold=use_hold, action_mode=action_mode)
    model = load_model(checkpoint, env, device) if checkpoint is not None else None

    cell_size = args.cell_size
    screen = pygame.display.set_mode((env.width * cell_size + 220, env.height * cell_size))
    pygame.display.set_caption("DQN Tetris Evaluation")
    font = pygame.font.SysFont("consolas", 20)
    clock = pygame.time.Clock()

    state = env.reset()
    running = True
    action = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = env.reset()

        if not env.done:
            if model is None:
                action = env.random.randrange(env.action_space_n)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = model(state_t).squeeze(0)
                    valid_mask = torch.tensor(env.valid_action_mask(), dtype=torch.bool, device=device)
                    q_values = q_values.masked_fill(~valid_mask, -1e9)
                    action = int(q_values.argmax(dim=0).item())
            state, _, _, info = env.step(action)
        else:
            info = {"score": env.score, "lines": env.lines_cleared, "holes": 0, "bumpiness": 0}

        screen.fill((12, 14, 18))
        board_surface = env.render(cell_size=cell_size)
        screen.blit(board_surface, (0, 0))

        panel_x = env.width * cell_size + 20
        lines = [
            "DQN Tetris",
            f"Model: {'loaded' if model else 'random'}",
            f"Score: {env.score}",
            f"Lines: {env.lines_cleared}",
            f"Piece: {env.current_piece.name}",
            f"Next: {env.next_piece_name}",
            f"Hold: {env.hold_piece_name or '-'}",
            f"Mode: {env.action_mode}",
            f"Action: {env.describe_action(action) if not env.done else '-'}",
            "",
            "Press R to reset",
        ]
        for i, text in enumerate(lines):
            color = (235, 238, 245) if i == 0 else (174, 181, 194)
            screen.blit(font.render(text, True, color), (panel_x, 25 + i * 28))

        pygame.display.flip()
        clock.tick(args.fps)
        if args.delay:
            time.sleep(args.delay)

    pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained DQN Tetris agent.")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--cell-size", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-hold", action="store_true")
    parser.add_argument("--action-mode", choices=["primitive", "placement"], default="primitive")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
