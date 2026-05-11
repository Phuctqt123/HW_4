from __future__ import annotations

import argparse

import pygame

from environment import TetrisEnv


KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_SPACE: 3,
    pygame.K_c: 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual Tetris play for environment testing.")
    parser.add_argument("--use-hold", action="store_true")
    args = parser.parse_args()

    pygame.init()
    env = TetrisEnv(use_hold=args.use_hold)
    env.reset()
    cell_size = 30
    screen = pygame.display.set_mode((env.width * cell_size, env.height * cell_size))
    clock = pygame.time.Clock()
    fall_timer = 0
    running = True

    while running:
        dt = clock.tick(30)
        fall_timer += dt
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = KEY_TO_ACTION.get(event.key)
                if event.key == pygame.K_r:
                    env.reset()

        if action is not None and action < env.action_space_n:
            env.step(action)
        elif fall_timer > 500 and not env.done:
            env.step(None)
            fall_timer = 0

        screen.blit(env.render(cell_size=cell_size), (0, 0))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
