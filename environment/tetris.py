from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


BOARD_WIDTH = 10
BOARD_HEIGHT = 20

Action = int


TETROMINOES: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    "O": [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}


@dataclass
class Piece:
    name: str
    x: int = 3
    y: int = 0
    rotation: int = 0

    @property
    def blocks(self) -> List[Tuple[int, int]]:
        return TETROMINOES[self.name][self.rotation]


class TetrisEnv:
    """OpenAI Gym style Tetris environment with feature observations."""

    metadata = {"render_modes": ["human", "rgb_array"]}
    primitive_action_names = ["left", "right", "rotate", "drop", "hold"]

    def __init__(
        self,
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        use_hold: bool = True,
        action_mode: str = "primitive",
        seed: Optional[int] = None,
    ) -> None:
        if action_mode not in {"primitive", "placement"}:
            raise ValueError("action_mode must be 'primitive' or 'placement'")
        self.width = width
        self.height = height
        self.use_hold = use_hold
        self.action_mode = action_mode
        self.random = random.Random(seed)
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)
        self.current_piece: Piece = self._new_piece()
        self.next_piece_name = self._random_piece_name()
        self.hold_piece_name: Optional[str] = None
        self.can_hold = True
        self.done = False
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0

    @property
    def action_space_n(self) -> int:
        if self.action_mode == "placement":
            return self.width * 4 + int(self.use_hold)
        return len(self.primitive_action_names) if self.use_hold else 4

    @property
    def observation_space_shape(self) -> Tuple[int]:
        return (self.state_size,)

    @property
    def state_size(self) -> int:
        # 10 heights + aggregate height + holes + bumpiness + completed lines
        # + current piece one-hot + next piece one-hot + hold piece one-hot/empty
        # + normalized x/y/rotation + can-hold flag.
        return self.width + 4 + len(TETROMINOES) * 2 + (len(TETROMINOES) + 1) + 4

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_piece = self._new_piece()
        self.next_piece_name = self._random_piece_name()
        self.hold_piece_name = None
        self.can_hold = True
        self.done = False
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        return self.get_state()

    def step(self, action: Optional[Action]) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        if self.done:
            return self.get_state(), 0.0, True, self._info(0)

        prev_holes = self._count_holes()
        prev_bumpiness = self._bumpiness()
        reward = 0.05
        cleared = 0

        if self.action_mode == "placement":
            cleared, invalid = self._step_placement(action)
            if invalid:
                reward -= 1.0
        else:
            if action == 0:
                self._try_move(-1, 0)
            elif action == 1:
                self._try_move(1, 0)
            elif action == 2:
                self._try_rotate()
            elif action == 3:
                cleared = self._hard_drop()
            elif action == 4 and self.use_hold:
                if self.can_hold:
                    self._hold()
                else:
                    reward -= 0.2

        if self.action_mode == "primitive" and action != 3 and not self.done:
            if not self._try_move(0, 1):
                cleared = self._lock_piece()

        holes = self._count_holes()
        bumpiness = self._bumpiness()
        aggregate_height = self._aggregate_height()

        reward += [0.0, 1.0, 3.0, 7.0, 12.0][cleared]
        reward -= 0.08 * max(0, holes - prev_holes)
        reward -= 0.01 * max(0, bumpiness - prev_bumpiness)
        reward -= 0.002 * aggregate_height

        if self.done:
            reward -= 10.0

        self.steps += 1
        return self.get_state(), float(reward), self.done, self._info(cleared)

    def valid_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_space_n, dtype=np.bool_)
        if self.done:
            return np.zeros(self.action_space_n, dtype=np.bool_)
        if self.action_mode == "primitive":
            if self.use_hold:
                mask[4] = self.can_hold
            return mask

        mask.fill(False)
        for action in range(self.width * 4):
            rotation = action // self.width
            x = action % self.width
            piece = Piece(self.current_piece.name, x=x, y=self.current_piece.y, rotation=rotation)
            mask[action] = self._is_valid(piece)
        if self.use_hold:
            mask[-1] = self.can_hold
        return mask

    def describe_action(self, action: int) -> str:
        if self.action_mode == "placement":
            if self.use_hold and action == self.width * 4:
                return "hold"
            rotation = action // self.width
            x = action % self.width
            return f"rot {rotation}, x {x}"
        if 0 <= action < len(self.primitive_action_names):
            return self.primitive_action_names[action]
        return "-"

    def get_state(self) -> np.ndarray:
        heights = np.array(self._column_heights(), dtype=np.float32) / self.height
        aggregate_height = np.array([self._aggregate_height() / (self.width * self.height)], dtype=np.float32)
        holes = np.array([self._count_holes() / (self.width * self.height)], dtype=np.float32)
        bumpiness = np.array([self._bumpiness() / (self.width * self.height)], dtype=np.float32)
        complete_lines = np.array([self._completed_lines() / self.height], dtype=np.float32)

        piece_names = list(TETROMINOES.keys())
        current_one_hot = np.zeros(len(piece_names), dtype=np.float32)
        current_one_hot[piece_names.index(self.current_piece.name)] = 1.0

        next_one_hot = np.zeros(len(piece_names), dtype=np.float32)
        next_one_hot[piece_names.index(self.next_piece_name)] = 1.0

        hold_one_hot = np.zeros(len(piece_names) + 1, dtype=np.float32)
        if self.hold_piece_name is None:
            hold_one_hot[-1] = 1.0
        else:
            hold_one_hot[piece_names.index(self.hold_piece_name)] = 1.0

        pose = np.array(
            [
                self.current_piece.x / self.width,
                self.current_piece.y / self.height,
                self.current_piece.rotation / 3.0,
                1.0 if self.can_hold else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                heights,
                aggregate_height,
                holes,
                bumpiness,
                complete_lines,
                current_one_hot,
                next_one_hot,
                hold_one_hot,
                pose,
            ]
        )

    def render(self, mode: str = "human", cell_size: int = 30):
        import pygame

        colors = {
            0: (18, 20, 24),
            1: (80, 195, 255),
            2: (255, 216, 85),
            3: (178, 112, 255),
            4: (102, 220, 143),
            5: (255, 99, 99),
            6: (97, 139, 255),
            7: (255, 171, 82),
        }
        piece_ids = {name: idx + 1 for idx, name in enumerate(TETROMINOES)}

        surface = pygame.Surface((self.width * cell_size, self.height * cell_size))
        surface.fill((10, 12, 16))

        display_board = self.board.copy()
        for x, y in self._piece_cells(self.current_piece):
            if 0 <= y < self.height and 0 <= x < self.width:
                display_board[y, x] = piece_ids[self.current_piece.name]

        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(surface, colors[int(display_board[y, x])], rect)
                pygame.draw.rect(surface, (42, 47, 56), rect, 1)

        if mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return surface

    def _random_piece_name(self) -> str:
        return self.random.choice(list(TETROMINOES.keys()))

    def _new_piece(self, name: Optional[str] = None) -> Piece:
        return Piece(name or self._random_piece_name(), x=self.width // 2 - 2, y=0, rotation=0)

    def _piece_cells(self, piece: Piece) -> List[Tuple[int, int]]:
        return [(piece.x + bx, piece.y + by) for bx, by in piece.blocks]

    def _is_valid(self, piece: Piece) -> bool:
        for x, y in self._piece_cells(piece):
            if x < 0 or x >= self.width or y >= self.height:
                return False
            if y >= 0 and self.board[y, x]:
                return False
        return True

    def _try_move(self, dx: int, dy: int) -> bool:
        moved = Piece(self.current_piece.name, self.current_piece.x + dx, self.current_piece.y + dy, self.current_piece.rotation)
        if self._is_valid(moved):
            self.current_piece = moved
            return True
        return False

    def _try_rotate(self) -> bool:
        next_rotation = (self.current_piece.rotation + 1) % len(TETROMINOES[self.current_piece.name])
        for kick_x in (0, -1, 1, -2, 2):
            rotated = Piece(self.current_piece.name, self.current_piece.x + kick_x, self.current_piece.y, next_rotation)
            if self._is_valid(rotated):
                self.current_piece = rotated
                return True
        return False

    def _hard_drop(self) -> int:
        while self._try_move(0, 1):
            pass
        return self._lock_piece()

    def _step_placement(self, action: Optional[Action]) -> Tuple[int, bool]:
        if action is None:
            return self._hard_drop(), False
        if self.use_hold and action == self.width * 4:
            if not self.can_hold:
                return 0, True
            self._hold()
            return 0, False
        if action < 0 or action >= self.width * 4:
            return 0, True

        rotation = action // self.width
        x = action % self.width
        placed = Piece(self.current_piece.name, x=x, y=self.current_piece.y, rotation=rotation)
        if not self._is_valid(placed):
            return 0, True
        self.current_piece = placed
        return self._hard_drop(), False

    def _hold(self) -> None:
        if not self.can_hold:
            return
        current_name = self.current_piece.name
        if self.hold_piece_name is None:
            self.hold_piece_name = current_name
            self._spawn_next_piece()
        else:
            self.current_piece = self._new_piece(self.hold_piece_name)
            self.hold_piece_name = current_name
        self.can_hold = False
        if not self._is_valid(self.current_piece):
            self.done = True

    def _lock_piece(self) -> int:
        piece_ids = {name: idx + 1 for idx, name in enumerate(TETROMINOES)}
        for x, y in self._piece_cells(self.current_piece):
            if y < 0:
                self.done = True
                return 0
            self.board[y, x] = piece_ids[self.current_piece.name]

        cleared = self._clear_lines()
        self.lines_cleared += cleared
        self.score += [0, 100, 300, 500, 800][cleared]
        self._spawn_next_piece()
        return cleared

    def _spawn_next_piece(self) -> None:
        self.current_piece = self._new_piece(self.next_piece_name)
        self.next_piece_name = self._random_piece_name()
        self.can_hold = True
        if not self._is_valid(self.current_piece):
            self.done = True

    def _clear_lines(self) -> int:
        full_rows = np.where(np.all(self.board > 0, axis=1))[0]
        cleared = len(full_rows)
        if cleared:
            self.board = np.delete(self.board, full_rows, axis=0)
            self.board = np.vstack([np.zeros((cleared, self.width), dtype=np.uint8), self.board])
        return cleared

    def _column_heights(self) -> List[int]:
        heights = []
        for x in range(self.width):
            filled = np.where(self.board[:, x] > 0)[0]
            heights.append(0 if len(filled) == 0 else self.height - int(filled[0]))
        return heights

    def _aggregate_height(self) -> int:
        return int(sum(self._column_heights()))

    def _count_holes(self) -> int:
        holes = 0
        for x in range(self.width):
            column = self.board[:, x]
            filled = np.where(column > 0)[0]
            if len(filled):
                holes += int(np.sum(column[int(filled[0]) :] == 0))
        return holes

    def _bumpiness(self) -> int:
        heights = self._column_heights()
        return int(sum(abs(heights[i] - heights[i + 1]) for i in range(self.width - 1)))

    def _completed_lines(self) -> int:
        return int(np.sum(np.all(self.board > 0, axis=1)))

    def _info(self, cleared: int) -> Dict[str, int]:
        return {
            "score": self.score,
            "lines": self.lines_cleared,
            "cleared": cleared,
            "holes": self._count_holes(),
            "bumpiness": self._bumpiness(),
            "aggregate_height": self._aggregate_height(),
            "steps": self.steps,
        }
