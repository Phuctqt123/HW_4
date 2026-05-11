# Deep Q-Learning Tetris

This project is a complete Deep Reinforcement Learning system where a Deep Q-Network learns to play Tetris from scratch. It includes a custom Gym-style Tetris environment, PyTorch DQN models, replay memory, target network training, checkpoints, analytics plots, and Pygame visualization.

## Features

- 10x20 Tetris board with tetrominoes, rotation, collision, line clearing, scoring, and game over detection.
- Feature-based state representation: column heights, holes, bumpiness, aggregate height, completed lines, current piece, and piece pose.
- Actions: move left, move right, rotate, hard drop, and optional hold.
- DQN with replay buffer, target network, epsilon-greedy exploration, GPU support, and Huber loss.
- Optional Double DQN and Dueling DQN.
- Training metrics: reward, loss, score, and epsilon charts.
- Pygame AI evaluation window and manual play mode.

## Project Structure

```text
environment/      Tetris game environment
models/           DQN and Dueling DQN networks
training/         Training loop and replay buffer
utils/            Plotting helpers
visualization/    Manual visualization utilities
checkpoints/      Saved model weights
training_plots/   Generated training graphs
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Start with a short smoke run:

```bash
python -m training.train --episodes 20 --save-every 10 --log-every 5
```

Longer training for a presentation:

```bash
python -m training.train --episodes 2000 --dueling --double-dqn --use-hold
```

Checkpoints are saved in `checkpoints/`, and graphs are saved in `training_plots/training_metrics.png`.

## Evaluate AI

```bash
python evaluate.py --model checkpoints/best_model.pt --fps 12 --use-hold
```

If no model exists yet, the evaluator runs a random policy so you can still verify the environment and renderer.

## Manual Play

```bash
python -m visualization.human_play --use-hold
```

Controls:

- Left/right arrows: move
- Up arrow: rotate
- Space: hard drop
- C: hold
- R: reset

## Notes for Good Results

DQN needs many episodes to become visibly strong at Tetris. For a university demo, train overnight if possible, keep `best_model.pt`, and show both the live gameplay window and `training_metrics.png` to demonstrate improvement over time.
