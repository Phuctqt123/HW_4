# Theory Report: Deep Q-Learning for Tetris

## Reinforcement Learning

Reinforcement Learning is a machine learning paradigm where an agent learns by interacting with an environment. At each time step the agent observes a state, chooses an action, receives a reward, and moves to a new state. The goal is to learn a policy that maximizes long-term cumulative reward.

## Q-Learning

Q-Learning estimates the value of taking action `a` in state `s`, written as `Q(s, a)`. The classical tabular update is:

```text
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

Here, `alpha` is the learning rate, `gamma` is the discount factor, `r` is the reward, and `s'` is the next state.

## Deep Q-Learning

Classic Q-Learning stores values in a table. Tetris has too many possible board configurations for a table to be practical. Deep Q-Learning replaces the table with a neural network:

```text
Q(s, a; theta)
```

The network receives board features and outputs one Q-value per action.

## Bellman Equation

DQN uses the Bellman optimality target:

```text
y = r + gamma * max_a' Q_target(s', a')
```

If the episode is finished, the future term is removed:

```text
y = r
```

## Loss Function

The model is trained to reduce the difference between predicted Q-values and Bellman targets. This project uses Huber loss:

```text
L(theta) = Huber(Q_policy(s, a; theta) - y)
```

Huber loss is less sensitive to large temporal-difference errors than mean squared error.

## Experience Replay

Experience replay stores transitions:

```text
(state, action, reward, next_state, done)
```

During training, random mini-batches are sampled from memory. This improves data efficiency and reduces correlation between consecutive game frames.

## Target Network

DQN uses two networks:

- Policy network: updated every gradient step.
- Target network: copied from the policy network every few episodes.

The target network stabilizes training by making Bellman targets change more slowly.

## Exploration vs Exploitation

The epsilon-greedy policy chooses a random action with probability `epsilon`, otherwise it chooses the action with the highest predicted Q-value:

```text
action = random_action, with probability epsilon
action = argmax_a Q(s, a), otherwise
```

Epsilon decays over time, so the agent explores heavily early and exploits learned behavior later.

## Reward Design

The environment rewards useful Tetris behavior:

- Positive reward for clearing lines.
- Small survival reward.
- Penalty for creating holes.
- Penalty for increasing bumpiness.
- Penalty for excessive aggregate height.
- Large negative reward for game over.

This guides the agent toward flatter boards, fewer holes, and more line clears.

## Why DQN Works for Tetris

Tetris requires sequential decision making under uncertainty. A move that looks good immediately can create holes that cause failure later. DQN is suitable because it learns long-term action values, not only immediate rewards. The feature representation captures important board quality signals, and the neural network learns how these signals relate to future score.

## Advanced Extensions

This project includes optional Double DQN and Dueling DQN:

- Double DQN reduces overestimation by selecting the best next action with the policy network and evaluating it with the target network.
- Dueling DQN separates state value and action advantage, which can help when many actions have similar value in a state.
