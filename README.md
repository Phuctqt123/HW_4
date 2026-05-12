# Deep Q-Learning Tetris

This project is a complete Deep Reinforcement Learning system where a Deep Q-Network learns to play Tetris from scratch. It includes a custom Gym-style Tetris environment, PyTorch DQN models, replay memory, target network training, checkpoints, analytics plots, and Pygame visualization.

## Features

- 10x20 Tetris board with tetrominoes, rotation, collision, line clearing, scoring, and game over detection.
- Feature-based state representation: column heights, holes, bumpiness, aggregate height, completed lines, current/next/hold pieces, piece pose, and hold availability.
- Actions: move left, move right, rotate, hard drop, optional hold, or fast placement actions for training.
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

### 1. Chạy thử nhanh để kiểm tra code

Lệnh này chỉ dùng để kiểm tra chương trình có chạy được không. AI sẽ chưa học được gì đáng kể vì chỉ train 20 ván.

```bash
python -m training.train --episodes 20 --save-every 10 --log-every 5
```

- `--episodes 20`: train 20 ván.
- `--save-every 10`: mỗi 10 ván lưu checkpoint một lần.
- `--log-every 5`: mỗi 5 ván in thông tin train ra terminal.

### 2. Train nhanh và hiệu quả hơn bằng placement action

Đây là lệnh nên dùng nếu muốn AI học nhanh và cho kết quả khác random sớm hơn. Thay vì học từng nút bấm `left`, `right`, `rotate`, AI sẽ chọn trực tiếp cách đặt khối: xoay kiểu nào và đặt ở cột nào, sau đó game tự hard drop.

```bash
python -m training.train --episodes 2000 --action-mode placement --dueling --double-dqn
```

- `--episodes 2000`: train 2000 ván.
- `--action-mode placement`: dùng kiểu hành động đặt khối trực tiếp, học nhanh hơn.
- `--dueling`: dùng Dueling DQN, giúp model đánh giá trạng thái/hành động ổn định hơn.
- `--double-dqn`: dùng Double DQN, giúp giảm việc model đánh giá quá cao hành động.

Nên dùng lệnh này cho demo hoặc khi muốn train nhanh.

### 3. Train theo hành động giống người chơi

Lệnh này để AI học bằng các hành động cơ bản như người chơi: sang trái, sang phải, xoay, thả nhanh. Cách này gần với manual play hơn, nhưng học chậm hơn nhiều vì model phải học cả chuỗi hành động để đặt được một khối.

```bash
python -m training.train --episodes 5000 --dueling --double-dqn
```

- Không có `--action-mode placement`, nên mặc định là `primitive`.
- Cần nhiều episode hơn để thấy AI tiến bộ.
- Phù hợp nếu muốn AI điều khiển game theo từng nút bấm.

### 4. Thêm hold piece

Có thể thêm `--use-hold` vào các lệnh train, ví dụ:

```bash
python -m training.train --episodes 5000 --action-mode placement --dueling --double-dqn --use-hold
```

`--use-hold` cho AI dùng chức năng giữ khối. Nó có thể giúp chơi tốt hơn về sau, nhưng làm bài toán khó hơn, nên chỉ nên bật sau khi model không hold đã học ổn.

### So sánh nhanh

| Lệnh | Tốc độ học | Mục đích |
| --- | --- | --- |
| `--episodes 20` | Rất nhanh | Kiểm tra code chạy được |
| `--action-mode placement --episodes 2000` | Nhanh hơn | Nên dùng để demo/train hiệu quả |
| `--episodes 5000` không placement | Chậm hơn | Học điều khiển như người chơi |
| Thêm `--use-hold` | Chậm hơn nữa | Cho AI thêm chức năng hold |

Checkpoints được lưu trong `checkpoints/`, và biểu đồ train được lưu tại `training_plots/training_metrics.png`.

## Evaluate AI

```bash
python evaluate.py --model checkpoints/best_model.pt --fps 12
```

If no model exists yet, the evaluator runs a random policy so you can still verify the environment and renderer.

## Train trên GPU bên ngoài

Nếu máy cá nhân train chậm, có thể đưa project lên dịch vụ có GPU như Google Colab, Kaggle Notebook, RunPod hoặc Vast.ai. Cách dễ nhất là nén project, upload lên notebook GPU, train ở đó, rồi tải file checkpoint về máy để evaluate.

### Cách 1. Google Colab hoặc Kaggle

1. Nén toàn bộ project thành file `.zip`, ví dụ `HW_4.zip`.
2. Tạo notebook mới và bật GPU trong phần runtime/accelerator.
3. Upload file `.zip`.
4. Chạy các lệnh sau trong notebook:

```bash
unzip HW_4.zip
cd HW_4
pip install -r requirements.txt
python -m training.train --episodes 10000 --action-mode placement --dueling --double-dqn --save-every 500 --log-every 50
```

Sau khi train xong, tải các file quan trọng về:

```text
checkpoints/best_model.pt
checkpoints/final_model.pt
training_plots/training_metrics.png
```

Đưa `best_model.pt` về đúng thư mục `checkpoints/` trên máy cá nhân, rồi xem AI chơi:

```bash
python evaluate.py --model checkpoints/best_model.pt --fps 12
```

### Cách 2. Thuê GPU cloud

Nếu cần train lâu hơn, dùng RunPod/Vast.ai sẽ ổn định hơn Colab miễn phí. Quy trình thường là:

```bash
git clone <repo-cua-ban>
cd <repo>
pip install -r requirements.txt
python -m training.train --episodes 30000 --action-mode placement --dueling --double-dqn --save-every 1000 --log-every 100
```

Sau đó tải thư mục `checkpoints/` và `training_plots/` về máy.

### Kiểm tra có đang dùng GPU không

Khi train, code tự dùng CUDA nếu có GPU NVIDIA. Không thêm `--cpu`. Có thể kiểm tra nhanh bằng:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu kết quả là `True` và hiện tên GPU, train đang có thể chạy bằng GPU.

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

DQN needs many episodes to become visibly strong at Tetris. For faster and more effective learning, prefer `--action-mode placement`; the model chooses a rotation and x-position for each piece, then the environment hard-drops it. Primitive actions such as left, right, rotate, and drop are useful for manual-style control, but they learn much more slowly. `--use-hold` makes the game more powerful but also harder, so add it only after a no-hold baseline is working.
