import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import ale_py
import cv2
plt.switch_backend('Agg')
torch.backends.cudnn.benchmark = True

frame_shape = (84, 84)
repeat = 4
gamma = 0.99
learning_rate = 2.5e-4
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay_frames = 1_000_000
total_train_frames = 10_000_000
batch_size = 32
train_start = 20_000
train_freq = 4
target_update_freq = 10_000
rolling_average_n = 200
plot_folder = "plots-dualingDQN"
video_folder = "video-dualingDQN"
model_folder = "model-dualingDQN"
buffer_frame_limit = 200_000

os.makedirs(plot_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

env = gym.make("ALE/Enduro-v5", render_mode=None, frameskip=1)
env = gym.wrappers.AtariPreprocessing(env, grayscale_obs=True, frame_skip=repeat, screen_size=frame_shape[0])
env = NoopResetEnv(env)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

class DuelingQNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *frame_shape)
            conv_out_dim = self.conv(dummy).shape[1]

        self.advantage = nn.Sequential(
            nn.Linear(conv_out_dim, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(conv_out_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = x / 255.0
        features = self.conv(x)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, max_frames=200_000, frame_stack=4):
        self.buffer = []
        self.capacity = max_frames // frame_stack
        self.index = 0

    def push(self, s, a, r, s_, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.stack, zip(*batch))
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = 4
num_actions = env.action_space.n
q_network = DuelingQNetwork(input_channels, num_actions).to(device)
target_network = DuelingQNetwork(input_channels, num_actions).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.RMSprop(
    q_network.parameters(),
    lr=learning_rate,
    alpha=0.99,
    eps=0.01,
    momentum=0.95
)
loss_fn = nn.SmoothL1Loss()
buffer = ReplayBuffer()

def equalize_observation(obs):
    return np.stack([cv2.equalizeHist(frame) for frame in obs], axis=0)

def get_epsilon(step):
    fraction = min(step / epsilon_decay_frames, 1.0)
    return epsilon_start + fraction * (epsilon_final - epsilon_start)

episode_rewards = []
loss_history = []
global_step = 0
ep = 0

while global_step < total_train_frames:
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0
    losses = []

    for t in range(10000):
        if global_step >= total_train_frames:
            break

        epsilon = get_epsilon(global_step)
        global_step += 1

        if random.random() < epsilon:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)

        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(buffer) >= train_start and global_step % train_freq == 0:
            b_s, b_a, b_r, b_s_, b_d = buffer.sample(batch_size)
            b_s = torch.tensor(b_s, dtype=torch.float32).to(device)
            b_a = torch.tensor(b_a, dtype=torch.int64).unsqueeze(1).to(device)
            b_r = torch.tensor(b_r, dtype=torch.float32).unsqueeze(1).to(device)
            b_s_ = torch.tensor(b_s_, dtype=torch.float32).to(device)
            b_d = torch.tensor(b_d, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = q_network(b_s).gather(1, b_a)

            with torch.no_grad():
                next_q_online = q_network(b_s_).argmax(1, keepdim=True)
                next_q_target = target_network(b_s_).gather(1, next_q_online)
                target_q = b_r + gamma * (1 - b_d) * next_q_target

            loss = loss_fn(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if global_step % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        if done:
            break

    episode_rewards.append(total_reward)
    loss_history.append(np.mean(losses) if losses else 0)

    if (ep + 1) % 10 == 0:
        plt.figure(figsize=(10, 4))
        plt.plot(episode_rewards, label="Total Reward")
        if len(episode_rewards) >= rolling_average_n:
            avg = np.convolve(episode_rewards, np.ones(rolling_average_n) / rolling_average_n, mode='valid')
            plt.plot(range(rolling_average_n - 1, len(episode_rewards)), avg, label=f"{rolling_average_n}-ep Avg")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_folder, "reward_plot.png"))
        plt.close()

        plt.figure()
        plt.plot(loss_history, label="Loss", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid()
        plt.savefig(os.path.join(plot_folder, "loss_plot.png"))
        plt.close()

        print(f"[Ep {ep+1}] Frame: {global_step:,} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Avg Loss: {loss_history[-1]:.5f}")

    if (ep + 1) % 100 == 0:
        torch.save(q_network.state_dict(), os.path.join(model_folder, f"dueling_dqn_enduro_ep{ep+1}.pth"))

    ep += 1

torch.save(q_network.state_dict(), os.path.join(model_folder, "dueling_dqn_enduro_final.pth"))
print("Training complete. Final model saved.")
env.close()
