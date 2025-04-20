import datetime
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import cv2
import random

random.seed(42)

ENV_NAME = 'BipedalWalker-v3'
SEED = 42
NUMBER_OF_STEPS = 2048         
NUMBER_OF_EPISODES = 3000 
BATCH_SIZE = 2048       
MINIBATCH_SIZE = 32      
UPDATE_STEPS = 10   
GAE = True
GAMMA = 0.99
LAMBDA = 0.96
CLIPPING_EPSILON = 0.2
LEARNING_RATE_POLICY = 4e-4
LEARNING_RATE_CRITIC = 3e-4
ANNEAL_LR = True
MAX_GRAD_NORM = 0.5
EPSILON_ADAM = 1e-5   
ENTROPY_COEF = 2e-4
ENV_SCALE_CROP = True  
WRITER_FLAG = True       
SAVE_INTERVAL = 100
OUT_DIR = "models-128"
VALUE_COEF = 0.5

plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

def entropy_coef_schedule(step):
    return ENTROPY_COEF

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(x).squeeze(-1)
        return dist, value

    def get_action(self, obs, act_low, act_high, deterministic=False):
        dist, value = self.forward(obs)
        action = dist.mean if deterministic else dist.sample()
        action_clamped = torch.clamp(action, act_low, act_high)
        logp = dist.log_prob(action).sum(-1)
        return action_clamped, logp, value

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs        = np.zeros((size, obs_dim), np.float32)
        self.actions    = np.zeros((size, act_dim), np.float32)
        self.logp       = np.zeros(size, np.float32)
        self.rewards    = np.zeros(size, np.float32)
        self.values     = np.zeros(size, np.float32)
        self.dones      = np.zeros(size, np.float32)
        self.advantages = np.zeros(size, np.float32)
        self.returns    = np.zeros(size, np.float32)
        self.ptr = 0
        self.size = size

    def add(self, o, a, lp, r, v, d):
        idx = self.ptr
        self.obs[idx]     = o
        self.actions[idx] = a
        self.logp[idx]    = lp
        self.rewards[idx] = r
        self.values[idx]  = v
        self.dones[idx]   = d
        self.ptr += 1

    def finish(self, last_val):
        adv = 0
        for t in reversed(range(self.ptr)):
            mask  = 1 - self.dones[t]
            delta = self.rewards[t] + GAMMA * last_val * mask - self.values[t]
            adv   = delta + GAMMA * LAMBDA * mask * adv
            self.advantages[t] = adv
            last_val = self.values[t]
        self.returns = self.advantages + self.values[:self.ptr]
        adv_slice = self.advantages[:self.ptr]
        self.advantages[:self.ptr] = (adv_slice - adv_slice.mean()) / (adv_slice.std() + 1e-8)

    def get_batches(self, batch_size):
        idxs = np.random.permutation(self.ptr)
        for start in range(0, self.ptr, batch_size):
            b = idxs[start:start+batch_size]
            yield (
                torch.tensor(self.obs[b], dtype=torch.float32),
                torch.tensor(self.actions[b], dtype=torch.float32),
                torch.tensor(self.logp[b], dtype=torch.float32),
                torch.tensor(self.advantages[b], dtype=torch.float32),
                torch.tensor(self.returns[b], dtype=torch.float32)
            )

def train():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = gym.make(ENV_NAME, render_mode=None)
    torch.manual_seed(SEED); np.random.seed(SEED)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low, act_high = float(env.action_space.low[0]), float(env.action_space.high[0])

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer_policy = optim.Adam(
        list(model.actor.parameters()) + [model.log_std],
        lr=LEARNING_RATE_POLICY
    )
    optimizer_critic = optim.Adam(
        model.critic.parameters(),
        lr=LEARNING_RATE_CRITIC
    )
    if ANNEAL_LR:
        scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_policy, T_max=NUMBER_OF_EPISODES, eta_min=LEARNING_RATE_POLICY/10
        )
        scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_critic, T_max=NUMBER_OF_EPISODES, eta_min=LEARNING_RATE_CRITIC/10
        )

    buffer = RolloutBuffer(NUMBER_OF_STEPS, obs_dim, act_dim)

    obs, _ = env.reset(seed=SEED)
    episode_rewards = []
    policy_losses = []
    value_losses = []
    best_total = -1000
    for update in range(1, NUMBER_OF_EPISODES + 1):
        buffer.ptr = 0
        rollout_reward = 0

        for _ in range(NUMBER_OF_STEPS):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a, lp, v = model.get_action(obs_t, act_low, act_high)
            a_np, lp_np, v_np = a.cpu().numpy()[0], lp.cpu().item(), v.cpu().item()
            obs2, r, done, trunc, _ = env.step(a_np)
            buffer.add(obs, a_np, lp_np, r, v_np, float(done or trunc))
            rollout_reward += r
            obs = obs2 if not (done or trunc) else env.reset()[0]

        with torch.no_grad():
            _, last_v = model.forward(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
        buffer.finish(last_v.cpu().item())

        total_policy, total_value = 0, 0
    
        for _ in range(UPDATE_STEPS):
            for obs_b, act_b, lp_old, adv_b, ret_b in buffer.get_batches(MINIBATCH_SIZE):
                obs_b, act_b = obs_b.to(DEVICE), act_b.to(DEVICE)
                lp_old, adv_b, ret_b = lp_old.to(DEVICE), adv_b.to(DEVICE), ret_b.to(DEVICE)

                dist, val = model.forward(obs_b)
                lp_new = dist.log_prob(act_b).sum(-1)
                ratio = torch.exp(lp_new - lp_old)
                p1 = ratio * adv_b
                p2 = torch.clamp(ratio, 1 - CLIPPING_EPSILON, 1 + CLIPPING_EPSILON) * adv_b
                loss_p = -torch.min(p1, p2).mean()
                loss_v = nn.MSELoss()(val, ret_b)
                ent = dist.entropy().mean()

                entropy_coef = entropy_coef_schedule(update)
                loss = loss_p + VALUE_COEF * loss_v - entropy_coef * ent

                optimizer_policy.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer_policy.step()

                optimizer_critic.zero_grad()
                loss_v.backward()
                optimizer_critic.step()

                total_policy += loss_p.item()
                total_value  += loss_v.item()

        if ANNEAL_LR:
            scheduler_policy.step()
            scheduler_critic.step()

        episode_rewards.append(max(rollout_reward, -100))  
        policy_losses.append(min(max(total_policy, -100), 100))
        value_losses.append(min(max(total_value, -20000), 20000))   

        print(f"[Episode {update:03d}] Reward: {rollout_reward:.2f} | "
              f"Policy Loss: {total_policy:.4f} | Value Loss: {total_value:.4f}")

        axs[0].clear()
        axs[0].plot(episode_rewards, label='Reward')
        axs[0].set_title("Episode Reward")
        axs[0].grid(True)

        axs[1].clear()
        axs[1].plot(policy_losses, label='Policy Loss', color='orange')
        axs[1].set_title("Policy Loss")
        axs[1].grid(True)

        axs[2].clear()
        axs[2].plot(value_losses, label='Value Loss', color='red')
        axs[2].set_title("Value Loss")
        axs[2].grid(True)

        plt.pause(0.01)
        fig.tight_layout()

        plot_path = os.path.join(OUT_DIR, "training_curves.png")
        fig.tight_layout()
        fig.savefig(plot_path)

        if update % SAVE_INTERVAL == 0:
            ckpt = os.path.join(OUT_DIR, f"ppo_{ENV_NAME}_{update}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint saved: {ckpt}")

        if rollout_reward > best_total:
            best_total = rollout_reward
            ckpt = os.path.join(OUT_DIR, f"ppo_{ENV_NAME}_best.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"best saved: {ckpt}")

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final.pth"))
    env.close()
    print("Training complete.")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
