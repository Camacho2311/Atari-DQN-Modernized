import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import torch
import time
from wrapper import AtariWrapper

MODEL_PATH = "log_pong/double_dueldqn/model450.pth"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

env = gym.make("ALE/Pong-v5", render_mode="human")
env = AtariWrapper(env)

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()

obs, info = env.reset()
obs = torch.from_numpy(obs).to(device)
obs = torch.stack((obs, obs, obs, obs)).unsqueeze(0)

total_reward = 0

while True:
    with torch.no_grad():
        action = model(obs).max(1)[1].item()

    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    next_obs = torch.from_numpy(next_obs).to(device)
    next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
    obs = next_obs

    time.sleep(0.01)

    if terminated or truncated:
        break

env.close()
print(f"Final reward: {total_reward}")