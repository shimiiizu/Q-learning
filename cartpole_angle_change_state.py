import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

env.unwrapped.x_threshold = 10
env.unwrapped.theta_threshold_radians = 90 * np.pi / 180

state, info = env.reset()

for _ in range(1000):
    x, x_dot, theta, theta_dot = state

    # ★ θの変化率で判断
    if theta_dot > 0:
        action = 1   # 右に押す
    else:
        action = 0   # 左に押す

    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        state, info = env.reset()

env.close()
