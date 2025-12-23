import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")


env.unwrapped.x_threshold = 10
env.unwrapped.theta_threshold_radians = 90 * np.pi / 180


state, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    print("action",action)
    state, reward, terminated, truncated, info = env.step(action)
    print("state:",state)
    print("reward",reward)
    print("terminated",terminated)
    print("truncated",truncated)
    print("info",info)

    if terminated or truncated:
        state, info = env.reset()
        print('reset')

env.close()
