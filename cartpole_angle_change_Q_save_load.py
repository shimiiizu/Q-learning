import gymnasium as gym
import numpy as np

# =====================
# 環境の準備
# =====================
env = gym.make("CartPole-v1", render_mode="human")

# 終了条件をゆるく（学習しやすく）
env.unwrapped.x_threshold = 10
env.unwrapped.theta_threshold_radians = 90 * np.pi / 180

# =====================
# Q-learningの設定
# =====================
ACTIONS = [0, 1]  # 0: 左, 1: 右
N_STATE = 3       # θ̇を3段階に離散化

#Q = np.zeros((N_STATE, len(ACTIONS)))
Q = np.load("Qtable/q_table.npy")

alpha = 0.1    # 学習率
gamma = 0.99   # 割引率
epsilon = 0.05  # 探索率

# =====================
# θ̇の離散化関数
# =====================
def discretize(theta_dot):
    if theta_dot < -0.5:
        return 0  # 左に速く倒れる
    elif theta_dot > 0.5:
        return 2  # 右に速く倒れる
    else:
        return 1  # ほぼ直立

# =====================
# 学習ループ
# =====================
state, info = env.reset()

for episode in range(100):
    state, info = env.reset()
    total_reward = 0

    for step in range(1000):
        # 状態から θ̇ を取り出す
        theta_dot = state[3]
        state_idx = discretize(theta_dot)

        # -------- 行動選択（ε-greedy）--------
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state_idx])

        # -------- 環境を1ステップ進める --------
        next_state, reward, terminated, truncated, info = env.step(action)

        next_theta_dot = next_state[3]
        next_state_idx = discretize(next_theta_dot)

        # -------- Q-learning更新 --------
        Q[state_idx, action] += alpha * (
            reward
            + gamma * np.max(Q[next_state_idx])
            - Q[state_idx, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    # εを少しずつ減らす（探索 → 活用）
    epsilon = max(0.01, epsilon * 0.99)

    print(f"Episode {episode:2d} | total_reward = {total_reward}")

np.save("Qtable/q_table.npy", Q)
print("学習後のQテーブル:")
print(Q)

env.close()
