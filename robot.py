import numpy as np
import random
import matplotlib.pyplot as plt

# =====================
# 環境設定
# =====================
H, W = 5, 5

WALLS = {
    (0,0),(0,1),(0,2),(0,3),(0,4),
    (4,0),(4,1),(4,2),(4,3),(4,4),
    (1,0),(2,0),(3,0),
    (1,4),(2,4),(3,4),
}

START = (1, 1)
GOAL  = (3, 3)

# 行動
# 0: 前進, 1: 左旋回, 2: 右旋回
ACTIONS = [0, 1, 2]
ARROWS = {0: '↑', 1: '←', 2: '→'}

# =====================
# Qテーブル
# state = (front, left, right)
# =====================
Q = np.zeros((3, 3, 3, len(ACTIONS)))

alpha = 0.01    # 学習率
gamma = 0.3    # 割引率
epsilon = 0.01  # 探索率
EPISODES = 1000

# =====================
# 距離センサ
# =====================
def get_distance(state, direction):
    r, c = state
    dist = 0

    while True:
        if direction == 'up':
            r -= 1
        elif direction == 'left':
            c -= 1
        elif direction == 'right':
            c += 1

        if (r, c) in WALLS:
            break

        dist += 1
        if dist >= 2:
            break

    return dist

def get_sensor_state(state):
    front = get_distance(state, 'up')
    left  = get_distance(state, 'left')
    right = get_distance(state, 'right')
    return (front, left, right)

# =====================
# 行動実行
# =====================
def step(state, action):
    r, c = state

    if action == 0:      # 前進
        r -= 1
    elif action == 1:    # 左旋回
        c -= 1
    elif action == 2:    # 右旋回
        c += 1

    next_state = (r, c)

    if next_state in WALLS:
        return state, -10, True

    if next_state == GOAL:
        return next_state, 10, True

    return next_state, -1, False

# =====================
# 学習
# =====================
episode_rewards = []

for ep in range(EPISODES):
    state = START
    done = False
    total_reward = 0

    while not done:
        s = get_sensor_state(state)

        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(Q[s])

        next_state, reward, done = step(state, action)
        ns = get_sensor_state(next_state)

        # Q-learning 更新
        Q[s][action] += alpha * (
            reward + gamma * np.max(Q[ns]) - Q[s][action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

# =====================
# 学習曲線の表示
# =====================
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Progress (Robot Q-learning)")
plt.grid()
plt.show()

# =====================
# 学習後の方策表示
# =====================
print("\n=== Learned Policy (sensor-based) ===")
for f in range(3):
    for l in range(3):
        for r in range(3):
            a = np.argmax(Q[f, l, r])
            print(f"state(front={f}, left={l}, right={r}) -> {ARROWS[a]}")
