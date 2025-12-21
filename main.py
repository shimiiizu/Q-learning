import numpy as np
import random

# ----- 環境 -----
N_STATE = 5
GOAL = 4

def step(state, action):
    if action == 0:   # left
        next_state = max(0, state - 1)
    else:             # right
        next_state = min(N_STATE - 1, state + 1)

    reward = 1 if next_state == GOAL else 0
    done = next_state == GOAL
    return next_state, reward, done


# ----- Q-learning -----
Q = np.zeros((N_STATE, 2))

alpha = 0.1   # 学習率
gamma = 0.9   # 割引率
epsilon = 0.2 # 探索率

for episode in range(200):
    state = 0

    while True:
        # ε-greedy
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(Q[state])

        next_state, reward, done = step(state, action)

        # Q-learning 更新式
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        if done:
            break

print("学習後のQテーブル:")
print(Q)
