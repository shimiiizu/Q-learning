import numpy as np
import random

N_STATE = 5
GOAL = 4

def step(state, action):
    if action == 0:
        next_state = max(0, state - 1)
    else:
        next_state = min(N_STATE - 1, state + 1)

    reward = 1 if next_state == GOAL else 0
    done = next_state == GOAL
    return next_state, reward, done


Q = np.zeros((N_STATE, 2))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

def choose_action(state):
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        return np.argmax(Q[state])

for episode in range(200):
    state = 0
    action = choose_action(state)

    while True:
        next_state, reward, done = step(state, action)
        next_action = choose_action(next_state)

        # ★ SARSA 更新式（ここが違う）
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action

        if done:
            break

print(Q)
