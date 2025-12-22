import numpy as np
import random

H, W = 4, 12
START = (0, 0)
GOAL = (0, 11)
CLIFF = [(1, i) for i in range(1, 11)]

#print(CLIFF)

ACTIONS = [0, 1, 2, 3]  # up, down, left, right

def step(state, action):
    r, c = state

    if action == 0: r -= 1
    if action == 1: r += 1
    if action == 2: c -= 1
    if action == 3: c += 1

    r = max(0, min(H-1, r))
    c = max(0, min(W-1, c))
    next_state = (r, c)

    reward = -1
    done = False


    if next_state in CLIFF:
        reward = -100
        next_state = START

    if next_state == GOAL:
        done = True

    return next_state, reward, done

Q_q = np.zeros((H, W, 4))
alpha, gamma, epsilon = 0.1, 0.9, 0.1

for episode in range(500):
    state = START

    while True:
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(Q_q[state])

        next_state, reward, done = step(state, action)

        Q_q[state][action] += alpha * (
            reward + gamma * np.max(Q_q[next_state]) - Q_q[state][action]
        )

        state = next_state
        if done:
            break

ARROWS = {
    0: '↑',  # up
    1: '↓',  # down
    2: '←',  # left
    3: '→',  # right
}

def print_policy(Q, title):
    print(f"\n=== {title} ===")

    for r in range(H):
        row = ""
        for c in range(W):
            state = (r, c)

            if state == START:
                row += " S "
            elif state == GOAL:
                row += " G "
            elif state in CLIFF:
                row += " C "
            else:
                action = np.argmax(Q[state])
                row += f" {ARROWS[action]} "
        print(row)

print_policy(Q_q, "Q-learning policy")