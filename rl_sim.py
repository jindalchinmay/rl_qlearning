from rl_env import Env
import numpy as np
import time
import os

env = Env()

qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

epochs = 50
gamma = 0.1
epsilon = 0.08
decay = 0.1

for i in range(epochs):
    state, reward, done = env.reset()
    steps = 0

    while not done:
        os.system('clear')
        print("epoch #", i+1, "/", epochs)
        env.render()
        time.sleep(0.05)

        steps += 1

        if np.random.uniform() < epsilon:
            action = env.randomAction()
        else:
            action = qtable[state].index(max(qtable[state]))

        next_state, reward, done = env.step(action)

        qtable[state][action] = reward + gamma * max(qtable[next_state])

        state = next_state
    epsilon -= decay*epsilon

    print("\nDone in", steps, "steps".format(steps))
    time.sleep(0.8)



    