from rl_env import Env
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Create environment
env = Env()

# Initialize Q-table
qtable = np.zeros((env.stateCount, env.actionCount))

# Hyperparameters
epochs = 50
gamma = 0.9
epsilon = 0.8
decay = 0.05
alpha = 0.1  # Learning rate

# Lists to store metrics
steps_per_episode = []
rewards_per_episode = []

# Training loop
for i in range(epochs):
    state, reward, done = env.reset()
    steps = 0
    total_reward = 0

    while not done:
        # os.system('clear')
        # env.render()
        # time.sleep(0.05)

        steps += 1

        # Choose action (epsilon-greedy policy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.randomAction()
        else:
            action = np.argmax(qtable[state])

        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Update Q-table
        qtable[state][action] = qtable[state][action] + alpha * (
            reward + gamma * np.max(qtable[next_state]) - qtable[state][action]
        )

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon - decay * epsilon, 0.01)

    # Store metrics
    steps_per_episode.append(steps)
    rewards_per_episode.append(total_reward)

    print(f"Epoch {i+1}/{epochs} completed in {steps} steps.")

# Plotting learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

plt.subplot(1, 2, 2)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')

plt.tight_layout()
plt.show()
