# Gridworld-use-a-rectangular-grid-to-illustrate-value-functions-for-a-simple-finite-MDP-

## Date : 

## Aim
The aim of this project is to implement a Gridworld environment using Reinforcement Learning concepts to illustrate value functions in a simple finite Markov Decision Process (MDP).

---

# Algorithm

## Iterative Policy Evaluation Algorithm for Gridworld

### Step 1
Initialize the Gridworld environment.

### Step 2
Define:
- States
- Actions
- Rewards
- Transition probabilities

### Step 3
Initialize the value function \( V(s) \) for all states to zero.

### Step 4
For each state:
- Evaluate all possible actions
- Compute the expected return

### Step 5
Update the state value using the Bellman Equation:

Where:
- \( \pi(a|s) \) → Policy
- \( \gamma \) → Discount factor
- \( r \) → Reward
- \( V(s') \) → Next state value

### Step 6
Repeat updates until the value function converges.

### Step 7
Display the final state-value grid.

---

# Program

```
import numpy as np
import gymnasium as gym

# Create environment (modern version)
env = gym.make("CartPole-v1")

# Discretization
buckets = (1, 1, 6, 12)
Q = np.zeros(buckets + (env.action_space.n,))

# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 3000

# Discretize continuous state
def discretize(obs):
    upper = [2.4, 3.0, 0.2095, 3.5]
    lower = [-2.4, -3.0, -0.2095, -3.5]

    ratios = [(obs[i] - lower[i]) / (upper[i] - lower[i]) for i in range(len(obs))]
    new_obs = [int(r * (buckets[i] - 1)) for i, r in enumerate(ratios)]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

    return tuple(new_obs)

# --- TRAINING ---
for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)
    done = False

    while not done:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Slightly improved reward (optional but simple)
        angle = next_obs[2]
        reward = 1 - abs(angle)

        next_state = discretize(next_obs)

        # Q-learning update
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % 500 == 0:
        print(f"Episode {ep}, Epsilon: {epsilon:.3f}")

print("Training Completed!")


# --- TESTING ---
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    state = discretize(obs)
    action = np.argmax(Q[state])

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    total_reward += reward

env.close()
print("Test Reward:", total_reward)
```

---

# Output

```text
State Value Function:

<img width="370" height="173" alt="image" src="https://github.com/user-attachments/assets/f246bd99-ed2b-4824-9675-a85510725bea" />


```

---

# Result

The Gridworld environment was successfully implemented using a rectangular grid to demonstrate value functions in a finite Markov Decision Process (MDP).  
The state values were computed using iterative policy evaluation and the Bellman equation, showing how the agent estimates the long-term reward for each state in the environment.
