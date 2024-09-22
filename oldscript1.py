import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from datetime import datetime

# Environment Initialization
grid_size = 10
red_position = (0, 9)  # Goal position
blue_start_position = (9, 0)  # Start position
obstacles = [
    (1,0),(1, 1), (1, 2), (1, 3), (1, 4),  # Vertical wall
    (2, 4), (3, 4), (4, 4),          # Vertical wall continuation
    (5, 1), (5, 2), (5, 3), (5, 4),  # Another vertical wall
    (6, 2), (7, 2),                  # Vertical wall continuation
    (8, 3),                          # Horizontal blockage
    (3, 6), (4, 6), (5, 6)           # Additional horizontal obstacles
]

initial_power = 100
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize Q-table for Q-learning and TD-learning
Q = np.zeros((grid_size, grid_size, initial_power + 1, len(actions)))  # For Q-learning
V = np.zeros((grid_size, grid_size, initial_power + 1))  # Value function for TD-learning
visit_counter = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)  # Track visits for penalties

# Define reward function with visit penalty
def get_reward(new_position, current_power, visit_count):
    if new_position == red_position:
        return 200  # Large reward for reaching the goal
    elif new_position in obstacles:
        return -100  # Large penalty for hitting an obstacle
    else:
        revisit_penalty = -10 * visit_count  # Add penalty based on how many times a position is visited
        return -1 + revisit_penalty  # Small penalty for each move, plus revisit penalty

# Update Q-table using Q-learning algorithm
def update_q_table(state, action, reward, new_state, alpha, gamma, done):
    current_q = Q[state[0], state[1], state[2], action]  # Get current Q-value
    max_future_q = 0 if done else np.max(Q[new_state[0], new_state[1], new_state[2]])  # Max Q for next state
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)  # Update Q-value
    Q[state[0], state[1], state[2], action] = new_q

# Update Value function using TD-learning
def update_td_value(state, reward, new_state, alpha, gamma):
    current_value = V[state[0], state[1], state[2]]
    new_value = (1 - alpha) * current_value + alpha * (reward + gamma * np.max(V[new_state[0], new_state[1], new_state[2]]))
    V[state[0], state[1], state[2]] = new_value

# Training function for Q-learning with revisit penalty
def train_q_learning(episodes, alpha, gamma, epsilon, epsilon_decay):
    reward_history = []
    for episode in range(1, episodes + 1):
        # Random initial state
        state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size), initial_power)
        done = False
        total_reward = 0
        visit_counter.fill(0)  # Reset visit counter for each episode
        
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(Q[state[0], state[1], state[2]])  # Exploit using Q-table
            else:
                action = np.random.randint(0, len(actions))  # Explore new actions randomly

            # Calculate new position and power
            new_position = (
                state[0] + action_dict[actions[action]][0],
                state[1] + action_dict[actions[action]][1]
            )
            new_position = (
                max(0, min(grid_size - 1, new_position[0])),
                max(0, min(grid_size - 1, new_position[1]))
            )

            new_power = state[2] - 1  # Deduct power for each move
            if new_position in obstacles:
                new_power -= 10  # Penalty for hitting an obstacle

            # Update visit counter for the new position
            visit_counter[new_position[0], new_position[1], max(0, new_power)] += 1

            # Get reward with visit penalty
            reward = get_reward(new_position, new_power, visit_counter[new_position[0], new_position[1], max(0, new_power)])
            total_reward += reward

            new_state = (new_position[0], new_position[1], max(0, new_power))

            if new_position == red_position or new_power <= 0:
                done = True

            update_q_table(state, action, reward, new_state, alpha, gamma, done)
            state = new_state

            if epsilon > 0.1:
                epsilon = max(0.1, epsilon * epsilon_decay)  # Decay epsilon

        reward_history.append(total_reward)
        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes} completed in Q-learning with revisit penalty.")

    return Q, reward_history

# Training function for TD-learning with revisit penalty
def train_td_learning(episodes, alpha, gamma, epsilon, epsilon_decay):
    reward_history = []
    for episode in range(1, episodes + 1):
        # Random initial state
        state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size), initial_power)
        done = False
        total_reward = 0
        visit_counter.fill(0)  # Reset visit counter for each episode

        while not done:
            if np.random.random() > epsilon:
                # Exploit the learned value function (choose action with highest value for next state)
                best_action = None
                best_value = float('-inf')

                # Search through all actions and evaluate the value of the next state
                for a, action_name in enumerate(actions):
                    new_position = (
                        state[0] + action_dict[action_name][0],
                        state[1] + action_dict[action_name][1]
                    )
                    new_position = (
                        max(0, min(grid_size - 1, new_position[0])),
                        max(0, min(grid_size - 1, new_position[1]))
                    )
                    if V[new_position[0], new_position[1], state[2] - 1] > best_value:
                        best_value = V[new_position[0], new_position[1], state[2] - 1]
                        best_action = a

                action = best_action
            else:
                action = np.random.randint(0, len(actions))  # Explore new actions randomly

            # Calculate new position and power
            new_position = (
                state[0] + action_dict[actions[action]][0],
                state[1] + action_dict[actions[action]][1]
            )
            new_position = (
                max(0, min(grid_size - 1, new_position[0])),
                max(0, min(grid_size - 1, new_position[1]))
            )
            new_power = state[2] - 1  # Deduct power for each move
            if new_position in obstacles:
                new_power -= 10  # Penalty for hitting an obstacle

            # Update visit counter for the new position
            visit_counter[new_position[0], new_position[1], max(0, new_power)] += 1

            # Get reward with visit penalty
            reward = get_reward(new_position, new_power, visit_counter[new_position[0], new_position[1], max(0, new_power)])
            total_reward += reward

            new_state = (new_position[0], new_position[1], max(0, new_power))

            if new_position == red_position or new_power <= 0:
                done = True

            update_td_value(state, reward, new_state, alpha, gamma)
            state = new_state

            if epsilon > 0.1:
                epsilon = max(0.1, epsilon * epsilon_decay)  # Decay epsilon

        reward_history.append(total_reward)
        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes} completed in TD-learning with revisit penalty.")

    return V, reward_history

# Get policy from Q-table
def get_policy(Q):
    policy = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)
    for x in range(grid_size):
        for y in range(grid_size):
            for power in range(initial_power + 1):
                policy[x, y, power] = np.argmax(Q[x, y, power])
    return policy

# Simulate game using the policy
def simulate_game(policy, initial_power):
    state = (blue_start_position[0], blue_start_position[1], initial_power)
    path = [state]
    while state[:2] != red_position and state[2] > 0:
        action = policy[state[0], state[1], state[2]]  # Use policy from Q-table
        new_position = (
            state[0] + action_dict[actions[action]][0],
            state[1] + action_dict[actions[action]][1]
        )
        new_position = (
            max(0, min(grid_size - 1, new_position[0])),
            max(0, min(grid_size - 1, new_position[1]))
        )
        new_power = state[2] - 1
        state = (new_position[0], new_position[1], new_power)
        path.append(state)
    return path

# Print policy table
def print_policy(policy, title):
    print(f"\n{title} Policy Table:")
    for power in range(initial_power + 1):
        print(f"\nPower Level: {power}")
        for x in range(grid_size):
            for y in range(grid_size):
                print(f"{actions[policy[x, y, power]]:>6}", end=" ")
            print()

# Visualization of paths
def visualize_paths(logged_paths, final_path, grid_size, red_position, blue_start_position, obstacles, visit_counter=None):
    def draw_grid(ax, path, title, visit_counter=None):
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(True)

        # Optionally display the visit counts as a heatmap
        if visit_counter is not None:
            heatmap = np.sum(visit_counter, axis=2)  # Sum over power levels
            ax.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[0, grid_size, 0, grid_size])

        for pos in path:
            ax.add_patch(patches.Circle((pos[1] + 0.5, grid_size - pos[0] - 0.5), 0.3, color='blue', alpha=0.5))

        ax.add_patch(patches.Circle((red_position[1] + 0.5, grid_size - red_position[0] - 0.5), 0.5, color='red'))
        ax.add_patch(patches.Circle((blue_start_position[1] + 0.5, grid_size - blue_start_position[0] - 0.5), 0.5, color='green'))

        for obstacle in obstacles:
            ax.add_patch(patches.Rectangle((obstacle[1], grid_size - obstacle[0] - 1), 1, 1, color='black'))

        ax.set_title(title)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    draw_grid(ax1, logged_paths, 'Previous Paths', visit_counter)
    draw_grid(ax2, final_path, 'Optimal Path')
    plt.show()

# Run training for both Q-learning and TD-learning
def run_experiment():
    episodes = 20000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9
    epsilon_decay = 0.995

    # Q-learning
    Q, q_reward_history = train_q_learning(episodes, alpha, gamma, epsilon, epsilon_decay)
    q_policy = get_policy(Q)

    # TD-learning
    V, td_reward_history = train_td_learning(episodes, alpha, gamma, epsilon, epsilon_decay)
    td_policy = get_policy(V)

    # Simulate paths
    q_final_path = simulate_game(q_policy, initial_power)
    td_final_path = simulate_game(td_policy, initial_power)

    # Visualize
    visualize_paths(q_final_path[:-1], q_final_path, grid_size, red_position, blue_start_position, obstacles)
    visualize_paths(td_final_path[:-1], td_final_path, grid_size, red_position, blue_start_position, obstacles)

    # Plot reward history for comparison
    plt.plot(q_reward_history, label='Q-learning')
    plt.plot(td_reward_history, label='TD-learning')
    plt.legend()
    plt.title('Reward History')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.show()

    # Compare path lengths
    print(f"Q-learning path length: {len(q_final_path)}")
    print(f"TD-learning path length: {len(td_final_path)}")

    # # Print policy tables
    # print_policy(q_policy, "Q-learning")
    # print_policy(td_policy, "TD-learning")

# Execute the experiment
run_experiment()