import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd
from datetime import datetime

# Environment Initialization
grid_size = 25
red_position = (0, 24)  # Zero-indexed (1, 25)
blue_start_position = (24, 0)  # Zero-indexed (25, 1)
obstacles = [
    (4, 1), (4, 2), (4, 3), (6, 3), (2, 5), (2, 6), (2, 7),
    (10, 10), (15, 15), (20, 20), (5, 5), (12, 12), (18, 18),
    (7, 14), (14, 7), (21, 4), (3, 21), (10, 15), (15, 10),
    (23, 22), (22, 23)
]

initial_power = 250
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, initial_power + 1, len(actions)))  # (grid_x, grid_y, power, actions)

# Define reward function
def get_reward(new_position, current_power):
    if new_position == red_position:
        return 100  # Large reward for reaching the goal
    elif new_position in obstacles:
        return -10  # Large penalty for hitting an obstacle
    else:
        return -1  # Small penalty for moving


# Update Q-table using Q-learning algorithm
def update_q_table(state, action, reward, new_state, alpha, gamma, done):
    current_q = Q[state[0], state[1], state[2], action]  # Get current Q-value
    max_future_q = 0 if done else np.max(Q[new_state[0], new_state[1], new_state[2]])  # Max Q for next state
    
    # Update Q-value using the Q-learning formula
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
    
    Q[state[0], state[1], state[2], action] = new_q  # Save updated Q-value

# Create a directory based on the current date and time
def create_run_directory(base_dir="runs"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, current_time)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Save Q-table, policy, and reward history to files
def save_run_data(Q, policy, reward_history, run_dir):
    np.save(os.path.join(run_dir, 'Q_table.npy'), Q)
    np.save(os.path.join(run_dir, 'policy.npy'), policy)
    
    # Save reward history as CSV
    reward_df = pd.DataFrame(reward_history, columns=["Episode", "Reward"])
    reward_df.to_csv(os.path.join(run_dir, 'reward_history.csv'), index=False)
    
    print(f"Run data saved to: {run_dir}")

# Training function with path logging and reward tracking
def train(episodes, alpha, gamma, epsilon, epsilon_decay, log_every=10000):
    logged_paths = {}
    reward_history = []
    
    for episode in range(1, episodes + 1):
        state = (blue_start_position[0], blue_start_position[1], initial_power)  # Initial state
        path = [state]
        done = False
        total_reward = 0
        
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(Q[state[0], state[1], state[2]])  # Exploit: choose the best action from Q-table
            else:
                action = np.random.randint(0, len(actions))  # Explore: choose a random action

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
                new_power -= 10  # Additional penalty for hitting an obstacle

            new_state = (new_position[0], new_position[1], max(0, new_power))
            reward = get_reward(new_position, new_power)
            total_reward += reward

            # Check if the episode is done
            if new_position == red_position or new_power <= 0:
                done = True  # Episode ends when reaching the goal or running out of power

            # Update Q-table
            update_q_table(state, action, reward, new_state, alpha, gamma, done)

            # Move to new state
            state = new_state
            path.append(state)

            if epsilon > 0.1:
                epsilon = max(0.1, epsilon * epsilon_decay)  # Decay epsilon

        reward_history.append((episode, total_reward))

        # Log the path for specific episodes
        if episode % log_every == 0:
            logged_paths[episode] = path.copy()
            print(f"Episode {episode}/{episodes} logged.")

        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes} completed.")

    return Q, logged_paths, reward_history

# Get policy from Q-table
def get_policy(Q):
    policy = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)
    for x in range(grid_size):
        for y in range(grid_size):
            for power in range(initial_power + 1):
                policy[x, y, power] = np.argmax(Q[x, y, power])
    return policy

# Simulate game using the policy
def simulate_game(policy):
    state = (blue_start_position[0], blue_start_position[1], initial_power)
    path = [state]
    while state[:2] != red_position and state[2] > 0:
        action = policy[state[0], state[1], state[2]]
        new_position = (
            state[0] + action_dict[actions[action]][0],
            state[1] + action_dict[actions[action]][1]
        )
        new_position = (
            max(0, min(grid_size - 1, new_position[0])),
            max(0, min(grid_size - 1, new_position[1]))
        )
        new_power = state[2] - 1
        if new_position in obstacles:
            new_power -= 10
        state = (new_position[0], new_position[1], max(0, new_power))
        path.append(state)
    return path

# Visualization of paths
def visualize_paths(logged_paths, final_path, grid_size, red_position, blue_start_position, obstacles):
    # Function to draw the grid
    def draw_grid(ax, path, title):
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.grid(True)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Draw the red agent (goal)
        ax.add_patch(patches.Rectangle((red_position[1], grid_size - red_position[0] - 1), 1, 1, fill=True, color='red'))

        # Draw the blue agent's starting position
        ax.add_patch(patches.Rectangle((blue_start_position[1], grid_size - blue_start_position[0] - 1), 1, 1, fill=True, color='blue', alpha=0.3))

        # Draw the obstacles
        for obs in obstacles:
            if 0 <= obs[0] < grid_size and 0 <= obs[1] < grid_size:
                ax.add_patch(patches.Rectangle((obs[1], grid_size - obs[0] - 1), 1, 1, fill=True, color='black'))

        # Draw the path the agent took
        for step in path:
            ax.add_patch(patches.Rectangle((step[1], grid_size - step[0] - 1), 1, 1, fill=True, color='green', alpha=0.5))

        ax.invert_yaxis()  # Reverse the y-axis to start from the bottom

        # Set x-ticks and y-ticks
        ax.set_xticks(np.arange(0, grid_size, 5))
        ax.set_xticklabels(np.arange(1, grid_size + 1, 5))  
        ax.set_yticks(np.arange(0, grid_size, 5))
        ax.set_yticklabels(np.arange(grid_size, 0, -5))  

    # Draw logged paths
    for episode, path in logged_paths.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_grid(ax, path, title=f'Episode {episode}')

    # Draw the final path using learned policy
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_grid(ax, final_path, title='Final Learned Path')
    plt.show()

# Main Execution
episodes = 100000  # You can increase this to more (e.g., 50000 or 100000) for better results
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.9995  # Epsilon decay rate per episode

Q, logged_paths, reward_history = train(episodes, alpha, gamma, epsilon, epsilon_decay)

policy = get_policy(Q)  # Extract policy from learned Q-table
final_path = simulate_game(policy)  # Simulate a game using the final policy

run_dir = create_run_directory()  # Create a directory for this run's data
save_run_data(Q, policy, reward_history, run_dir)  # Save Q-table, policy, and reward history

visualize_paths(logged_paths, final_path, grid_size, red_position, blue_start_position, obstacles)  # Visualize paths
