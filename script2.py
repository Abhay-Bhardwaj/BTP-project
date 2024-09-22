import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
from datetime import datetime
import pandas as pd  

# Environment Initialization
grid_size = 18
red_position = (0, 17)
blue_start_position = (17, 0)

# Obstacles (updated and more complex)
obstacles = [
    # Horizontal walls
    (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 8), (2, 8),
    # Vertical walls
    (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
    # Bottom left obstacles
    (15, 5), (15, 6), (15, 7), (15, 8), (16, 5), (16, 6),
    # Upper right obstacles
    (12, 0), (12, 1),
    # Bottom right obstacles
    (15, 12), (15, 13), (15, 14), (15, 15), (15, 16),
    # Random obstacles in the middle
    (5, 13), (6, 14), (7, 13),
]


# Function to visualize the initial map with obstacles
def visualize_initial_map(grid_size, red_position, blue_start_position, obstacles, title="Initial Map"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True)

    # Draw goal position
    ax.add_patch(patches.Circle((red_position[1] + 0.5, grid_size - red_position[0] - 0.5), 0.5, color='red'))
    
    # Draw start position
    ax.add_patch(patches.Circle((blue_start_position[1] + 0.5, grid_size - blue_start_position[0] - 0.5), 0.5, color='green'))

    # Draw obstacles
    for obstacle in obstacles:
        ax.add_patch(patches.Rectangle((obstacle[1], grid_size - obstacle[0] - 1), 1, 1, color='black'))

    ax.set_title(title)
    plt.show()

# Visualize the initial map
visualize_initial_map(grid_size, red_position, blue_start_position, obstacles)

initial_power = 800
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize Q-table for Q-learning and TD-learning
Q = np.zeros((grid_size, grid_size, initial_power + 1, len(actions)))  # For Q-learning
V = np.zeros((grid_size, grid_size, initial_power + 1))  # Value function for TD-learning
visit_counter = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)  # Track visits for penalties

# Define reward function with visit penalty
def get_reward(new_position, current_power, visit_count):
    if new_position == red_position:
        return 100  # Large reward for reaching the goal
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

# Get policy from Q-table
def get_policy(Q):
    policy = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)
    for x in range(grid_size):
        for y in range(grid_size):
            for power in range(initial_power + 1):
                policy[x, y, power] = np.argmax(Q[x, y, power])
    return policy

# Training function for TD-learning with improved action selection
# Adjusting TD-learning action selection and exploration strategy
def train_td_learning(episodes, initial_alpha, gamma, epsilon, epsilon_decay):
    reward_history = []
    alpha = initial_alpha
    for episode in range(1, episodes + 1):
        state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size), initial_power)
        done = False
        total_reward = 0
        visit_counter.fill(0)

        while not done:
            if np.random.random() < epsilon:  # Epsilon-greedy
                action = np.random.randint(0, len(actions))
            else:
                action_values = []
                for a, action_name in enumerate(actions):
                    new_position = (
                        state[0] + action_dict[action_name][0],
                        state[1] + action_dict[action_name][1]
                    )
                    new_position = (
                        max(0, min(grid_size - 1, new_position[0])),
                        max(0, min(grid_size - 1, new_position[1]))
                    )
                    new_power = state[2] - 1
                    action_values.append(V[new_position[0], new_position[1], max(0, new_power)])
                action = np.argmax(action_values)

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

            visit_counter[new_position[0], new_position[1], max(0, new_power)] += 1
            reward = get_reward(new_position, new_power, visit_counter[new_position[0], new_position[1], max(0, new_power)])
            total_reward += reward

            new_state = (new_position[0], new_position[1], max(0, new_power))
            if new_position == red_position or new_power <= 0:
                done = True

            update_td_value(state, reward, new_state, alpha, gamma)
            state = new_state

        reward_history.append(total_reward)

        # Epsilon decay
        epsilon = max(0.1, epsilon * epsilon_decay)

        # Dynamic learning rate adjustment
        alpha = initial_alpha / (1 + episode / 1000)

        if episode % 1000 == 0:
            print(f"Episode {episode}/{episodes} completed in TD-learning.")

    return V, reward_history

# Get policy from Value function for TD-learning
def get_policy_from_value(V):
    policy = np.zeros((grid_size, grid_size, initial_power + 1), dtype=int)
    for x in range(grid_size):
        for y in range(grid_size):
            for power in range(initial_power + 1):
                action_values = []
                for a, action_name in enumerate(actions):
                    new_position = (
                        x + action_dict[action_name][0],
                        y + action_dict[action_name][1]
                    )
                    new_position = (
                        max(0, min(grid_size - 1, new_position[0])),
                        max(0, min(grid_size - 1, new_position[1]))
                    )
                    new_power = power - 1
                    action_values.append(V[new_position[0], new_position[1], max(0, new_power)])
                policy[x, y, power] = np.argmax(action_values)
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

# Function to create a directory for each run
def create_run_directory(base_dir="runs"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, current_time)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Save Q-table, policy, reward history, and plots to files
def save_run_data(Q, policy, reward_history, run_dir, algo_name, q_path=None, td_path=None, plot_fig=None):
    # Save Q-table/Value function
    np.save(os.path.join(run_dir, f'{algo_name}_table.npy'), Q)
    
    # Save policy
    np.save(os.path.join(run_dir, f'{algo_name}_policy.npy'), policy)
    
    # Save reward history as CSV
    reward_df = pd.DataFrame(reward_history, columns=["Reward"])
    reward_df.to_csv(os.path.join(run_dir, f'{algo_name}_reward_history.csv'), index_label="Episode")
    
    # Save paths as images if provided
    if q_path:
        q_path.savefig(os.path.join(run_dir, f'{algo_name}_path.png'))
    if td_path:
        td_path.savefig(os.path.join(run_dir, f'{algo_name}_td_path.png'))

    # Save reward plot if provided
    if plot_fig:
        plot_fig.savefig(os.path.join(run_dir, f'{algo_name}_reward_history.png'))
    
    print(f"{algo_name} run data saved to: {run_dir}")

# Visualization of paths with the option to save them
def visualize_paths(logged_paths, final_path, grid_size, red_position, blue_start_position, obstacles, visit_counter=None, title="Path Visualization"):
    fig, ax = plt.subplots(figsize=(7, 7))
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

    draw_grid(ax, final_path, title)
    plt.show()
    return fig  # Return figure for saving

# Run training for both Q-learning and TD-learning and save the data
def run_experiment_and_save():
    episodes = 20000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.99
    epsilon_decay = 0.99


    # Q-learning
    Q, q_reward_history = train_q_learning(episodes, alpha, gamma, epsilon, epsilon_decay)
    q_policy = get_policy(Q)
    
    # Visualize and save Q-learning path
    q_final_path = simulate_game(q_policy, initial_power)
    q_path_fig = visualize_paths(q_final_path[:-1], q_final_path, grid_size, red_position, blue_start_position, obstacles, title="Q-learning Path")

    # TD-learning
    V, td_reward_history = train_td_learning(episodes, alpha, gamma, epsilon, epsilon_decay)
    td_policy = get_policy_from_value(V)

    # Visualize and save TD-learning path
    td_final_path = simulate_game(td_policy, initial_power)
    td_path_fig = visualize_paths(td_final_path[:-1], td_final_path, grid_size, red_position, blue_start_position, obstacles, title="TD-learning Path")

    # Plot and save reward history comparison
    reward_fig, ax = plt.subplots()
    ax.plot(q_reward_history, label='Q-learning')
    ax.plot(td_reward_history, label='TD-learning')
    ax.legend()
    ax.set_title('Reward History')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total Reward')

    # Create run directory
    run_dir = create_run_directory()

    # Save data and plots
    save_run_data(Q, q_policy, q_reward_history, run_dir, "Q_learning", q_path=q_path_fig, plot_fig=reward_fig)
    save_run_data(V, td_policy, td_reward_history, run_dir, "TD_learning", td_path=td_path_fig, plot_fig=reward_fig)

    # Compare path lengths
    print(f"Q-learning path length: {len(q_final_path)}")
    print(f"TD-learning path length: {len(td_final_path)}")

# Execute the experiment and save the results
run_experiment_and_save()
