import pandas as pd
import matplotlib.pyplot as plt

# Function to plot episodes and rewards from a CSV file
def plot_rewards_from_csv(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Check if the required columns are present
    if 'Episode' not in data.columns or 'Reward' not in data.columns:
        raise ValueError("CSV file must contain 'episode' and 'reward' columns.")

    # Extract episode numbers and rewards
    episodes = data['Episode']
    rewards = data['Reward']

    # Filter for every 1000 episodes
    filtered_episodes = episodes[episodes % 1000 == 0]
    filtered_rewards = rewards[episodes % 1000 == 0]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_episodes, filtered_rewards, marker='o', linestyle='-', color='b', label='Reward')
    plt.title('Episode vs Reward (Every 1000 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xticks(filtered_episodes)  # Show x-ticks for filtered episodes
    plt.grid()
    plt.legend()
    plt.show()

# Example usage
csv_file = 'C:/Users/abhay/OneDrive/Desktop/BTP/grid_based_python/runs/20240922_122758/reward_history.csv'  # Path to your CSV file
plot_rewards_from_csv(csv_file)
