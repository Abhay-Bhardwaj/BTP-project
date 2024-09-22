import pandas as pd
import matplotlib.pyplot as plt
import os

file_name = "20240922_170901"
td_reward_file = f'C:/Users/abhay/OneDrive/Desktop/BTP/grid_based_python/runs/{file_name}/Q_learning_reward_history.csv'
q_learning_reward_file = f'C:/Users/abhay/OneDrive/Desktop/BTP/grid_based_python/runs/{file_name}/TD_learning_reward_history.csv'
output_dir = f'C:/Users/abhay/OneDrive/Desktop/BTP/grid_based_python/runs/{file_name}/'
output_file = os.path.join(output_dir, 'comparison_E_vs_R.png')

# Check if the directory exists
if not os.path.exists(output_dir):
    print(f"Directory {output_dir} does not exist. Creating the directory.")
    os.makedirs(output_dir)

# Read the CSV files
td_rewards = pd.read_csv(td_reward_file)
q_learning_rewards = pd.read_csv(q_learning_reward_file)

# Calculate the moving average
window_size = 500
td_rewards['Moving_Average'] = td_rewards['Reward'].rolling(window=window_size).mean()
q_learning_rewards['Moving_Average'] = q_learning_rewards['Reward'].rolling(window=window_size).mean()

# Plot the reward histories
plt.figure(figsize=(12, 6))

# Plot TD rewards (every 100th episode)
plt.plot(td_rewards['Episode'][::window_size], td_rewards['Reward'][::window_size], label='TD Learning', color='blue', alpha=0.3)
plt.plot(td_rewards['Episode'], td_rewards['Moving_Average'], label='TD Learning (Moving Average)', color='blue')

# Plot Q-learning rewards (every 100th episode)
plt.plot(q_learning_rewards['Episode'][::window_size], q_learning_rewards['Reward'][::window_size], label='Q Learning', color='red', alpha=0.3)
plt.plot(q_learning_rewards['Episode'], q_learning_rewards['Moving_Average'], label='Q Learning (Moving Average)', color='red')

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Episode vs Reward Comparison: TD Learning vs Q Learning (Every {window_size} Episodes)')
plt.legend()

# Save the plot
plt.savefig(output_file)
print(f"Plot saved at {output_file}")

# Show the plot
plt.show()