import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from JSON file
with open('src/data/feature_engineered.json', 'r') as file:
    data = json.load(file)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)
# Convert a column from string to integer
df['white_rating'] = df['white_rating'].astype(int)
df['black_rating'] = df['black_rating'].astype(int)
# Calculate correlation matrix
plot_df = df[["white_rating", "black_rating","area_ratio", "area_ratio_rev", "relative_game_advantage", "Advantage swings", "White blunders", "Black blunders", "relative_area_white", "relative_area_black"]]
correlation_matrix = plot_df.corr()
print(correlation_matrix)

# Generate correlation matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close()