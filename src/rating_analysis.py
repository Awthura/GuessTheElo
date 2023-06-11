import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from JSON file
with open('src/data/dataset.json', 'r') as json_file:
    data = json.load(json_file)

# Create a Pandas DataFrame from the data
df = pd.DataFrame(data)

# Convert rating columns to numeric values
df['white_rating'] = pd.to_numeric(df['white_rating'])
df['black_rating'] = pd.to_numeric(df['black_rating'])

# Create a 2x2 grid of subplots with adjusted spacing
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False, gridspec_kw={'hspace': 0.4, 'wspace': 0.2})

# Contour plot of white's rating vs black's rating
sns.kdeplot(data=df, x='white_rating', y='black_rating', ax=axs[0, 0], cmap='YlOrBr', fill=True, levels=10, thresh=0)

# Set the labels for x and y axes
axs[0, 0].set_xlabel("White's Rating")
axs[0, 0].set_ylabel("Black's Rating")
axs[0, 0].set_title("Density Heatmap of Ratings")

# Histogram of white's and black's ratings
axs[0, 1].hist([df['white_rating'], df['black_rating']], bins=20, color=['skyblue', 'lightcoral'], label=["White's Ratings", "Black's Ratings"])
axs[0, 1].set_xlabel("Rating")
axs[0, 1].set_ylabel("Count")
axs[0, 1].set_title("White and Black Ratings Histogram")
axs[0, 1].legend()

# Custom analysis - Difference in ratings
rating_difference = df['white_rating'] - df['black_rating']
axs[1, 0].hist(rating_difference, bins=20, color='lightgreen', edgecolor='k')
axs[1, 0].set_xlabel("Rating Difference (White - Black)")
axs[1, 0].set_ylabel("Count")
axs[1, 0].set_title("Rating Difference Histogram")

# Scatter plot of white's rating vs black's rating with heatmap-like color scheme
sns.scatterplot(data=df, x='white_rating', y='black_rating', ax=axs[1, 1], palette='YlOrBr', alpha=0.5)

# Set the labels for x and y axes
axs[1, 1].set_xlabel("White's Rating")
axs[1, 1].set_ylabel("Black's Rating")
axs[1, 1].set_title("White's Rating vs Black's Rating")

# Adjust the spacing and layout of subplots
fig.tight_layout()

# Show the plot
plt.show()

# Close the plot window
plt.close()