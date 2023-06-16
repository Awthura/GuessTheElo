import json
import matplotlib.pyplot as plt

# Load the JSON data from the file
json_file_path = 'src/data/sample.json'
with open(json_file_path) as file:
    data = json.load(file)

# Extract the ratings from the JSON data
ratings = [int(game['white_rating']) for game in data] + [int(game['black_rating']) for game in data]

# Plot the histogram
plt.hist(ratings, bins=20, color='skyblue', edgecolor='black')

# Set the title and labels
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

# Find the highest and lowest rating
highest_rating = max(ratings)
lowest_rating = min(ratings)

# Print the highest and lowest rating
print(f'Highest Rating: {highest_rating}')
print(f'Lowest Rating: {lowest_rating}')

# Add text to the plot
plt.text(highest_rating, 50, f'Highest: {highest_rating}', ha='right', va='center', fontsize=10, color='red')
plt.text(lowest_rating, 50, f'Lowest: {lowest_rating}', ha='left', va='center', fontsize=10, color='red')

# Show the plot
plt.tight_layout()
plt.show()
plt.close()

