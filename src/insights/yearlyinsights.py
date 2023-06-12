import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import chess.pgn

# Load data from JSON file
with open('src/data/dataset.json', 'r') as json_file:
    data = json.load(json_file)

# Extract date from "pgn" field of each item in the list
dates = []
for item in data:
    pgn = item['pgn']
    game = chess.pgn.read_game(io.StringIO(pgn))
    date = game.headers.get('Date')
    if date:
        dates.append(date.split('.')[0])  # Extract year from the date and append to the list

# Create a Pandas DataFrame from the dates list
df = pd.DataFrame({'Year': dates})

# Generate bar chart for each year
df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue')

# Set the labels for x and y axes
plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Number of Games per Year")

# Show the plot
plt.show()

# Close the plot
plt.close()