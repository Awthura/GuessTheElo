import json
import pandas as pd
import matplotlib.pyplot as plt
import chess.pgn
import io

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
        year, month, _ = date.split('.')
        if year == '2021':
            dates.append(month)

# Create a Pandas DataFrame from the dates list
df = pd.DataFrame({'Month': dates})

# Generate bar chart for each month in 2021
df['Month'].value_counts().sort_index().plot(kind='bar', color='skyblue')

# Set the labels for x and y axes
plt.xlabel("Month")
plt.ylabel("Count")
plt.title("Number of Games per Month in 2021")

# Show the plot
plt.show()

# Close it
plt.close()