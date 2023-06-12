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
        year, month, day = date.split('.')
        if year == '2021' and month == '05' and 21 <= int(day) <= 25:
            dates.append(day)

# Create a Pandas DataFrame from the dates list
df = pd.DataFrame({'Day': dates})

# Create a new column for the day
df['Day'] = pd.to_numeric(df['Day'])

# Generate bar chart for each day from May 21 to May 25
df['Day'].value_counts().sort_index().plot(kind='bar', color='skyblue')

# Set the labels for x and y axes
plt.xlabel("Day")
plt.ylabel("Count")
plt.title("Number of Games per Day in May 2021 (May 21-25)")

# Show the plot
plt.show()

# Close
plt.close()