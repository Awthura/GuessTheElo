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
        if year == '2021' and month == '05':
            dates.append(int(day))

# Create a Pandas DataFrame from the dates list
df = pd.DataFrame({'Day': dates})

# Define date intervals in May
date_intervals = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-31']

# Convert the 'Day' column to numeric values
df['Day'] = pd.to_numeric(df['Day'])

# Create a new column for date intervals
df['Date Interval'] = pd.cut(df['Day'], bins=[0, 5, 10, 15, 20, 25, 31], labels=date_intervals)

# Generate bar chart for each date interval in May
df['Date Interval'].value_counts().sort_index().plot(kind='bar', color='skyblue')

# Set the labels for x and y axes
plt.xlabel("Date Interval")
plt.ylabel("Count")
plt.title("Number of Games per Date Interval in May 2021")

# Show the plot
plt.show()

# Close
plt.close()