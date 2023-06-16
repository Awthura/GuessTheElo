import pandas as pd
import chess.pgn
from datetime import datetime, timedelta
import io

# Load data from CSV file
csv_file_path = 'src/data/club_games_data.csv'
df = pd.read_csv(csv_file_path)

# Convert 'pgn' column to string type
df['pgn'] = df['pgn'].astype(str)

# Filter games from 23rd May and 24th May 2021
filtered_games = []
for index, row in df.iterrows():
    game = chess.pgn.read_game(io.StringIO(row['pgn']))
    date_str = game.headers.get('Date')
    game_date = datetime.strptime(date_str, '%Y.%m.%d')
    start_date = datetime(2021, 5, 20)
    end_date = datetime(2021, 5, 24)
    if start_date <= game_date <= end_date and row['time_class'] == 'blitz':
        filtered_games.append(row)

# Create a DataFrame for the filtered games
filtered_df = pd.DataFrame(filtered_games)

# Save the filtered games as a CSV file
filtered_csv_file_path = 'sample.csv'
filtered_df.to_csv(filtered_csv_file_path, index=False)

print("Sample CSV file created successfully.")



