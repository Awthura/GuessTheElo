import json
import re

file_dataset = 'src/data/preprocess.json'
file_output = 'src/data/preprocess.json'

# Open the JSON file
with open(file_dataset) as file:
    chess_data = json.load(file)

filtered_data = []

# Access the 'pgn' variable and print its value
for game in chess_data:
    moves_with_time = game['moves_with_time']

    # Remove time annotations using regex
    moves_without_time = re.sub(r'\{[^}]+\}', '', moves_with_time).strip()

    # Remove move numbers and end result from the move notation
    moves_without_numbers_result = re.sub(r'\d+\.{1,3}\s*|\d-\d$', '', moves_without_time).strip()

    filtered_item = {
        'white_username': game['white_username'],
        'black_username': game['black_username'],
        'white_rating': game['white_rating'],
        'black_rating': game['black_rating'],
        'white_result': game['white_result'],
        'black_result': game['black_result'],
        'time_control': game['time_control'],
        'fen': game['fen'],
        'pgn': game['pgn'],
        'moves_with_time': moves_with_time,
        'moves': moves_without_time,
        'moves_san':moves_without_numbers_result
    }
    filtered_data.append(filtered_item)

with open(file_output, 'w') as file:
    json.dump(filtered_data, file, indent=4)

