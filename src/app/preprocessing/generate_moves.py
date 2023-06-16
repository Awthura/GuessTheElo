import json

file_dataset = '../../data/sample.json'
file_output = '../../data/moves.json'

# Open the JSON file
with open(file_dataset) as file:
    chess_data = json.load(file)

filtered_data = []

# Access the 'pgn' variable and print its value
for game in chess_data:
    moves = game['pgn'].split('\n\n')
    moves = moves[1]
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
        'moves': moves
    }
    filtered_data.append(filtered_item)

print(moves)

with open(file_output, 'w') as file:
    json.dump(filtered_data, file, indent=4)

