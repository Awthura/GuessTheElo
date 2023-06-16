import json

# Read the input JSON data
with open('src/data/preprocess.json') as file:
    data = json.load(file)

# Extract the moves with time
moves_with_time = data['moves_with_time']

# Split the moves with time into individual moves
moves = moves_with_time.split()

# Extract the moves without numbers and result
moves_san = [move for move in moves if not move[0].isdigit() and move != '1-0']

# Update the data with the new moves_san attribute
data['moves_san'] = ' '.join(moves_san)

# Write the updated data to a new JSON file
with open('src/data/preprocess2.json', 'w') as file:
    json.dump(data, file, indent=4)
