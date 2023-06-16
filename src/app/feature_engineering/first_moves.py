import json
import matplotlib.pyplot as plt
from collections import Counter

file_dataset = '../../data/moves.json'
image_out_first = '../../insights/first_moves.png'
image_out_second = '../../insights/second_moves.png'


# Open the JSON file
with open(file_dataset) as file:
    chess_data = json.load(file)

first_moves = []
second_moves = []

for game in chess_data:
    moves = game['moves'].split('{')
    moves = moves[0]
    list_move = moves.split(' ')
    try:
        move = list_move[1]
    except:
        print("not a move")

    first_moves.append(move)

# Count the frequency of each move in first_moves
move_counts = Counter(first_moves)

# Extract the moves and their corresponding frequencies
moves = list(move_counts.keys())
frequencies = list(move_counts.values())

# Plotting the bar graph
plt.bar(moves, frequencies)
plt.xlabel('Moves')
plt.ylabel('Frequency')
plt.title('First Moves Taken by Chess Players')
plt.xticks(rotation=90)

plt.savefig(image_out_first, dpi=300)

plt.show()

for game in chess_data:
    moves = game['moves']
    moves = moves.split('}')
    try:
        list_move = moves[1].split(' ')
        move = list_move[2]
        second_moves.append(move)
    except:
        print('not a move')

# Count the frequency of each move in first_moves
move_counts = Counter(second_moves)

# Extract the moves and their corresponding frequencies
moves = list(move_counts.keys())
frequencies = list(move_counts.values())

# Plotting the bar graph
plt.bar(moves, frequencies)
plt.xlabel('Moves')
plt.ylabel('Frequency')
plt.title('Second Moves Taken by Chess Players')
plt.xticks(rotation=90)

plt.savefig(image_out_second, dpi=300)

plt.show()
