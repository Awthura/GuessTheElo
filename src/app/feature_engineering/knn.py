import json
import matplotlib.pyplot as plt

file_dataset = '../../data/preprocess.json'
image_out_first = '../../insights/moves_rating_white.png'
image_out_second = '../../insights/moves_rating_black.png'

# Open the JSON file
with open(file_dataset) as file:
    chess_data = json.load(file)

white_moves = []
white_rating = []
black_moves = []
black_rating = []

# Access the 'pgn' variable and print its value
for game in chess_data:
    moves = game['moves_san']
    moves = moves.split(' ')
    moves = [moves[i] for i in range(0, len(moves), 2)]
    try:
        white = [moves[i] for i in range(0, 6, 2)]
        black = [moves[i] for i in range(1, 7, 2)]
        white_moves.append(tuple(white))
        black_moves.append(tuple(black))
        white_rating.append(game['white_bin'])
        black_rating.append(game['black_bin'])
    except:
        continue

# Sort white_moves and white_rating based on white_rating
sorted_data_white = sorted(zip(white_moves, white_rating), key=lambda x: x[1])
sorted_data_black = sorted(zip(black_moves, black_rating), key=lambda x: x[1])
white_moves, white_rating = zip(*sorted_data_white)
black_moves, black_rating = zip(*sorted_data_black)

white_moves, white_rating = list(white_moves), list(white_rating)
black_moves, black_rating = list(black_moves), list(black_rating)

white_moves_dict = {}
black_moves_dict = {}

for move in white_moves:
    if move in white_moves_dict.keys():
        white_moves_dict[move] += 1
    else:
        white_moves_dict[move] = 1

for move in black_moves:
    if move in black_moves_dict.keys():
        black_moves_dict[move] += 1
    else:
        black_moves_dict[move] = 1

for i, key in enumerate(white_moves_dict.keys()):
    white_moves_dict[key] = (i, white_moves_dict[key])

for i, key in enumerate(black_moves_dict.keys()):
    black_moves_dict[key] = (i, black_moves_dict[key])

for i in range(len(white_moves)):
    if white_moves[i] in white_moves_dict.keys():
        white_moves[i] = white_moves_dict[white_moves[i]][0]
    if black_moves[i] in black_moves_dict.keys():
        black_moves[i] = black_moves_dict[black_moves[i]][0]


plt.scatter(white_moves, white_rating)
plt.xlabel('White Moves')
plt.ylabel('White Rating')
plt.title('White Moves vs Rating')
plt.savefig(image_out_first, dpi=300)
plt.show()

plt.scatter(black_moves, black_rating)
plt.xlabel('Black Moves')
plt.ylabel('Black Rating')
plt.title('Black Moves vs Rating')
plt.savefig(image_out_second, dpi=300)
plt.show()
