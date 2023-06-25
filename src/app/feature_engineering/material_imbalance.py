import chess
import chess.pgn
import chess.engine
import json
import re

# Load the preprocess.json file
with open('../../data/feature_engineered.json', 'r') as file:
    data = json.load(file)

# Iterate over the games and calculate material imbalance after each move
for entry in data:
    moves_san = entry['moves_san'].split()

    material_imbalance_vector = []
    material_imbalance = 0  # Variable to store material imbalance after each move

    # Create a chess board and play the moves on it
    board = chess.Board()
    # Calculate material imbalance after each move
    piece_values = {
        'P': 1,  # Pawn
        'N': 3,  # Knight
        'B': 3,  # Bishop
        'R': 5,  # Rook
        'Q': 9,  # Queen
        'p': 1,  # Lowercase pawn
        'n': 3,  # Lowercase knight
        'b': 3,  # Lowercase bishop
        'r': 5,  # Lowercase rook
        'q': 9   # Lowercase queen
    }   

    for move in moves_san:
        board.push_san(move)

        fen = board.fen()
        fen_parts = fen.split(' ')
        fen_pieces = fen_parts[0]
        # print(fen_pieces)

        white_pieces = re.findall('[PNBRQK]', fen_pieces)
        black_pieces = re.findall('[pnbrqk]', fen_pieces)
        # print(white_pieces)
        # print(black_pieces)
        white_count = sum(piece_values.get(piece, 0) for piece in white_pieces)
        black_count = sum(piece_values.get(piece, 0) for piece in black_pieces)
        # print(white_count)
        # print(black_count)

        material_imbalance = white_count - black_count
        material_imbalance_vector.append(material_imbalance)

    # Update the entry with the evaluation values after each move
    entry['material imbalance'] = ','.join(str(val) for val in material_imbalance_vector)
    print("Success")


# Save the updated data to a new file feature_engineered.json
with open('../../data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)
