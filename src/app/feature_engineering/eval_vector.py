import chess
import chess.pgn
import chess.engine
import json
import re 

# Path to Stockfish engine executable
stockfish_path = '../../../Stockfish/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe'

# Function to evaluate the position using Stockfish
def evaluate_position(position):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Evaluate the position using Stockfish
    analysis = engine.analyse(position, chess.engine.Limit(depth=10))
    score = analysis["score"]

    engine.quit()

    return str(score)

# Load the preprocess.json file
with open('../../data/feature_engineered.json', 'r') as file:
    data = json.load(file)

# Iterate over the games and calculate evaluation values after each move
for entry in data:
    moves_san = entry['moves_san'].split()

    evaluations = []  # List to store evaluations after each move

    # Create a chess board and play the moves on it
    board = chess.Board()
    for move in moves_san:
        board.push_san(move)

        # Evaluate the position after each move using Stockfish
        evaluation = evaluate_position(board)
        evaluations.append(evaluation)

    # Update the entry with the evaluation values after each move
    entry['evaluations'] = evaluations
    print("Success")

# Save the updated data to a new file feature_engineered.json
with open('../../data/evaluation.json', 'w') as file:
    json.dump(data, file, indent=4)
