import chess
import chess.pgn
import chess.engine
import json
import re 

# Path to Stockfish engine executable
stockfish_path = 'Stockfish/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe'

# Function to evaluate the position using Stockfish
def evaluate_position(position):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Evaluate the position using Stockfish
    analysis = engine.analyse(position, chess.engine.Limit(depth=20))
    score = analysis["score"].relative.score()

    engine.quit()

    return score

# Load the preprocess.json file
with open('src/data/preprocess.json', 'r') as file:
    data = json.load(file)

# Iterate over the games and calculate evaluation values after the opening
for entry in data:
    moves_san = entry['moves_san'].split()
    moves_5 = entry['moves_san'].split()[:10]
    moves_10 = entry['moves_san'].split()[:20]
    moves_except_last_5 = entry['moves_san'].split()[:-10]


    
    # Create a chess board and play the opening moves on it
    board = chess.Board()
    for move in moves_san:
        board.push_san(move)

    # Evaluate the position after the opening using Stockfish
    evaluation_final= evaluate_position(board)

    board = chess.Board()
    for move in moves_5:
        board.push_san(move)

    # Evaluate the position after the opening using Stockfish
    evaluation_5= evaluate_position(board)

    board = chess.Board()
    for move in moves_10:
        board.push_san(move)

    # Evaluate the position after the opening using Stockfish
    evaluation_10= evaluate_position(board)

    # Create a chess board and play the opening moves on it
    board = chess.Board()
    for move in moves_except_last_5:
        board.push_san(move)

    # Evaluate the position after the opening using Stockfish
    evaluation_last_5= evaluate_position(board)

    # Update the entry with the evaluation value after the opening
    entry['evaluation_5_moves'] = evaluation_5
    entry['evaluation_10_moves'] = evaluation_10
    entry['evaluation_last_5'] = evaluation_last_5
    entry['evaluation_final'] = evaluation_final
    print("success")
# Save the updated data to a new file feature_engineered.json
with open('src/data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)
