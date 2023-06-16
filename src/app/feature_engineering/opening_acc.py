import chess
import chess.pgn
import chess.engine
import json
import re 

# Path to Stockfish engine executable
stockfish_path = 'Stockfish/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe'

# Function to evaluate opening moves using Stockfish
def evaluate_opening_moves(moves):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board()

    # Play the opening moves on the board
    for move in moves:
        board.push_san(move)

    # Evaluate the position using Stockfish
    analysis = engine.analysis(board, chess.engine.Limit(time=2.0))
    score = analysis.best()["score"].relative.score()

    engine.quit()

    return score

# Load the preprocess.json file
with open('src/data/preprocess.json', 'r') as file:
    data = json.load(file)

# Iterate over the games and calculate opening accuracy scores
for entry in data:
    moves = entry['moves'].split()
    
    # Remove move numbers (e.g., '1.', '2.', etc.) and game result notation
    moves = [re.sub(r'^\d+\.', '', move) for move in moves]
    moves = [re.sub(r'[\+\-]{1,2}$', '', move) for move in moves]

    # Evaluate the opening moves using Stockfish
    opening_accuracy = evaluate_opening_moves(moves)

    # Update the entry with the opening accuracy score
    entry['opening_accuracy'] = opening_accuracy

# Save the updated data to a new file feature_engineering.json
with open('src/data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)