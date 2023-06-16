import json
import chess
import chess.pgn
import chess.engine
import io

# Path to Stockfish engine executable
stockfish_path = 'Stockfish/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe'

# Function to evaluate opening moves using Stockfish
def evaluate_opening_moves(board):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Evaluate the position using Stockfish
    result = engine.play(board, chess.engine.Limit(time=2.0))
    score = result.score().relative
    
    engine.quit()
    
    return score

# Load the preprocess.json file
with open('src/data/preprocess.json', 'r') as file:
    data = json.load(file)

# Iterate over the games and calculate opening accuracy scores
for entry in data:
    pgn_data = entry['pgn']
    game = chess.pgn.read_game(io.StringIO(pgn_data))
    
    # Create a separate board for each side
    white_board = chess.Board()
    black_board = chess.Board()
    
    # Play the opening moves on the respective boards
    for move in game.mainline_moves():
        white_board.push(move)
        black_board.push(move)

    # Evaluate the opening moves for white using Stockfish
    white_score = evaluate_opening_moves(white_board)
    
    # Reverse the board for evaluating black's opening moves
    black_board.turn = chess.BLACK
    black_score = evaluate_opening_moves(black_board)
    
    # Update the entry with opening accuracy scores
    entry['white_opening_accuracy'] = white_score.score()
    entry['black_opening_accuracy'] = black_score.score()

# Save the updated data to a new file feature_engineering.json
with open('src/data/feature_engineering.json', 'w') as file:
    json.dump(data, file, indent=4)
