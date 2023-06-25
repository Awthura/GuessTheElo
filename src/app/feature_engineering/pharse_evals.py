import chess
import chess.pgn
import chess.engine
import json
import re

# Load the preprocess.json file
with open('src/data/fetiny2.json', 'r') as file:
    data = json.load(file)



for entry in data:
    evaluations = entry['evaluations']
    evaluation_white = []
    evaluation_black = []

    for i, evaluation in enumerate(evaluations):

        if evaluation is None:
            continue

        if i % 2 == 0:
            evaluation_white.append(round(evaluation * (-0.01),2))
            evaluation_black.append(round(evaluation * 0.01,2))
        else:
            evaluation_black.append(round(evaluation * (-0.01),2))
            evaluation_white.append(round(evaluation * 0.01,2))

    entry['eval_white_perspective'] = evaluation_white
    entry['eval_black_perspective'] = evaluation_black
    print("Success")

# Save the updated data to a new file feature_engineered.json
with open('src/data/fetiny2.json', 'w') as file:
    json.dump(data, file, indent=4)
