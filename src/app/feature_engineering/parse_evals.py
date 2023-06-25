import chess
import chess.pgn
import chess.engine
import json
import re

# Load the preprocess.json file
with open('src/data/feature_engineered.json', 'r') as file:
    data = json.load(file)



for entry in data:
    evaluations = entry['evaluations']
    evaluation_white = []
    evaluation_black = []
    extracted_number = 0
    prev_ext = 0

    for i, evaluation in enumerate(evaluations):

        pattern = r"Cp\(([+-]?\d+)\)"
        match = re.search(pattern, evaluation)

        if match:
            extracted_number = int(match.group(1))

        pattern2 = r"Mate\(([+-]?\d+)\)"
        match2 = re.search(pattern2, evaluation)

        if match2:
            extracted_number = int(match2.group(1))
            if extracted_number > 0:
                extracted_number = 2000
            elif extracted_number < 0:
                extracted_number = -2000
            elif extracted_number == 0:
                extracted_number = prev_ext*(-1)
            prev_ext = extracted_number

        if evaluation is None:
            continue

        if i % 2 == 0:
            evaluation_white.append(round(extracted_number * (-0.01),2))
            evaluation_black.append(round(extracted_number * 0.01,2))
        else:
            evaluation_black.append(round(extracted_number * (-0.01),2))
            evaluation_white.append(round(extracted_number * 0.01,2))

    entry['eval_white_perspective'] = evaluation_white
    print("Success")

# Save the updated data to a new file feature_engineered.json
with open('src/data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)
