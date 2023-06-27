import json

blunder_tolerence = 3 
advantage_swing_tolerence = 1
swing_threshold = 0.2
# Load the JSON data
with open('src/data/feature_engineered.json') as json_file:
    data = json.load(json_file)

for entry in data:

    eval = entry['eval_white_perspective']
    advantage_swing = 0
    white_blunders = 0
    black_blunders = 0
    for i in range(len(eval) - 1):
        current_evaluation = eval[i]
        next_evaluation = eval[i + 1]
        if current_evaluation >= swing_threshold and next_evaluation <= -swing_threshold and current_evaluation - next_evaluation >= advantage_swing_tolerence:
            advantage_swing += 1
        if current_evaluation <= -swing_threshold and next_evaluation >= swing_threshold and next_evaluation - current_evaluation >= advantage_swing_tolerence:
            advantage_swing += 1
        if current_evaluation - next_evaluation >= blunder_tolerence:
            white_blunders += 1
        if next_evaluation - current_evaluation >= blunder_tolerence:
            black_blunders += 1
    entry["Advantage swings"] = advantage_swing
    entry["White blunders"] = white_blunders
    entry["Black blunders"] = black_blunders

    print("Success")

# Save the updated data to a new file feature_engineered.json
with open('src/data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)
