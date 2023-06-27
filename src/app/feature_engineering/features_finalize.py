import json

# Read the JSON file
with open('src/data/feature_engineered.json','r') as file:
    data = json.load(file)

# Remove specific attributes (keys)
attributes_to_remove = ['white_username',
                        'black_username', 
                        'white_rating', 
                        'black_rating',  
                        'black_username', 
                        'time_control', 
                        'fen', 'pgn', 
                        'moves_with_time', 
                        'moves', 'moves_san', 
                        'evaluation_5_moves', 
                        'evaluation_10_moves', 
                        'evaluation_last_5',
                        'evaluation_final',
                        'evaluations']
for entry in data:
    for attr in attributes_to_remove:
        entry.pop(attr, None)

# Save the updated data to a new JSON file
with open('src/data/train.json', 'w') as file:
    json.dump(data, file, indent=1)