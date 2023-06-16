import json

# Define the rating bins
rating_bins = [
    (0, 399),
    (400, 599),
    (600, 799),
    (800, 999),
    (1000, 1199),
    (1200, 1399),
    (1400, 1599),
    (1600, 1799),
    (1800, 1999),
    (2000, 2199),
    (2200, float('inf'))
]

# Load the JSON data from the file
json_file_path = 'src/data/preprocess.json'
with open(json_file_path) as file:
    data = json.load(file)

# Process the ratings and add the bins as attributes
processed_data = []
for game in data:
    white_rating = int(game['white_rating'])
    black_rating = int(game['black_rating'])
    white_bin = None
    black_bin = None
    for bin_start, bin_end in rating_bins:
        if bin_start <= white_rating <= bin_end:
            white_bin = f'{bin_start}-{bin_end}'
        if bin_start <= black_rating <= bin_end:
            black_bin = f'{bin_start}-{bin_end}'
    game['white_bin'] = white_bin
    game['black_bin'] = black_bin
    processed_data.append(game)

# Save the processed data to a new JSON file
preprocess_file_path = 'src/data/preprocess.json'
with open(preprocess_file_path, 'w') as file:
    json.dump(processed_data, file, indent=4)

print(f'Preprocessed data saved to {preprocess_file_path}.')