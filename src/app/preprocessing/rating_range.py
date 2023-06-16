import json

# Load the JSON data from the file
json_file_path = 'src/data/sample.json'
with open(json_file_path) as file:
    data = json.load(file)

# Extract the ratings from the JSON data
ratings = [int(game['white_rating']) for game in data] + [int(game['black_rating']) for game in data]

# Determine the range of ratings
min_rating = min(ratings)
max_rating = max(ratings)

# Print the range of ratings
print(f"Range of ratings: {min_rating} - {max_rating}")