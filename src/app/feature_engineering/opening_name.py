import re
import json

# Load the JSON data
with open('src/data/fetiny2.json') as json_file:
    data = json.load(json_file)

# Openings dictionary
openings_dict = {
    "1.e4 e5": "Open Game",
    "1.e4 c5": "Sicilian Defense",
    "1.e4 e6": "French Defense",
    "1.e4 c6": "Caro-Kann Defense",
    "1.e4 d6": "Pirc Defense",
    "1.e4 g6": "Modern Defense",
    "1.e4 Nf6": "Alekhine's Defense",
    "1.d4 d5": "Queen's Pawn Game",
    "1.d4 Nf6": "Indian Game",
    "1.d4 Nc6": "Nimzowitsch Defense",
    # Add more openings here
}

# Iterate over the data
for entry in data:
    moves_san = entry['moves_san']

    match = re.search(r'(\w+\s+\w+)', moves_san)
    if match:
        first_two_moves = match.group(1)
        if opening in openings_dict:
            opening_name = openings_dict[opening]
        else:
            opening_name = "Unknown Opening"
    else:
        opening_name = "Unknown Opening"

    print(first_two_moves)


# Save the updated data to the JSON file
# with open('src/data/fetiny2.json', 'w') as json_file:
#     json.dump(data, json_file, indent=4)
