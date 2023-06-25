import re
import json
from typing import KeysView

# Load the JSON data
with open('src/data/feature_engineered.json') as json_file:
    data = json.load(json_file)

# Openings dictionary
openings_dict = {
    "e4  e5": "Double King's Pawn Opening",
    "e4  c5": "Sicilian Defense",
    "e4  e6": "French Defense",
    "e4  c6": "Caro-Kann Defense",
    "e4  d6": "Pirc Defense",
    "e4  g6": "Modern Defense",
    "e4  Nf6": "Alekhine's Defense",
    "e4  d5": "Scandinavian Defense",
    "d4  d5": "Double Queen's Pawn Opening",
    "d4  Nf6": "Indian Game",
    "d4  d6": "Philidor Defense",
    "d4  f5": "Dutch Defense",
    "d4  g6": "Modern Defense",
    "d4  c5": "Benoni Defense",
    "Nf3  Nf6": "Reti Opening",
    "Nf3  e6": "Nimzo-Indian Defense",
    "Nf3  c5": "English Opening",
    "c4  c5": "English Opening",
    "c4  b6": "English Opening",
    "c4  Nf6": "English Opening",
    "g3  e5": "King's Fianchetto Opening",
    "g3  d5": "King's Fianchetto Opening",
    "g3  Nf6": "King's Fianchetto Opening",
    "Nc3  Nf6": "Dunst Opening",
    "Nc3  d5": "Dunst Opening",
    "Nc3  e5": "Dunst Opening",
    "b3  e5": "Nimzo-Larsen Attack",
    "b3  d5": "Nimzo-Larsen Attack",
    "b3  Nf6": "Nimzo-Larsen Attack",
    "c3  e5": "Saragossa Opening",
    "c3  d5": "Saragossa Opening",
    "c3  Nf6": "Saragossa Opening",
    "g4  e5": "Grob's Attack",
    "g4  d5": "Grob's Attack",
    "g4  Nf6": "Grob's Attack",
    "f4  e5": "Bird's Opening",
    "f4  d5": "Bird's Opening",
    "f4  Nf6": "Bird's Opening",
    "f4  c5": "Bird's Opening",
    # Add more opening moves and names as desired
}

# Iterate over the data
for entry in data:
    moves_san = entry['moves_san']

    match = re.search(r'(\w+\s+\w+)', moves_san)
    if match:
        first_two_moves = match.group(1)

        if first_two_moves in openings_dict.keys():

            opening_name = openings_dict[first_two_moves]
        else:
            opening_name = "Unknown Opening"
    else:
        opening_name = "Unknown Opening no match"

    entry['opening_name']= opening_name

#Save the updated data to the JSON file
with open('src/data/feature_engineered.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
