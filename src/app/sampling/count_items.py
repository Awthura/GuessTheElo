import json

# Read the JSON file
json_file_path = 'src/data/sample.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Count the number of items
num_items = len(data)

print(f"Number of items: {num_items}")