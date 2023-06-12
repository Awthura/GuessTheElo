import json

# Read the JSON file
json_file_path = 'src/data/sample.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Filter the data based on attributes
filtered_data = [item for item in data if item['rules'] == 'chess' and item['rated'] == 'True']

# Update the JSON file with the filtered results
with open(json_file_path, 'w') as file:
    json.dump(filtered_data, file, indent=4)

print("Sample JSON file updated with filtered results.")