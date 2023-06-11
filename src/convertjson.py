import csv
import json

def convert_csv_to_json(csv_file_path, json_file_path, encoding='utf-8'):
    # Read CSV file with specified encoding
    with open(csv_file_path, 'r', encoding=encoding) as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)

    # Convert CSV data to JSON
    json_data = json.dumps(data, indent=4)

    # Write JSON data to file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

# Specify the paths of the CSV and JSON files
csv_file_path = 'src/data/club_games_data.csv'
json_file_path = 'src/data/dataset.json'

# Call the function to convert CSV to JSON
convert_csv_to_json(csv_file_path, json_file_path)

