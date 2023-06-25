import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('src/data/feature_engineered.json') as json_file:
    data = json.load(json_file)

# Prompt for the item number
item_number = int(input("Enter the game number: "))

# Check if the item numbers are valid
if item_number < 1 or item_number > len(data):
    print("Invalid item number!")
else:
    item_data = data[item_number - 1]
    # Get the data for White
    white_count_str = item_data['white material count']
    white_count = [int(value) for value in white_count_str.split(',')]
    x_values = [i * 0.5 for i in range(len(white_count))]
    y_values = white_count

    # Get the data for Black
    black_count_str = item_data['black material count']
    black_count = [int(value) for value in black_count_str.split(',')]
    x_values2 = [i * 0.5 for i in range(len(black_count))]
    y_values2 = black_count

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, markersize=5, color='red', label="White")
    plt.plot(x_values2, y_values2, marker='o', linestyle='-', linewidth=2, markersize=5, color='blue', label="Black")
    plt.xlabel("Moves")
    plt.ylabel("Piece Value Count")
    plt.title(f"Piece Value Count of White Vs Black for game {item_number}")
    plt.grid(True)
    plt.yticks(range(min(min(y_values), min(y_values2)), max(max(y_values), max(y_values2))+1))
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
