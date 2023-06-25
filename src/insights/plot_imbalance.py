import json
import matplotlib.pyplot as plt

# Load the JSON data
with open('src/data/feature_engineered.json') as json_file:
    data = json.load(json_file)

# Prompt for the item number
item_number = int(input("Enter the item number: "))

# Check if the item number is valid
if item_number < 1 or item_number > len(data):
    print("Invalid item number!")
else:
    # Get the nth item from the data
    item_data = data[item_number - 1]
    material_imbalance_str = item_data['material imbalance']

    # Split the string into a list of values
    material_imbalance = [int(value) for value in material_imbalance_str.split(',')]

    # Create lists for x and y values
    x_values = [i * 0.5 for i in range(len(material_imbalance))]
    y_values = material_imbalance

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, markersize=4, color='red')
    plt.xlabel("Moves")
    plt.ylabel("Material Imbalance")
    plt.title(f"Material Imbalance of Game {item_number}")
    plt.grid(True)
    plt.yticks(range(min(y_values), max(y_values)+1))
    plt.tight_layout()

    # Add margin line at y=0
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Fill region below y=0 with grey color
    plt.fill_between(x_values, min(y_values),y2=0,  where=None, color='lightgrey')

    plt.show()
    plt.close()
