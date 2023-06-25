import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('src/data/fetiny2.json') as json_file:
    data = json.load(json_file)

# Prompt for the item number
item_number = int(input("Enter the item number: "))

# Check if the item number is valid
if item_number < 1 or item_number > len(data):
    print("Invalid item number!")
else:
    # Get the nth item from the data
    item_data = data[item_number - 1]
    eval = item_data['eval_white_perspective']
    moves = item_data['moves_san']

    # Create lists for x and y values
    x_values = [i * 0.5 for i in range(len(eval))]
    y_values = eval

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, markersize=4, color='green')
    plt.xlabel("Moves")
    plt.ylabel("Evaluation")
    plt.title(f"Evaluation of Game {item_number}")
    plt.grid(True)
    plt.yticks(range(round(min(y_values)-1), round(max(y_values)+1)))
    plt.tight_layout()

    # Add margin line at y=0
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Calculate area under the curve above zero and below zero
    area_above_zero = np.trapz(np.maximum(y_values, 0), x_values)
    area_below_zero = np.trapz(np.minimum(y_values, 0), x_values)
    relative_area_above_zero = area_above_zero/len(eval)*0.5
    relative_area_below_zero = area_below_zero/len(eval)*0.5

    print(f"Area above zero: {area_above_zero:.2f}")
    print(f"Area below zero: {area_below_zero:.2f}")
    print(f"Relative Area above zero: {relative_area_above_zero:.2f}")
    print(f"Relative Area below zero: {relative_area_below_zero:.2f}")

    # Fill region below y=0 with grey color
    plt.fill_between(x_values, min(y_values), y2=0, where=None, color='lightgrey')

    plt.show()
    plt.close()
