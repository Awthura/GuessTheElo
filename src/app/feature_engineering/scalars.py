import json
import numpy as np
import math

# Load the JSON data
with open('src/data/feature_engineered.json') as json_file:
    data = json.load(json_file)

def areaRatioHandle(a1,a2):
    try: 
        ratio = (a1/a2)
        if ratio == math.inf:
            ratio = 100000
    except ZeroDivisionError:
        ratio = 100000
    ratio = round(abs(ratio),3)
    return ratio

for entry in data:

    eval = entry['eval_white_perspective']
    moves = entry['moves_san']

    # Create lists for x and y values
    x_values = [i * 0.5 for i in range(len(eval))]
    y_values = eval


    area_above_zero = np.trapz(np.maximum(y_values, 0), x_values)
    area_below_zero = np.trapz(np.minimum(y_values, 0), x_values)

    relative_area_above_zero = area_above_zero/len(eval)*0.5
    relative_area_below_zero = area_below_zero/len(eval)*0.5

    relative_game_advantage = relative_area_above_zero + relative_area_below_zero


    entry['area_ratio'] = areaRatioHandle(area_above_zero,area_below_zero)
    entry['area_ratio_rev'] = areaRatioHandle(area_below_zero,area_above_zero)
    entry['relative_game_advantage'] = round(relative_game_advantage,3)

    print("Success")

# Save the updated data to a new file feature_engineered.json
with open('src/data/feature_engineered.json', 'w') as file:
    json.dump(data, file, indent=4)