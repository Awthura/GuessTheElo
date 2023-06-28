import json
import matplotlib.pyplot as plt

# Read the JSON file
json_file_path = 'src/data/train.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Count the occurrences of each time class
time_class_counts = {}
for item in data:
    time_class = item['white_bin']
    time_class_counts[time_class] = time_class_counts.get(time_class, 0) + 1

# Extract the time classes and their corresponding counts
time_classes = list(time_class_counts.keys())
counts = list(time_class_counts.values())

# Define custom colors for the bars
bar_colors = ['orange', 'red', 'blue', 'yellow', 'green']

# Create a bar chart with custom colors
plt.bar(time_classes, counts, color=bar_colors)
plt.xlabel('Time Class')
plt.ylabel('Count')
plt.title('Distribution of type of Chess games on sample dataset')

# Add labels on top of each bar
for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()
plt.close()