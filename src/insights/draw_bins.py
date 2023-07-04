import json
import matplotlib.pyplot as plt

#Read the JSON file
json_file_path = 'src/data/preprocess.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

x_display = ['0-399', '400-599', '600-799', '800-999', '1000-1199', '1200-1399', '1400-1599', '1600-1799', '1800-1999', '2000-2199','2200-inf']

#Count the occurrences of each time class
time_class_counts = {}
for item in data:
    time_class = item['white_bin']
    time_class_counts[time_class] = time_class_counts.get(time_class, 0) + 1

#Extract the time classes and their corresponding counts
time_classes = list(time_class_counts.keys())
counts = list(time_class_counts.values())

#Define custom colors for the bars
bar_colors = ['orange', 'red', 'blue', 'yellow', 'green']

#Create a bar chart with custom colors and sorted x-axis labels
plt.bar([x_display.index(bin) for bin in time_classes], counts, color=bar_colors)
plt.xlabel('Bins')
plt.ylabel('Count')
plt.title('Distribution of bins')

#Set the x-axis tick labels to the custom labels in y_display
plt.xticks(range(len(time_classes)), x_display, rotation='horizontal')

# #Add labels on top of each bar
# for i, count in enumerate(counts):
#     plt.text(i, count, str(count), ha='center', va='bottom')

plt.show()
plt.close()