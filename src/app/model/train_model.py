import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import json

# from tensorflow.compat.v1.keras.models import Sequential
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Dropout
# from tensorflow.compat.v1.keras.utils import to_categorical

# Load data from JSON file
with open('src/data/train.json', 'r') as file:
    data = json.load(file)

# Convert categorical features to numerical representations (one-hot encoding)
categorical_features = ["white_result", "black_result", "opening_name", "white_bin", "black_bin"]
for feature in categorical_features:
    unique_values = np.unique([item[feature] for item in data])
    for value in unique_values:
        new_feature_name = f"{feature}_{value}"
        for item in data:
            item[new_feature_name] = int(item[feature] == value)
    del_keys = [feature]
    del_keys.extend([f"{feature}_{value}" for value in unique_values if value != ""])
    for item in data:
        for key in del_keys:
            del item[key]

# Normalize numerical features
numerical_features = [
    "area_ratio", "area_ratio_rev", "relative_game_advantage",
    "Advantage swings", "White blunders", "Black blunders",
    "relative_area_white", "relative_area_black"
]
scaler = MinMaxScaler()
numerical_data = np.array([item.get(feature) for item in data for feature in numerical_features])
scaled_data = scaler.fit_transform(numerical_data)

for i, item in enumerate(data):
    item[numerical_features] = scaled_data[i]

# Prepare sequential features
sequential_features = ["material imbalance", "white material count", "black material count", "eval_white_perspective"]
max_seq_length = max(len(item[feature]) for item in data for feature in sequential_features)
for item in data:
    for feature in sequential_features:
        sequence = np.array(item[feature])
        padding_length = max_seq_length - len(sequence)
        padded_sequence = np.pad(sequence, (0, padding_length), mode='constant')
        item[feature] = padded_sequence

# Feature extraction
selected_features = [
    "material imbalance", "white material count", "black material count",
    "eval_white_perspective", "area_ratio", "area_ratio_rev",
    "relative_game_advantage", "Advantage swings", "White blunders",
    "Black blunders", "relative_area_white", "relative_area_black"
]

X = np.array([item[selected_features] for item in data])
y_white = np.array([item["white_bin"] for item in data])
y_black = np.array([item["black_bin"] for item in data])

# Convert target variables to categorical
num_classes = len(np.unique(y_white))
y_white = to_categorical(y_white, num_classes=num_classes)
y_black = to_categorical(y_black, num_classes=num_classes)

# Dataset split
X_train, X_test, y_white_train, y_white_test, y_black_train, y_black_test = train_test_split(X, y_white, y_black, test_size=0.2, random_state=42)

# Model training
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, [y_white_train, y_black_train], epochs=10, batch_size=32)

# Model evaluation
loss, white_loss, black_loss, white_accuracy, black_accuracy = model.evaluate(X_test, [y_white_test, y_black_test])
print("Test Loss:", loss)
print("White Bin Loss:", white_loss)
print("Black Bin Loss:", black_loss)
print("White Bin Accuracy:", white_accuracy)
print("Black Bin Accuracy:", black_accuracy)
