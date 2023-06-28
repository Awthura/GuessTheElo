import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import json

# Load data from JSON file
with open('src/data/train.json', 'r') as file:
    data = json.load(file)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical features to numerical representations (one-hot encoding)
categorical_features = ["white_result", "black_result", "opening_name"]
for feature in categorical_features:
    df_encoded = pd.get_dummies(df[feature], prefix=feature)
    df = pd.concat([df, df_encoded], axis=1)
    df.drop([feature], axis=1, inplace=True)

# Normalize numerical features
numerical_features = [
    "area_ratio", "area_ratio_rev", "relative_game_advantage",
    "Advantage swings", "White blunders", "Black blunders",
    "relative_area_white", "relative_area_black"
]
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Prepare sequential features
sequential_features = ["material imbalance", "white material count", "black material count", "eval_white_perspective"]
max_seq_length = df[sequential_features].applymap(len).max().max()
for feature in sequential_features:
    df[feature] = df[feature].apply(lambda seq: seq + [0] * (max_seq_length - len(seq)))

# Feature extraction
selected_features = [
    "material imbalance", "white material count", "black material count",
    "eval_white_perspective", "area_ratio", "area_ratio_rev",
    "relative_game_advantage", "Advantage swings", "White blunders",
    "Black blunders", "relative_area_white", "relative_area_black"
]

# Convert sequence features to NumPy arrays
for feature in sequential_features:
    df[feature] = df[feature].apply(lambda seq: np.asarray(seq, dtype=np.float32))

print(df.head())
X = np.array(df[selected_features], dtype=np.float32)
print(X)

y_white = np.array(df["white_bin"])
y_black = np.array(df["black_bin"])

# Convert target variables to numerical labels
label_encoder_white = LabelEncoder()
label_encoder_black = LabelEncoder()
y_white_encoded = label_encoder_white.fit_transform(y_white)
y_black_encoded = label_encoder_black.fit_transform(y_black)

# Convert numerical labels to categorical
num_classes = len(np.unique(y_white))
y_white = to_categorical(y_white_encoded, num_classes=num_classes)
y_black = to_categorical(y_black_encoded, num_classes=num_classes)

# Dataset split
X_train, X_test, y_white_train, y_white_test, y_black_train, y_black_test = train_test_split(X, y_white, y_black, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# Assuming X_train is a list of lists
X_train = [np.asarray(item) for item in X_train]
X_tensor = tf.convert_to_tensor(np.stack(X_train, axis=0))

print(X_train.shape)
print(X_train.dtype)
print(X_train[0])
# Model training
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_tensor, y_white_train, epochs=10, batch_size=32)


# Model evaluation
loss, white_loss, black_loss, white_accuracy, black_accuracy = model.evaluate(X_test, [y_white_test, y_black_test])
print("Test Loss:", loss)
print("White Bin Loss:", white_loss)
print("Black Bin Loss:", black_loss)
print("White Bin Accuracy:", white_accuracy)
print("Black Bin Accuracy:", black_accuracy)
