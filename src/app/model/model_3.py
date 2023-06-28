import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from keras import layers
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

X = []
for _, row in df.iterrows():
    feature_tensors = []
    for feature in selected_features:
        if isinstance(row[feature], (list, tuple)):
            feature_tensor = tf.constant(row[feature])
        else:
            feature_tensor = tf.constant([row[feature]])
        feature_tensors.append(feature_tensor)
    X.append(feature_tensors)

print(X[0])

y_white = np.array(df["white_bin"])
y_black = np.array(df["black_bin"])

# Convert target variables to numerical labels
label_encoder_white = LabelEncoder()
label_encoder_black = LabelEncoder()
y_white_encoded = label_encoder_white.fit_transform(y_white)
y_black_encoded = label_encoder_black.fit_transform(y_black)

# Convert numerical labels to categorical
num_classes = len(np.unique(y_white))
y_white = keras.utils.to_categorical(y_white_encoded, num_classes)
y_black = keras.utils.to_categorical(y_black_encoded, num_classes)

# Dataset split
X_train, X_test, y_white_train, y_white_test, y_black_train, y_black_test = train_test_split(X, y_white, y_black, test_size=0.2, random_state=42)

# Pad sequences
X_train_padded = []
for tensors in X_train:
    padded_tensors = pad_sequences(tensors, padding="post", maxlen=max_seq_length, dtype=np.float32)
    X_train_padded.append(padded_tensors)

X_train_padded = np.stack(X_train_padded)

X_test_padded = []
for tensors in X_test:
    padded_tensors = pad_sequences(tensors, padding="post", maxlen=max_seq_length, dtype=np.float32)
    X_test_padded.append(padded_tensors)

X_test_padded = np.stack(X_test_padded)


# Create TensorDatasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_white_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_white_test))

# Create DataLoaders
batch_size = 32
train_loader = train_dataset.batch(batch_size).shuffle(len(train_dataset))
test_loader = test_dataset.batch(batch_size)

# Model definition
input_size = X_train_padded.shape[2]
hidden_size = 64
output_size = num_classes

model = keras.Sequential([
    layers.LSTM(hidden_size, return_sequences=False, input_shape=(max_seq_length, input_size)),
    layers.Dense(output_size, activation='softmax')
])

# Training
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    for inputs, targets in train_loader:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = loss_fn(targets, outputs)
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        total_loss += loss_value.numpy()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
total_loss = 0.0
num_batches = 0

for inputs, targets in test_loader:
    outputs = model(inputs)
    loss_value = loss_fn(targets, outputs)

    total_loss += loss_value.numpy()
    num_batches += 1

avg_loss = total_loss / num_batches
print(f"Test Loss: {avg_loss:.4f}")
