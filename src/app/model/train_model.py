import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import json
import keras
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

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
#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Prepare sequential features
sequential_features = ["material imbalance", "eval_white_perspective"]
max_seq_length = df[sequential_features].applymap(len).max().max()
for feature in sequential_features:
    df[feature] = df[feature].apply(lambda seq: seq + [0] * (max_seq_length - len(seq)))



# Convert sequence features to NumPy arrays and pad sequences
for feature in sequential_features:
    df[feature] = df[feature].apply(lambda seq: np.asarray(seq, dtype=np.float32))
    df[feature] = tf.keras.preprocessing.sequence.pad_sequences(df[feature], padding='post')

print(df.columns)

# Feature extraction
selected_features = [
    'material imbalance', 'eval_white_perspective',
    'relative_game_advantage', 'Advantage swings', 'White blunders',
    'Black blunders', 'white_result_checkmated',
    'white_result_resigned', 'white_result_timeout', 
    'white_result_win', 'black_result_checkmated',
    'black_result_resigned','black_result_timeout',
    'black_result_win', 'area_ratio'
]

X = np.array(df[selected_features])

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

X_tensor = tf.ragged.constant(X_train)

# Define your existing LSTM model
model = Sequential()

learning_rate = 0.01  # Specify your desired learning rate
optimizer = Adam(learning_rate=learning_rate)

# Define class weights for cost-sensitive learning
class_weights = {0: 2.0, 1: 1.0, 2: 1.8}  # Assign higher weight to misclassifications of white class

# Define your existing LSTM model
model = Sequential()
model.add(LSTM(units=32, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=32, activation='tanh'))

model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#history = model.fit(X_tensor, y_white_train, epochs=20, batch_size=32)
history = model.fit(X_tensor, y_white_train, epochs=10, batch_size=32, class_weight=class_weights)

# # Perform cross-validation
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(model, X, y_white, cv=kfold)

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Model evaluation
train_loss, train_accuracy = model.evaluate(X_train, y_white_train,  verbose=0)
loss, accuracy = model.evaluate(X_test, y_white_test, verbose=0)

print("Train Loss:", train_loss)
print("Train Accuracy", train_accuracy)

print("Test Loss:", loss)
print("Accuracy", accuracy)

# Generate predictions for the test set
y_pred_white = model.predict(X_test)
y_pred_white_train = model.predict(X_train)
#y_pred_white_labels = np.argmax(y_pred_white, axis=1)

# y_pred_white_labels = np.array(y_pred_white_labels)  # Convert the list to a NumPy array

# Calculate confusion matrix
confusion_matrix_white = confusion_matrix(y_white_test.argmax(axis=1), y_pred_white.argmax(axis=1))
confusion_matrix_white_train = confusion_matrix(y_white_train.argmax(axis=1), y_pred_white_train.argmax(axis=1))

print("Confusion Matrix Test - White Bin:")
print(confusion_matrix_white)

print("Confusion Matrix Train - White Bin:")
print(confusion_matrix_white_train)
# print("\nConfusion Matrix - Black Bin:")
# print(confusion_matrix_black)

test_report = classification_report(y_white_test.argmax(axis=1), y_pred_white.argmax(axis=1))
print("Test set Classification Report:")
print(test_report)

train_report = classification_report(y_white_train.argmax(axis=1), y_pred_white_train.argmax(axis=1))
print("Train set Classification Report:")
print(train_report)

# Plot the confusion matrices
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix_white, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Test Set Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix_white_train, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Training Set Confusion Matrix')

# # Plot the loss graph
# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()

plt.tight_layout()
plt.show()
plt.close()