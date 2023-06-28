import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

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

# List of features to eliminate
sequential_features = ["material imbalance", "white material count", "black material count", "eval_white_perspective"]
eliminated_features = ["white_bin", "black_bin", "material imbalance", "white material count", "black material count", "eval_white_perspective" ]

# Exclude eliminated features from selected features
X = df.drop(columns=eliminated_features)
print(X.columns)

X = np.array(X, dtype=np.float32)
print(X[0])
y_white = np.array(df["white_bin"])
y_black = np.array(df["black_bin"])

# Convert target variables to numerical labels
label_encoder_white = LabelEncoder()
label_encoder_black = LabelEncoder()
y_white_encoded = label_encoder_white.fit_transform(y_white)
y_black_encoded = label_encoder_black.fit_transform(y_black)

# Model training
#model_white = DecisionTreeClassifier()
#model_black = DecisionTreeClassifier()

model_white = RandomForestClassifier()
model_black = RandomForestClassifier()

model_white.fit(X, y_white_encoded)
model_black.fit(X, y_black_encoded)

# Dataset split
X_train, X_test, y_white_train, y_white_test = train_test_split(X, y_white_encoded, test_size=0.25, random_state=42)
_, _, y_black_train, y_black_test = train_test_split(X, y_black_encoded, test_size=0.25, random_state=42)

print(len(X_train))
print(len(X_test))

# Model evaluation
y_white_pred = model_white.predict(X_test)
y_black_pred = model_black.predict(X_test)

# Generate confusion matrices
white_cm = confusion_matrix(y_white_test, y_white_pred)
black_cm = confusion_matrix(y_black_test, y_black_pred)

# Model evaluation

# Calculate the loss for white predictions
white_acc = model_white.score(X_test, y_white_test)

# Calculate the loss for black predictions
black_acc = model_black.score(X_test, y_black_test)

# Print the loss values
print("White Acc:", white_acc)
print("Black Acc:", black_acc)

# Generate classification report
white_report = classification_report(y_white_test, y_white_pred)
black_report = classification_report(y_black_test, y_black_pred)

print("White Bin Classification Report:")
print(white_report)

print("\nBlack Bin Classification Report:")
print(black_report)

print("White Bin Confusion Matrix:")
print(white_cm)

print("\nBlack Bin Confusion Matrix:")
print(black_cm)

# Generate predicted probabilities for white predictions
y_white_pred_proba = model_white.predict_proba(X_test)

# Calculate loss for white predictions
white_loss = log_loss(y_white_test, y_white_pred_proba)
print("White Loss:", white_loss)

# Generate predicted probabilities for black predictions
y_black_pred_proba = model_black.predict_proba(X_test)

# Calculate loss for black predictions
black_loss = log_loss(y_black_test, y_black_pred_proba)
print("Black Loss:", black_loss)

# Randomly select an index from the test dataset
random_index = np.random.randint(0, len(X_test))
random_sample = X_test[random_index]

# Reshape the sample to match the input shape expected by the model
random_sample = random_sample.reshape(1, -1)

# Predict the outcome using the trained model
white_prediction = model_white.predict(random_sample)
black_prediction = model_black.predict(random_sample)

# Convert the predicted labels back to their original class values
white_predicted_label = label_encoder_white.inverse_transform(white_prediction)
black_predicted_label = label_encoder_black.inverse_transform(black_prediction)

# Generate predicted probabilities for white predictions
white_pred_proba = model_white.predict_proba(random_sample)

# Generate predicted probabilities for black predictions
black_pred_proba = model_black.predict_proba(random_sample)

print("Random Item Index:", random_index)
print("Random Item:", random_sample)

print("White Predicted Probabilities:")
print(white_pred_proba)

print("Black Predicted Probabilities:")
print(black_pred_proba)


# Print the random sample and the predicted outcome
print("Predicted White Outcome:", white_predicted_label)
print("Predicted Black Outcome:", black_predicted_label)

# Get the ground truth label for the random item
ground_truth_label_white = label_encoder_white.inverse_transform([y_white_test[random_index]])
ground_truth_label_black = label_encoder_white.inverse_transform([y_black_test[random_index]])

print(ground_truth_label_black)
print(ground_truth_label_white)