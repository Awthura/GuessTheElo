import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from JSON file
with open('src/data/train.json', 'r') as file:
    data = json.load(file)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Normalize numerical features
numerical_features = [
    "area_ratio", "area_ratio_rev", "relative_game_advantage",
    "Advantage swings", "White blunders", "Black blunders",
    "relative_area_white", "relative_area_black"
]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# List of features to eliminate
sequential_features = ["material imbalance", "white material count", "black material count", "eval_white_perspective"]
df = df.drop(columns=sequential_features)
eliminated_features = ["white_bin", "black_bin", "opening_name", "white_result", "black_result"]

# Exclude eliminated features from selected features
X = df.drop(columns=eliminated_features)

X = np.array(X, dtype=np.float32)
y_white = np.array(df["white_bin"])
y_black = np.array(df["black_bin"])

# Convert target variables to numerical labels
label_encoder_white = LabelEncoder()
label_encoder_black = LabelEncoder()
y_white_encoded = label_encoder_white.fit_transform(y_white)
y_black_encoded = label_encoder_black.fit_transform(y_black)

# Dataset split
X_train_w, X_test_w, y_white_train, y_white_test = train_test_split(X, y_white_encoded, test_size=0.25, random_state=42)
X_train_b, X_test_b, y_black_train, y_black_test = train_test_split(X, y_black_encoded, test_size=0.25, random_state=42)

# Find the best K value
k_values = range(1, 21)  # Range of K values to evaluate
best_accuracy = 0
best_k = None


for k in k_values:
    model_white = KNeighborsClassifier(n_neighbors=k)
    model_black = KNeighborsClassifier(n_neighbors=k)

    # Perform cross-validation
    cross_val_scores_white = cross_val_score(model_white, X_train_w, y_white_train, cv=5)
    cross_val_scores_black = cross_val_score(model_black, X_train_b, y_black_train, cv=5)

    # Calculate the mean accuracy from cross-validation scores
    mean_accuracy_white = np.mean(cross_val_scores_white)
    mean_accuracy_black = np.mean(cross_val_scores_black)

    # Update the best K if the current K has higher accuracy
    if mean_accuracy_white > best_accuracy:
        best_accuracy = mean_accuracy_white
        best_k_white = k

    if mean_accuracy_black > best_accuracy:
        best_accuracy = mean_accuracy_black
        best_k_black = k

print("Best K:", best_k_white)
print("Best K:", best_k_black)

# Model training with the best K
model_white = KNeighborsClassifier(n_neighbors=best_k_white)
model_black = KNeighborsClassifier(n_neighbors=best_k_black)

model_white.fit(X_train_w, y_white_train)
model_black.fit(X_train_b, y_black_train)


# Model evaluation
y_white_pred = model_white.predict(X_test_w)
y_black_pred = model_black.predict(X_test_b)

# Generate confusion matrices
white_cm = confusion_matrix(y_white_test, y_white_pred)
black_cm = confusion_matrix(y_black_test, y_black_pred)


# Calculate the accuracy for white predictions
white_acc = model_white.score(X_test_w, y_white_test)

# Calculate the accuracy for black predictions
black_acc = model_black.score(X_test_b, y_black_test)

# Print the accuracy values
print("White Accuracy:", white_acc)
print("Black Accuracy:", black_acc)

# Generate classification reports for each output separately
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

# Generate predicted probabilities for white predictions (KNN doesn't provide probabilities directly)
y_white_pred_proba = np.zeros_like(model_white.predict_proba(X_test_w))

# Calculate loss for white predictions
white_loss = log_loss(y_white_test, y_white_pred_proba)
print("White Loss:", white_loss)

# Generate predicted probabilities for black predictions (KNN doesn't provide probabilities directly)
y_black_pred_proba = np.zeros_like(model_black.predict_proba(X_test_b))

# Calculate loss for black predictions
black_loss = log_loss(y_black_test, y_black_pred_proba)
print("Black Loss:", black_loss)

# Randomly select an index from the test dataset
random_index = np.random.randint(0, len(X_test_w))
random_sample = X_test_w[random_index]

# Reshape the sample to match the input shape expected by the model
random_sample = random_sample.reshape(1, -1)

# Predict the outcome using the trained model
white_prediction = model_white.predict(random_sample)
black_prediction = model_black.predict(random_sample)

# Convert the predicted labels back to their original class values
white_predicted_label = label_encoder_white.inverse_transform(white_prediction)
black_predicted_label = label_encoder_black.inverse_transform(black_prediction)

print("Random Item Index:", random_index)
print("Random Item:", random_sample)

# Print the predicted outcome
print("Predicted White Outcome:", white_predicted_label)
print("Predicted Black Outcome:", black_predicted_label)

# Get the ground truth label for the random item
ground_truth_label_white = label_encoder_white.inverse_transform([y_white_test[random_index]])
ground_truth_label_black = label_encoder_black.inverse_transform([y_black_test[random_index]])

print("Ground Truth White Label:", ground_truth_label_white)
print("Ground Truth Black Label:", ground_truth_label_black)



# Plot the confusion matrices
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.heatmap(white_cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('White Bin Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(black_cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Black Bin Confusion Matrix')

plt.tight_layout()
plt.show()
plt.close()