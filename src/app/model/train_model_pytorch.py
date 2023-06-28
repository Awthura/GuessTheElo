import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import nn, optim
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
            feature_tensor = torch.tensor(row[feature])
        else:
            feature_tensor = torch.tensor([row[feature]])
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
y_white = torch.tensor(y_white_encoded)
y_white = torch.eye(num_classes)[y_white].float()
y_black = torch.tensor(y_black_encoded)
y_black = torch.eye(num_classes)[y_black].float()

# Dataset split
X_train, X_test, y_white_train, y_white_test, y_black_train, y_black_test = train_test_split(X, y_white, y_black, test_size=0.2, random_state=42)

# Pad sequences
X_train = torch.nn.functional.pad(X_train, batch_first=True)

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_white_train)
test_dataset = TensorDataset(*X_test, y_white_test)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)
        return output

input_size = X_train.shape[2]
hidden_size = 64
output_size = num_classes

model = LSTMModel(input_size, hidden_size, output_size)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
total_loss = 0.0

for inputs, targets in test_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    total_loss += loss.item() * inputs.size(0)

avg_loss = total_loss / len(test_loader.dataset)
print(f"Test Loss: {avg_loss:.4f}")
