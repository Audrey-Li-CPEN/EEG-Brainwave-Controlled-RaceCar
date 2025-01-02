# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from scipy.signal import butter, filtfilt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from tqdm import tqdm

# # Bandpass filter
# def butter_bandpass(lowcut, highcut, fs, order=4):
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     return filtfilt(b, a, data)

# # Dataset class
# class EEGDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.features[idx], self.labels[idx]

# # DeepCNN Model
# class DeepCNN1D(nn.Module):
#     def __init__(self, num_channels=14, num_classes=4):
#         super(DeepCNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=31, padding=15)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU()
#         self.resblock1 = self._residual_block(64)
#         self.resblock2 = self._residual_block(64)
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(32, num_classes)
#         )

#     def _residual_block(self, channels):
#         return nn.Sequential(
#             nn.Conv1d(channels, channels, kernel_size=15, padding=7),
#             nn.BatchNorm1d(channels),
#             nn.ReLU(),
#             nn.Conv1d(channels, channels, kernel_size=15, padding=7),
#             nn.BatchNorm1d(channels),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.global_pool(x).squeeze(-1)
#         x = self.fc(x)
#         return x

# def preprocess_data(file_path, fs=128, lowcut=8.0, highcut=50.0, window_size=50, overlap=45):
#     data = pd.read_csv(file_path)
#     print("Columns in CSV:", data.columns)

#     # Use the correct label column name
#     label_column_name = 'Tag'
#     data[label_column_name], uniques = pd.factorize(data[label_column_name])
#     print("Unique labels:", uniques)
#     labels = data[label_column_name]

#     # Identify EEG channels
#     eeg_channels = [col for col in data.columns if col.startswith('Channel')]

#     # Apply bandpass filter
#     for col in eeg_channels:
#         data[col] = butter_bandpass_filter(data[col].values, lowcut, highcut, fs)

#     # Sliding window
#     features, new_labels = [], []
#     step_size = window_size - overlap
#     for start in range(0, len(data) - window_size + 1, step_size):
#         end = start + window_size
#         window = data.iloc[start:end][eeg_channels].values.T
#         features.append(window)
#         new_labels.append(labels.iloc[start])

#     features = np.array(features)
#     new_labels = np.array(new_labels)

#     # Filter invalid labels
#     valid_indices = new_labels >= 0
#     features = features[valid_indices]
#     new_labels = new_labels[valid_indices]

#     # Normalize features
#     scaler = StandardScaler()
#     for i in range(features.shape[1]):
#         features[:, i, :] = scaler.fit_transform(features[:, i, :])

#     return features, new_labels

# # Training
# def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, save_path="model.pth"):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

#     best_val_acc = 0.0  # Track the best validation accuracy
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)
#         history['train_loss'].append(train_loss)

#         # Validation
#         model.eval()
#         val_loss, correct, total = 0.0, 0, 0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)

#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())

#         val_loss /= len(val_loader)
#         val_acc = correct / total
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)

#         print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

#         # Save the best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), save_path)
#             print(f"Model saved with validation accuracy: {val_acc:.4f}")

#     # Calculate F1 Score, Precision, Recall, and Accuracy
#     f1 = f1_score(all_labels, all_preds, average='weighted')
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     accuracy = accuracy_score(all_labels, all_preds)

#     print("\nFinal Metrics:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")

#     return history

# # Main Script
# if __name__ == "__main__":
#     file_path = r"D:\Train\processed_test_with_tags1.csv"
#     file_path1 = r"D:\Train\processed_test_with_tags2.csv"
#     file_path2 = r"D:\Train\processed_test_with_tags3.csv"
#     file_path3 = r"D:\Train\processed_test_with_tags4.csv"

#     file_path_array = [file_path, file_path1, file_path2, file_path3]
#     save_path = r"D:\Train\trained_model.pth"

#     # Preprocess data
#     features, labels = preprocess_data(file_path)

#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
#     train_dataset = EEGDataset(X_train, y_train)
#     val_dataset = EEGDataset(X_val, y_val)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DeepCNN1D(num_channels=14, num_classes=len(np.unique(labels))).to(device)

#     # Train the model and save it
#     train_model(model, train_loader, val_loader, device, save_path=save_path)

#     print(f"Model training completed. Model saved to {save_path}")



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# DeepCNN Model
class DeepCNN1D(nn.Module):
    def __init__(self, num_channels=14, num_classes=4):
        super(DeepCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=31, padding=15)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.resblock1 = self._residual_block(64)
        self.resblock2 = self._residual_block(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Main Script
def preprocess_data(file_path, label_mapping, fs=128, lowcut=8.0, highcut=50.0, window_size=50, overlap=45):
    data = pd.read_csv(file_path)
    print("Columns in CSV:", data.columns)

    # Map labels to integers using the provided label mapping
    label_column_name = 'Tag'
    data[label_column_name] = data[label_column_name].map(label_mapping)
    labels = data[label_column_name]

    # Identify EEG channels
    eeg_channels = [col for col in data.columns if col.startswith('Channel')]

    # Apply bandpass filter
    for col in eeg_channels:
        data[col] = butter_bandpass_filter(data[col].values, lowcut, highcut, fs)

    # Sliding window
    features, new_labels = [], []
    step_size = window_size - overlap
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end][eeg_channels].values.T
        features.append(window)
        new_labels.append(labels.iloc[start])

    features = np.array(features)
    new_labels = np.array(new_labels)

    # Filter invalid labels
    valid_indices = new_labels >= 0
    features = features[valid_indices]
    new_labels = new_labels[valid_indices]

    # Normalize features
    scaler = StandardScaler()
    for i in range(features.shape[1]):
        features[:, i, :] = scaler.fit_transform(features[:, i, :])

    return features, new_labels


def preprocess_multiple_files(file_paths, label_mapping, fs=128, lowcut=8.0, highcut=50.0, window_size=50, overlap=45):
    """
    Preprocess data from multiple CSV files and combine them into a single dataset.
    """
    all_features = []
    all_labels = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        features, labels = preprocess_data(file_path, label_mapping, fs, lowcut, highcut, window_size, overlap)
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate all features and labels
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    return combined_features, combined_labels

# Training
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0.0  # Track the best validation accuracy
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation accuracy: {val_acc:.4f}")

    # Calculate F1 Score, Precision, Recall, and Accuracy
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return history



if __name__ == "__main__":
    file_paths = [
        r"D:\Train\processed_test_with_tags1.csv",
        r"D:\Train\processed_test_with_tags2.csv",
        r"D:\Train\processed_test_with_tags3.csv",
        r"D:\Train\processed_test_with_tags4.csv"
    ]
    save_path = r"D:\Train\trained_model.pth"

    # Define a consistent label mapping for all files
    label_mapping = {'left': 0, 'rest': 1, 'right': 2, 'stop': 3}

    # Preprocess data from all files
    features, labels = preprocess_multiple_files(file_paths, label_mapping)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCNN1D(num_channels=14, num_classes=len(label_mapping)).to(device)

    # Train the model and save it
    train_model(model, train_loader, val_loader, device, save_path=save_path)

    print(f"Model training completed. Model saved to {save_path}")

