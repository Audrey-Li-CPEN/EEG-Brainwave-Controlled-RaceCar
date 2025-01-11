import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from pyquaternion import Quaternion
from PyEMD import EMD
from tqdm import tqdm
from sklearn.metrics import f1_score
import joblib

# Quaternion Feature Extraction
def quaternion_features(signal):
    variance = np.var(signal)
    homogeneity = np.mean(np.abs(np.diff(signal)))
    contrast = np.max(signal) - np.min(signal)
    return [variance, homogeneity, contrast]

# EMD Feature Extraction
def emd_features(signal, num_imfs=3):
    emd = EMD()
    imfs = emd(signal)
    return [np.mean(imf) for imf in imfs[:num_imfs]] + [0.0] * (num_imfs - len(imfs))

# Dataset Class
class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# CNN-LSTM Model
class CNNLSTM(torch.nn.Module):
    def __init__(self, num_channels=14, num_classes=4):
        super(CNNLSTM, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(num_channels, 64, kernel_size=31, padding=15),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=15, padding=7),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

# Training Function with Scheduler and Metrics
def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3, weight_decay=1e-4, save_path="model.pth"):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        correct_val, total_val, all_preds, all_labels = 0, 0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with val_acc = {val_acc:.4f}")

        scheduler.step()

# Main Script
if __name__ == "__main__":
    csv_paths = ["processed_test_with_tags1.csv", "processed_test_with_tags2.csv", "processed_test_with_tags3.csv", "processed_test_with_tags4.csv"]
    label_mapping = {"left": 0, "rest": 1, "right": 2, "stop": 3}
    features, labels = [], []

    for csv_path in csv_paths:
        data = pd.read_csv(csv_path)
        eeg_channels = [col for col in data.columns if col.startswith('Channel')]
        for _, row in data.iterrows():
            try:
                signal = row[eeg_channels].values
                quaternion_feat = quaternion_features(signal)
                emd_feat = emd_features(signal)
                features.append(quaternion_feat + emd_feat)
                labels.append(label_mapping[row["Tag"]])
            except Exception as e:
                print(f"Skipping row due to error: {e}")

    sequence_length = 128
    num_channels = 14
    padded_features = []

    for feature in features:
        feature = np.array(feature).flatten()
        if len(feature) < sequence_length * num_channels:
            feature = np.pad(feature, (0, sequence_length * num_channels - len(feature)), 'constant')
        elif len(feature) > sequence_length * num_channels:
            feature = feature[:sequence_length * num_channels]
        padded_features.append(feature)

    features = np.array(padded_features).reshape(-1, num_channels, sequence_length)
    scaler = StandardScaler()
    reshaped_features = features.reshape(features.shape[0], -1)
    reshaped_features = scaler.fit_transform(reshaped_features)
    features = reshaped_features.reshape(features.shape[0], num_channels, sequence_length)
    joblib.dump(scaler, "scaler.pkl")

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_channels=14, num_classes=4).to(device)
    train_model(model, train_loader, val_loader, device)
