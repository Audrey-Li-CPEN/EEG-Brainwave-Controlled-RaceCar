import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import joblib

# ----------------------------------------------------------------------
#                      Utility Functions
# ----------------------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, b, a):
    return filtfilt(b, a, data)

def save_scaler(scaler, file_path):
    joblib.dump(scaler, file_path)
    print(f"Scaler saved to {file_path}")

# ----------------------------------------------------------------------
#                    Model and Dataset Classes
# ----------------------------------------------------------------------

class EEGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

# ----------------------------------------------------------------------
#                   Offline Preprocessing
# ----------------------------------------------------------------------
def offline_preprocessing(csv_paths, label_mapping, scaler_path, filter_coeffs_path,
                           fs=128, lowcut=8.0, highcut=50.0, 
                          window_size=50, overlap=45):
    """
    Preprocess data for training.
    Saves the scaler for real-time preprocessing.
    """
    all_features = []  # Corrected initialization
    all_labels = []    # Corrected initialization
    
    # Pre-compute filter coefficients (same for entire dataset)
    b, a = butter_bandpass(lowcut, highcut, fs, order=4)

    np.savez(filter_coeffs_path, b=b, a=a)
    print(f"Filter coefficients saved to {filter_coeffs_path}")

    eeg_channels = None
    label_col = 'Tag'
    

    # Step 1: Collect all data
    raw_data_list = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df[label_col] = df[label_col].map(label_mapping)
        raw_data_list.append(df)

    # Concatenate for consistent scaling
    full_data = pd.concat(raw_data_list, ignore_index=True)
    if not eeg_channels:
        eeg_channels = [col for col in full_data.columns if col.startswith('Channel')]

    # Process data
    for df in raw_data_list:
        step_size = window_size - overlap
        labels = df[label_col].values
        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size
            window = df.iloc[start:end]
            window_label = labels[start]
            if window_label < 0:
                continue

            # Filter each channel
            filtered_window = []
            for ch in eeg_channels:
                signal = window[ch].values
                filtered_sig = butter_bandpass_filter(signal, b, a)
                filtered_window.append(filtered_sig)

            filtered_window = np.array(filtered_window)
            all_features.append(filtered_window)
            all_labels.append(window_label)

    # Standardize features channel-wise
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    num_windows, num_ch, win_sz = all_features.shape
    reshaped_for_scaler = all_features.transpose(0, 2, 1).reshape(-1, num_ch)

    scaler = StandardScaler()
    scaler.fit(reshaped_for_scaler)

    # Transform the data
    scaled_data = scaler.transform(reshaped_for_scaler)
    scaled_data = scaled_data.reshape(num_windows, win_sz, num_ch).transpose(0, 2, 1)

    # Save the scalerand filter coefficients
    save_scaler(scaler, scaler_path)



    return scaled_data, all_labels, (b, a)

# ----------------------------------------------------------------------
#                   Training the Model
# ----------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Val Acc: {val_acc:.4f}")
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

# ----------------------------------------------------------------------
#                       Main Script
# ----------------------------------------------------------------------

if __name__ == "__main__":
    csv_paths = [
        r"D:\Train\processed_test_with_tags1.csv",
        r"D:\Train\processed_test_with_tags2.csv",
        r"D:\Train\processed_test_with_tags3.csv",
        r"D:\Train\processed_test_with_tags4.csv"
    ]
    label_mapping = {'left': 0, 'rest': 1, 'right': 2, 'stop': 3}
    save_path = r"D:\Train\trained_model_rt.pth"
    scaler_path = r"D:\Train\scaler.pkl"
    filter_coeffs_path = r"D:\Train\filter_coeffs.npz"

    features, labels, filter_coeffs = offline_preprocessing(
        csv_paths, label_mapping, scaler_path, filter_coeffs_path,
        fs=128, lowcut=8.0, highcut=50.0, window_size=50, overlap=45
    )

    
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCNN1D(num_channels=14, num_classes=len(label_mapping)).to(device)
    train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3, save_path=save_path)
