import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import joblib

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

# Add bandpass filtering for motor imagery relevant frequencies (8-30 Hz)
def bandpass_filter(data, lowcut=8, highcut=30, fs=128):
    nyquist = fs/2
    low = lowcut/nyquist
    high = highcut/nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

# Add Common Average Reference (CAR)
def apply_car(data):
    mean = np.mean(data, axis=1, keepdims=True)
    return data - mean

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
class DeepConvNet(nn.Module):
    def __init__(self, num_channels=14, num_classes=4):
        super(DeepConvNet, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, 25, kernel_size=10),  # Input: (batch, 14, timesteps)
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(3)
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=10),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(3)
        )
        
        # Calculate the size of the flattened features
        self._to_linear = None
        self._get_conv_output_size()
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, num_classes)
        )

    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size
        bs = 1
        x = torch.randn(bs, 14, 50)  # Assuming input size is (batch, channels, timesteps)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        self._to_linear = x.size(1) * x.size(2)

    def forward(self, x):
        # Ensure input is in the correct format (batch, channels, timesteps)
        if x.size(1) != 14:
            x = x.transpose(1, 2)  # Transpose from (batch, timesteps, channels) to (batch, channels, timesteps)
            
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Main Script
def preprocess_data(file_path, label_mapping, fs=128, lowcut=8.0, highcut=50.0, window_size=50, overlap=45):
    # Read data
    data = pd.read_csv(file_path)
    eeg_channels = [f'Channel {i}' for i in range(1, 15)]
    
    # Initialize scaler with first window to set mean/std
    scaler = StandardScaler()
    first_window = data[eeg_channels].iloc[:window_size].values
    scaler.fit(first_window)
    
    # Process data window by window (simulating real-time)
    windows = []
    labels = []
    step_size = window_size - overlap
    
    for i in range(0, len(data) - window_size + 1, step_size):
        # Get window of raw data
        window = data[eeg_channels].iloc[i:i + window_size].values
        
        # Normalize this window only (as we would in real-time)
        window_normalized = scaler.transform(window)
        
        # Apply bandpass filter to normalized window
        window_filtered = np.array([
            butter_bandpass_filter(
                window_normalized[:, ch], 
                lowcut, 
                highcut, 
                fs
            ) for ch in range(len(eeg_channels))
        ]).T
        
        #  convert to numeric using label_mapping
        window_label = data['Tag'].iloc[i]
        numeric_label = label_mapping[window_label]
        
        windows.append(window_filtered)
        labels.append(numeric_label)
    
    return np.array(windows), np.array(labels), scaler

def preprocess_multiple_files(file_paths, label_mapping):
    all_windows = []
    all_labels = []
    first_file = True
    
    for file_path in file_paths:
        # Pass label_mapping to preprocess_data
        windows, labels, scaler = preprocess_data(file_path, label_mapping)
        if first_file:
            # Save scaler from first file for real-time use
            joblib.dump(scaler, 'scaler.pkl')
            first_file = False
        
        all_windows.extend(windows)
        all_labels.extend(labels)
    
    return np.array(all_windows), np.array(all_labels)

# Add data augmentation
def augment_data(data, labels):
    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, data.shape)
    augmented_data = np.concatenate([data, data + noise])
    augmented_labels = np.concatenate([labels, labels])
    
    # Add time shifting
    shifted_data = np.roll(data, shift=5, axis=2)
    augmented_data = np.concatenate([augmented_data, shifted_data])
    augmented_labels = np.concatenate([augmented_labels, labels])
    
    return augmented_data, augmented_labels

# Training
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
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

# When creating the DataLoader, ensure the data is in the correct format
def prepare_data(windows, labels):
    # Convert to torch tensors
    X = torch.FloatTensor(windows)
    y = torch.LongTensor(labels)
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    return dataset

if __name__ == "__main__":
    file_paths = [
        r"D:\Train\processed_test_with_tags1.csv",
        r"D:\Train\processed_test_with_tags2.csv",
        r"D:\Train\processed_test_with_tags3.csv",
        r"D:\Train\processed_test_with_tags4.csv"
    ]
    save_path = r"D:\Train\model_v3.pth"

    # Define label mapping
    label_mapping = {'left': 0, 'rest': 1, 'right': 2, 'stop': 3}

    # Preprocess data window by window
    features, labels = preprocess_multiple_files(file_paths, label_mapping)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepConvNet(num_channels=14, num_classes=4).to(device)

    # Prepare data loaders
    train_dataset = prepare_data(X_train, y_train)
    val_dataset = prepare_data(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model and save it
    train_model(model, train_loader, val_loader, device, save_path=save_path)

    print(f"Model training completed. Model saved to {save_path}")

