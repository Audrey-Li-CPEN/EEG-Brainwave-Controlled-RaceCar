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
import random

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
def preprocess_data(file_path, label_mapping, fs=128, lowcut=8.0, highcut=30.0, window_size=128, overlap=120):
    """
    Modified to use real-time compatible normalization
    """
    data = pd.read_csv(file_path)
    eeg_channels = [f'Channel {i}' for i in range(1, 15)]
    
    windows = []
    labels = []
    step_size = window_size - overlap
    
    # Initialize running statistics for each channel
    channel_means = np.zeros(len(eeg_channels))
    channel_stds = np.ones(len(eeg_channels))
    alpha = 0.01  # Update rate for running statistics
    
    for i in range(0, len(data) - window_size + 1, step_size):
        # Get window
        window = data[eeg_channels].iloc[i:i + window_size].values
        
        # Update running statistics (as would be done in real-time)
        window_mean = np.mean(window, axis=0)
        window_std = np.std(window, axis=0)
        channel_means = (1 - alpha) * channel_means + alpha * window_mean
        channel_stds = (1 - alpha) * channel_stds + alpha * window_std
        
        # Normalize using running statistics
        window_normalized = (window - channel_means) / (channel_stds + 1e-6)
        
        # Apply bandpass filter
        window_filtered = np.array([
            butter_bandpass_filter(
                window_normalized[:, ch], 
                lowcut, 
                highcut, 
                fs
            ) for ch in range(len(eeg_channels))
        ]).T
        
        # Get label
        window_label = data['Tag'].iloc[i]
        numeric_label = label_mapping[window_label]
        
        windows.append(window_filtered)
        labels.append(numeric_label)
    
    return np.array(windows), np.array(labels)

def preprocess_multiple_files(file_paths, label_mapping):
    all_windows = []
    all_labels = []
    first_file = True
    
    for file_path in file_paths:
        # Pass label_mapping to preprocess_data
        windows, labels = preprocess_data(file_path, label_mapping)
        if first_file:
            # Save scaler from first file for real-time use
            joblib.dump(scaler, 'scaler_v5.pkl')
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
def train_model(model, train_loader, val_loader, device, save_path, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Add random noise for robustness
            if random.random() < 0.5:
                noise = torch.randn_like(inputs) * 0.1
                inputs = inputs + noise
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# When creating the DataLoader, ensure the data is in the correct format
def prepare_data(windows, labels):
    # Convert to torch tensors
    X = torch.FloatTensor(windows)
    y = torch.LongTensor(labels)
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    return dataset

class RealTimeEEGProcessor:
    def __init__(self, window_size=128, num_channels=14, fs=128, lowcut=8.0, highcut=30.0):
        self.window_size = window_size
        self.num_channels = num_channels
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        
        # Initialize running statistics
        self.channel_means = np.zeros(num_channels)
        self.channel_stds = np.ones(num_channels)
        self.alpha = 0.01  # Update rate
        
        # Initialize buffer
        self.buffer = np.zeros((window_size, num_channels))
        self.buffer_idx = 0
        
    def update(self, new_sample):
        """
        Update buffer with new sample and return processed window if available
        new_sample: array of shape (num_channels,)
        """
        # Update buffer
        self.buffer[self.buffer_idx] = new_sample
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        
        if self.buffer_idx == 0:  # Buffer is full
            # Update running statistics
            window_mean = np.mean(self.buffer, axis=0)
            window_std = np.std(self.buffer, axis=0)
            self.channel_means = (1 - self.alpha) * self.channel_means + self.alpha * window_mean
            self.channel_stds = (1 - self.alpha) * self.channel_stds + self.alpha * window_std
            
            # Normalize
            window_normalized = (self.buffer - self.channel_means) / (self.channel_stds + 1e-6)
            
            # Apply bandpass filter
            window_filtered = np.array([
                butter_bandpass_filter(
                    window_normalized[:, ch], 
                    self.lowcut, 
                    self.highcut, 
                    self.fs
                ) for ch in range(self.num_channels)
            ]).T
            
            return window_filtered
            
        return None

# Example usage in real-time:
def real_time_prediction(model, processor, new_sample):
    """
    Process new sample and make prediction if window is ready
    """
    processed_window = processor.update(new_sample)
    
    if processed_window is not None:
        # Prepare for model
        window_tensor = torch.FloatTensor(processed_window).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(window_tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        return prediction
    
    return None

if __name__ == "__main__":
    file_paths = [
        r"D:\Train\processed_test_with_tags1.csv",
        r"D:\Train\processed_test_with_tags2.csv",
        r"D:\Train\processed_test_with_tags3.csv",
        r"D:\Train\processed_test_with_tags4.csv"
    ]
    save_path = r"D:\Train\model_v5.pth"

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

