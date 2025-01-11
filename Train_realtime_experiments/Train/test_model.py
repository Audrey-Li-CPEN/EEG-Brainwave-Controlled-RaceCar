import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from joblib import load
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class DeepCNN1D(torch.nn.Module):
    def __init__(self, num_channels=14, num_classes=4):
        super(DeepCNN1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(num_channels, 64, kernel_size=31, padding=15)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()
        self.resblock1 = self._residual_block(64)
        self.resblock2 = self._residual_block(64)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, num_classes)
        )

    def _residual_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            torch.nn.BatchNorm1d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            torch.nn.BatchNorm1d(channels),
            torch.nn.ReLU()
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

class RealTimeEEGProcessor:
    def __init__(self, model_path, scaler_path, filter_coeffs_path, label_mapping, window_size=50, overlap=45):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.label_mapping = label_mapping
        self.buffer = []
        self.last_processed_size = 0
        self.last_prediction = None
        self.prediction_threshold = 0.35

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DeepCNN1D(num_channels=14, num_classes=len(label_mapping)).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load scaler and filter coefficients
        self.scaler = load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        filter_coeffs = np.load(filter_coeffs_path)
        self.b = filter_coeffs['b']
        self.a = filter_coeffs['a']
        print(f"Filter coefficients loaded from {filter_coeffs_path}")

        # Add prediction logging
        self.predictions_log = []  # Store predictions for CSV export

    def butter_bandpass_filter(self, data):
        return filtfilt(self.b, self.a, data)

    def process_window(self):
        try:
            print("\n" + "="*50)
            print(f"Processing window of size: {len(self.buffer)}")

            # Convert buffer to numpy array and transpose to match training format
            window = np.array(self.buffer).T  # Shape: (channels, samples)
            print(f"Initial window shape: {window.shape}")

            # Apply bandpass filter exactly as in training
            print("Applying bandpass filter...")
            for i in range(window.shape[0]):
                window[i] = self.butter_bandpass_filter(window[i])

            # Normalize using the loaded scaler
            print("Normalizing data...")
            reshaped_window = window.T  # Shape: (samples, channels)
            normalized_window = self.scaler.transform(reshaped_window).T

            # Prepare tensor for model (add batch dimension)
            features = torch.tensor(normalized_window[np.newaxis, :, :], dtype=torch.float32).to(self.device)
            print(f"Input feature shape: {features.shape}")

            # Get prediction with confidence
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)

                # Print all class probabilities
                probs_dict = {
                    label: prob.item()
                    for label, prob in zip(self.label_mapping.keys(), probabilities[0])
                }
                print("\nClass probabilities:")
                for label, prob in probs_dict.items():
                    print(f"{label}: {prob:.3f}")

                confidence, predicted = torch.max(probabilities, 1)

                if confidence.item() > self.prediction_threshold:
                    inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
                    predicted_label = inverse_label_mapping[predicted.item()]
                else:
                    predicted_label = 'rest'  # Default to rest for low confidence

                # Log prediction with confidence
                self.predictions_log.append({
                    'prediction': predicted_label,
                    'confidence': confidence.item(),
                    'probabilities': probs_dict
                })

                if predicted_label != self.last_prediction:
                    self.last_prediction = predicted_label
                    print(f"\n🎯 New prediction: {predicted_label}")
                    print(f"Confidence: {confidence.item():.3f}")
                else:
                    print(f"\n🔄 Same prediction as before: {predicted_label}")

            print("="*50 + "\n")

        except Exception as e:
            print(f"❌ Error processing window: {e}")

    def process_new_data(self, csv_path):
        try:
            # Read the entire CSV file
            data = pd.read_csv(csv_path)

            if len(data) <= self.last_processed_size:
                return

            # Identify EEG channels in the same order as training
            eeg_channels = [col for col in data.columns if col.startswith('Channel')]
            eeg_channels.sort()  # Ensure consistent channel ordering

            # Get new data
            new_data = data.iloc[self.last_processed_size:]

            # Update buffer with new data
            for _, row in new_data.iterrows():
                eeg_values = row[eeg_channels].astype(float).values
                self.buffer.append(eeg_values)

                if len(self.buffer) >= self.window_size:
                    self.process_window()
                    self.buffer = self.buffer[self.step_size:]  # Remove oldest samples based on step size

            self.last_processed_size = len(data)

        except Exception as e:
            print(f"Error processing data: {e}")

    def save_predictions_to_csv(self, output_path):
        """Save all predictions to a CSV file"""
        try:
            # Convert predictions log to DataFrame
            df = pd.DataFrame(self.predictions_log)

            # Save to CSV
            df.to_csv(output_path, index=False)
            print(f"Successfully saved {len(df)} predictions to: {output_path}")

        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")

class CSVHandler(FileSystemEventHandler):
    def __init__(self, processor, csv_path):
        self.processor = processor
        self.csv_path = csv_path

    def on_modified(self, event):
        if event.src_path == self.csv_path:
            self.processor.process_new_data(self.csv_path)

def monitor_csv(csv_path, model_path, scaler_path, filter_coeffs_path, label_mapping):
    # Initialize the EEG processor
    processor = RealTimeEEGProcessor(model_path, scaler_path, filter_coeffs_path, label_mapping)

    # Set up file monitoring
    event_handler = CSVHandler(processor, csv_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(csv_path), recursive=False)
    observer.start()

    try:
        print(f"Starting to monitor CSV file: {csv_path}")
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(0.1)  # Check every 100ms
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        observer.stop()
        # Save predictions before exiting
        predictions_output = "predictions_output_rt.csv"
        processor.save_predictions_to_csv(predictions_output)
    observer.join()

if __name__ == "__main__":
    csv_path = r"D:\Train\realtime_simulation.csv"
    model_path = r"D:\Train\trained_model.pth"
    scaler_path = r"D:\Train\scaler.pkl"
    filter_coeffs_path = r"D:\Train\filter_coeffs.npz"

    label_mapping = {'left': 0, 'rest': 1, 'right': 2, 'stop': 3}

    monitor_csv(csv_path, model_path, scaler_path, filter_coeffs_path, label_mapping)
