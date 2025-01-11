import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from scipy.signal import butter, filtfilt
import asyncio
import websockets
import json
import time

##########################
# Network Model & Filters
##########################
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# If a bandpass filter from 8 to 30 Hz is needed, you can use the following function
def bandpass_filter(data, lowcut=8, highcut=30, fs=128):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)

class DeepConvNet(nn.Module):
    def __init__(self, num_channels=14, num_classes=4):
        super(DeepConvNet, self).__init__()
        
        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, 25, kernel_size=10),
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(3)
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size=10),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.5)
        )
        
        # Additional convolution
        self.final_conv = nn.Sequential(
            nn.Conv1d(50, 100, kernel_size=5),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self._to_linear = None
        self._get_conv_output_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)
        )

    def _get_conv_output_size(self):
        bs = 1
        x = torch.randn(bs, 14, 128)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.final_conv(x)
        self._to_linear = x.size(1) * x.size(2)

    def forward(self, x):
        # Ensure the input format is (batch, channels, time_steps)
        if x.size(1) != 14:
            x = x.transpose(1, 2)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


##################################
# WebSocket-related global values
##################################
connected_clients = set()  # Used to store all connected WebSocket clients

#######################################
# Send prediction results to all clients
#######################################
async def broadcast_prediction(prediction: str):
    if not connected_clients:
        return
    message = json.dumps({"direction": prediction})  # The script on the car side will parse this field
    # Asynchronously send to all connected clients
    await asyncio.wait([ws.send(message) for ws in connected_clients])

############################################
# WebSocket server connection handling callback
############################################
async def handler(websocket, path):
    # New client connection
    connected_clients.add(websocket)
    try:
        # Keep the connection until an exception occurs or it is disconnected
        await asyncio.Future()  # Equivalent to infinite waiting
    except Exception as e:
        print(f"WebSocket client exception: {e}")
    finally:
        # Client disconnected
        connected_clients.remove(websocket)

################################################
# Real-time inference function: perform inference
# and send results through WebSocket
################################################
async def simulate_real_time_inference(
    csv_file='data.csv',
    model_path='model_final.pth',      
    scaler_path='scaler_final.pkl', 
    fs=128,
    lowcut=8.0,
    highcut=50.0,
    window_size=128,
    overlap=113
):
    # Read the CSV file initially to determine columns, etc.
    data = pd.read_csv(csv_file)
    eeg_channels = [f'Channel {i}' for i in range(1, 15)]
    has_tag = 'Tag' in data.columns

    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepConvNet(num_channels=14, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Index to label mapping
    idx_to_label = {0: 'left', 1: 'rest', 2: 'right', 3: 'stop'}

    step_size = window_size - overlap

    # Used to cache window data
    buffer_data = []
    buffer_tags = [] if has_tag else None

    correct_predictions = 0
    total_predictions = 0

    # Used to record how many rows have been processed
    last_row_count = 0

    while True:
        # Re-read CSV
        data = pd.read_csv(csv_file)
        row_count = len(data)
        
        # If the new row count is greater than the last processed count, new data has been added
        if row_count > last_row_count:
            # Extract newly added rows
            new_data = data.iloc[last_row_count: row_count]

            # Add rows to the buffer one by one
            for i in range(len(new_data)):
                # Only take EEG channel data
                current_row = new_data[eeg_channels].iloc[i].values.astype(float)
                buffer_data.append(current_row)

                # If there is a tag, store it together
                if has_tag:
                    current_tag = new_data['Tag'].iloc[i]
                    buffer_tags.append(current_tag)

                # When the buffer has enough data for a full window, perform inference
                if len(buffer_data) >= window_size:
                    window_array = np.array(buffer_data[:window_size])  # Take the first window_size length data

                    # Normalization
                    window_scaled = scaler.transform(window_array)

                    # Bandpass filter
                    filtered_window = np.array([
                        butter_bandpass_filter(window_scaled[:, ch], lowcut, highcut, fs)
                        for ch in range(len(eeg_channels))
                    ]).T

                    # Convert to tensor and infer
                    input_tensor = torch.tensor(filtered_window, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        pred_class = torch.argmax(outputs, dim=1).item()

                    predicted_label = idx_to_label[pred_class]

                    # If there is a true label, compute accuracy
                    if has_tag:
                        true_label = buffer_tags[window_size - 1]  # Take the label at the end of the window
                        is_correct = (predicted_label == true_label)
                        correct_predictions += int(is_correct)
                        total_predictions += 1
                        accuracy = (correct_predictions / total_predictions) * 100
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"windows [{last_row_count} ~ {last_row_count + window_size - 1}] -> "
                              f"Predicted: {predicted_label}, True Tag: {true_label}, "
                              f"Accuracy: {accuracy:.2f}%")
                    else:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"windows [{last_row_count} ~ {last_row_count + window_size - 1}] -> "
                              f"Predicted: {predicted_label}")

                    # Broadcast the prediction result through WebSocket
                    await broadcast_prediction(predicted_label)

                    # Move the window buffer
                    buffer_data = buffer_data[step_size:]
                    if has_tag:
                        buffer_tags = buffer_tags[step_size:]

            # Update last_row_count
            last_row_count = row_count

        # Sleep for a while to avoid overly frequent reads and writes
        await asyncio.sleep(1)


###############################
# Main function: start service
###############################
async def main():
    # Start WebSocket server listening on 0.0.0.0:5000
    server = await websockets.serve(handler, '0.0.0.0', 5000, ping_interval=None)
    print("WebSocket server started at ws://0.0.0.0:5000/signal")

    # Start real-time inference and send results to connected clients
    await simulate_real_time_inference(
        csv_file='data.csv',
        model_path='model_final.pth',      
        scaler_path='scaler_final.pkl', 
        fs=128, 
        lowcut=8.0,
        highcut=50.0,
        window_size=128,
        overlap=113
    )

    # If the inference ends, you can decide whether to keep the server open 
    # (typically does not end automatically)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
