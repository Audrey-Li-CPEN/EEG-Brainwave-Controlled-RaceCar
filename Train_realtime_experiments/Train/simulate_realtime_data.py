import pandas as pd
import time
import os

def simulate_realtime_streaming(input_csv_path, output_csv_path, delay=0.1):
    """
    Reads data from input CSV and writes it gradually to output CSV to simulate real-time streaming.
    
    Args:
        input_csv_path: Path to the complete recorded CSV
        output_csv_path: Path where the simulated real-time data will be written
        delay: Delay between writes in seconds (default: 0.1s)
    """
    # Read the complete dataset
    print(f"Reading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Get the header row
    header = df.columns.tolist()
    
    # Create a new CSV file with just the header
    print(f"Creating output file: {output_csv_path}")
    df.iloc[0:0].to_csv(output_csv_path, index=False)
    
    # Simulate real-time data streaming
    chunk_size = 10  # Number of rows to write at once
    total_rows = len(df)
    
    print(f"Starting simulation with {total_rows} rows...")
    print("Press Ctrl+C to stop...")
    
    try:
        for i in range(0, total_rows, chunk_size):
            # Get the next chunk of data
            chunk = df.iloc[i:i+chunk_size]
            
            # Append to the output CSV
            chunk.to_csv(output_csv_path, mode='a', header=False, index=False)
            
            # Print progress
            progress = (i + chunk_size) / total_rows * 100
            print(f"\rProgress: {min(progress, 100):.1f}%", end="")
            
            # Wait before writing next chunk
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
    else:
        print("\nSimulation completed successfully!")

if __name__ == "__main__":
    # Paths
    input_csv = r"D:\Train\processed_test_with_tags5.csv"
    output_csv = r"D:\Train\realtime_simulation.csv"
    
    # Start simulation
    simulate_realtime_streaming(input_csv, output_csv)
