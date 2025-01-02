import csv
import os

def load_csv(file_path):
    """Load a CSV file and return its content as a list of rows."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def save_csv(file_path, data):
    """Save data to a CSV file."""
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def process_test_with_tags(input_csv_path, output_csv_path):
    # Load the input CSV
    data = load_csv(input_csv_path)

    # Get the index of the 'Time:128Hz' and 'Tag' columns
    header = data[0]
    time_col_index = header.index('Time:128Hz')
    tag_col_index = header.index('Tag')

    # Step 1: Filter out rows without tags
    filtered_data = [row for row in data[1:] if len(row) > tag_col_index and row[tag_col_index].strip()]

    if not filtered_data:
        print("No data with tags found in the input file.")
        return

    # Step 2: Find the minimum time in the filtered data
    min_time = min(float(row[time_col_index]) for row in filtered_data)

    # Adjust the time values by subtracting the minimum time
    for row in filtered_data:
        row[time_col_index] = str(float(row[time_col_index]) - 0)

    # Step 3: Normalize values in Channel columns
    channel_columns = [col for col in header if col.startswith('Channel')]
    channel_indices = [header.index(col) for col in channel_columns]

    # Find the minimum value across all channel columns
    # min_channel_value = float('inf')
    # for row in filtered_data:
    #     for index in channel_indices:
    #         try:
    #             value = float(row[index])
    #             if value < min_channel_value:
    #                 min_channel_value = value
    #         except ValueError:
    #             continue  # Skip invalid or empty values

    # Subtract the minimum value from all channel values
    for row in filtered_data:
        for index in channel_indices:
            try:
                row[index] = str(float(row[index]) - 0)
            except ValueError:
                continue  # Skip invalid or empty values

    # Add the header back to the filtered data
    filtered_data.insert(0, header)

    # Save the processed data to the output CSV
    save_csv(output_csv_path, filtered_data)

if __name__ == "__main__":
    input_csv_path = r"D:\\Code\\EEG\\DATA\\test_with_tags1.csv"
    output_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags1.csv"
    process_test_with_tags(input_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    input_csv_path = r"D:\\Code\\EEG\\DATA\\test_with_tags2.csv"
    output_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags2.csv"
    process_test_with_tags(input_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    input_csv_path = r"D:\\Code\\EEG\\DATA\\test_with_tags3.csv"
    output_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags3.csv"
    process_test_with_tags(input_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    input_csv_path = r"D:\\Code\\EEG\\DATA\\test_with_tags4.csv"
    output_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags4.csv"
    process_test_with_tags(input_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    input_csv_path = r"D:\\Code\\EEG\\DATA\\test_with_tags5.csv"
    output_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags5.csv"
    process_test_with_tags(input_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")


