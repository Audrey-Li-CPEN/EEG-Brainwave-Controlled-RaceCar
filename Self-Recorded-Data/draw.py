import csv
import matplotlib.pyplot as plt

def load_csv(file_path):
    """Load a CSV file and return its content as a list of rows."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def plot_channel_sum_with_intervals(input_csv_path):
    # Load the CSV file
    data = load_csv(input_csv_path)

    # Get the header and determine the required columns
    header = data[0]
    time_col_index = header.index('Time:128Hz')
    tag_col_index = header.index('Tag')
    channel_columns = [col for col in header if col.startswith('Channel')]
    channel_indices = [header.index(col) for col in channel_columns]

    # Extract time, channel sums, and tags
    times = []
    channel_sums = []
    tags = []

    for row in data[1:]:  # Skip header
        try:
            time = float(row[time_col_index])
            channel_sum = sum(float(row[idx]) for idx in channel_indices if row[idx].strip())
            tag = row[tag_col_index].strip()

            times.append(time)
            channel_sums.append(channel_sum)
            tags.append(tag)
        except ValueError:
            continue  # Skip rows with invalid data

    # Plot the data with intervals
    plt.figure(figsize=(16, 8))

    unique_tags = ['left', 'right', 'rest', 'stop']
    colors = {'left': 'red', 'right': 'green', 'rest': 'gray', 'stop': 'blue'}

    start_time = None
    current_tag = None

    for i in range(len(tags)):
        if tags[i] != current_tag:  # Tag changes
            if current_tag in unique_tags and start_time is not None:
                plt.plot(times[start_idx:i], channel_sums[start_idx:i],
                         color=colors[current_tag])
            current_tag = tags[i]
            start_time = times[i]
            start_idx = i

    # Plot the final interval
    if current_tag in unique_tags and start_time is not None:
        plt.plot(times[start_idx:], channel_sums[start_idx:],
                 color=colors[current_tag])

    plt.xlabel('Time (s)')
    plt.ylabel('Sum of Channels')
    plt.title('Sum of Channels Over Time with Intervals')
    plt.legend(unique_tags, loc='upper right')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    input_csv_path = r"D:\\Code\\EEG\\DATA\\processed_test_with_tags1.csv"
    plot_channel_sum_with_intervals(input_csv_path)
