import csv
import os

# Define the input folder
input_folder = r"D:\\Code\\EEG\\DATA"
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

def add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path):
    # Load test.csv and tags.csv
    test_data = load_csv(test_csv_path)
    tags_data = load_csv(tags_csv_path)

    # Get the index of the 'Time:128Hz' column in test.csv
    time_col_index = test_data[0].index('Time:128Hz')

    # Add a new column for tags in test.csv
    if 'Tag' not in test_data[0]:
        test_data[0].append('Tag')

    # Parse tags.csv and process the tags
    current_tag = None
    tag_start_time = None

    for row in tags_data[1:]:  # Skip header row
        tag_time = float(row[0])
        tag_label = row[1]

        if 'rest' not in tag_label:
            if '_start' in tag_label:
                current_tag = tag_label.split('_')[0]
                tag_start_time = tag_time
            elif '_end' in tag_label and current_tag is not None:
                tag_end_time = tag_time

                # Add the tag to the relevant rows in test.csv
                for test_row in test_data[1:]:  # Skip header row
                    test_time = float(test_row[time_col_index])
                    if tag_start_time <= test_time <= tag_end_time:
                        if len(test_row) == len(test_data[0]) - 1:  # If Tag column is missing
                            test_row.append(current_tag)
                        else:
                            test_row[-1] = current_tag

                # Reset current tag
                current_tag = None
                tag_start_time = None

    # Add 'rest' to all remaining rows without a tag
    tag_start_time = float(tags_data[1][0])  # First time in tags.csv
    tag_end_time = float(tags_data[-1][0])  # Last time in tags.csv

    for test_row in test_data[1:]:  # Skip header row
        test_time = float(test_row[time_col_index])
        if tag_start_time <= test_time <= tag_end_time:
            if len(test_row) == len(test_data[0]) - 1:  # If Tag column is missing
                test_row.append('rest')
            elif not test_row[-1]:  # If Tag column is empty
                test_row[-1] = 'rest'

    # Save the modified test.csv to a new file
    save_csv(output_csv_path, test_data)

if __name__ == "__main__":
    test_csv_path = os.path.join(input_folder, "test1.csv")
    tags_csv_path = os.path.join(input_folder, "tags1.csv")
    output_csv_path = os.path.join(input_folder, "test_with_tags1.csv")

    add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    test_csv_path = os.path.join(input_folder, "test2.csv")
    tags_csv_path = os.path.join(input_folder, "tags2.csv")
    output_csv_path = os.path.join(input_folder, "test_with_tags2.csv")

    add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    test_csv_path = os.path.join(input_folder, "test3.csv")
    tags_csv_path = os.path.join(input_folder, "tags3.csv")
    output_csv_path = os.path.join(input_folder, "test_with_tags3.csv")

    add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    test_csv_path = os.path.join(input_folder, "test4.csv")
    tags_csv_path = os.path.join(input_folder, "tags4.csv")
    output_csv_path = os.path.join(input_folder, "test_with_tags4.csv")

    add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    test_csv_path = os.path.join(input_folder, "test5.csv")
    tags_csv_path = os.path.join(input_folder, "tags5.csv")
    output_csv_path = os.path.join(input_folder, "test_with_tags5.csv")

    add_tags_to_test_csv(test_csv_path, tags_csv_path, output_csv_path)
    print(f"Processed file saved to {output_csv_path}")

    

