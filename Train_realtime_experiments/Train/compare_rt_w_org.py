import pandas as pd
import numpy as np

# Parameters 
total_rows = 38366
window_size = 50
step_size = 5

# Load predictions and original data
predictions = pd.read_csv("predictions_output_rt.csv")["prediction"]
original_data = pd.read_csv("processed_test_with_tags5.csv")

# Initialize a list to hold the row-wise predictions
row_predictions = [-1] * total_rows  # -1 indicates no prediction for a row

# Assign predictions to rows based on window
for i, prediction in enumerate(predictions):
    start_idx = i * step_size
    end_idx = start_idx + window_size
    # Assign the prediction to all rows in this window
    for j in range(start_idx, min(end_idx, total_rows)):
        row_predictions[j] = prediction

# Create a DataFrame to compare original tags and predictions
comparison_df = original_data.copy()
comparison_df['predicted_tag'] = row_predictions

# Save the comparison for analysis
comparison_df.to_csv("comparison_output.csv", index=False)
print("Comparison saved to comparison_output.csv")
