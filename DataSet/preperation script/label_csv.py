import pandas as pd

def label_signal(signal_file, annotation_file, output_file):
    signal_data = pd.read_csv(signal_file)
    annotation_data = pd.read_csv(annotation_file)

    signal_data['Label'] = None

    for i, row in annotation_data.iterrows():
        if row['Annotation_Description'] == 'OVTK_GDF_Feedback_Continuous':
            start_time = row['Annotation_Time (s)']
            
            trial_label_row = annotation_data.iloc[:i].loc[
                annotation_data['Annotation_Description'].isin(['OVTK_GDF_Right', 'OVTK_GDF_Left'])
            ].iloc[-1]
            trial_label = 'right' if trial_label_row['Annotation_Description'] == 'OVTK_GDF_Right' else 'left'
            
            end_time = annotation_data.iloc[i+1:].loc[
                annotation_data['Annotation_Description'] == 'OVTK_GDF_End_Of_Trial'
            ].iloc[0]['Annotation_Time (s)']
            
            signal_data.loc[
                (signal_data['Time (s)'] >= start_time) & (signal_data['Time (s)'] < end_time), 'Label'
            ] = trial_label

    signal_data.to_csv(output_file, index=False)
    print(f"Labeled signal data has been saved to {output_file}")

label_signal(
    signal_file="data12_1_signals.csv",
    annotation_file="data12_1_annotations.csv",
    output_file="data12_1.csv"
)
