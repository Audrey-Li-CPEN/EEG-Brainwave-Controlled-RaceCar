import pyedflib
import pandas as pd

def edf_to_csv_with_annotations(edf_file, csv_file):
    
    f = pyedflib.EdfReader(edf_file)
    n_signals = f.signals_in_file  
    signal_labels = f.getSignalLabels()  
    sampling_frequencies = f.getSampleFrequencies()  

    data = pd.DataFrame()
    for i in range(n_signals):
        signal = f.readSignal(i)  
        data[signal_labels[i]] = signal  

    duration = f.file_duration
    time_index = pd.Series([i / sampling_frequencies[0] for i in range(len(data))])
    data.insert(0, "Time (s)", time_index)

    annotations = f.readAnnotations()
    stim_data = pd.DataFrame({
        "Annotation_Time (s)": annotations[0],
        "Annotation_Duration (s)": annotations[1],
        "Annotation_Description": annotations[2],
    })

    data.to_csv(csv_file.replace('.csv', '_signals.csv'), index=False)
    stim_data.to_csv(csv_file.replace('.csv', '_annotations.csv'), index=False)

    f._close() 
    print(f"EDF file '{edf_file}' has been converted to CSV files:")
    print(f"  - Signals: '{csv_file.replace('.csv', '_signals.csv')}'")
    print(f"  - Annotations: '{csv_file.replace('.csv', '_annotations.csv')}'")

edf_to_csv_with_annotations("../../MI-subject12/MI-subject12/1.edf", "data12_1.csv")
