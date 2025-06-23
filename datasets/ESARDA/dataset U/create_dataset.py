import pandas as pd
import os 
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import random
import json
from sklearn import preprocessing

def parse_spectrum_file(file_path):
    """
    Parse a gamma spectrum file and extract metadata.
    
    Args:
        file_path (str): Path to the spectrum file
        
    Returns:
        dict: Dictionary containing metadata and spectrum content
    """
    try:
        with open(file_path, encoding="utf8", errors='ignore') as f:
            file_lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Extract metadata (first 8 lines)
    metadata_lines = [file_lines[i].strip() for i in range(8) if i != 3]
    
    # Parse metadata
    parsed_metadata = []
    for line in metadata_lines:
        parts = line.split(":")
        for part in parts:
            parsed_metadata.extend(part.split('\t'))
    
    # Clean and create metadata dictionary
    clean_metadata = [item for item in parsed_metadata if len(item) > 1 and item != ' ']
    
    metadata_dict = {}
    for i in range(len(clean_metadata) // 2):
        key = re.sub(r"\s+", " ", clean_metadata[2*i]).strip()
        # Normalize units
        key = key.replace("(cm )", "(cm)").replace("( cm )", "(cm)").replace("( cm)", "(cm)")
        key = key.replace(" (s)", "(s)").replace("(s) ", "(s)").replace("( mm)", "(mm)")
        
        value = re.sub(r"\s+", " ", clean_metadata[2*i + 1]).strip()
        metadata_dict[key] = value
    
    return metadata_dict, file_lines

def extract_spectrum_data(file_lines):
    """
    Extract spectrum data (from line 8 onwards).
    
    Args:
        file_lines (list): List of file lines
        
    Returns:
        np.array: Array containing spectrum data
    """
    content = []
    for line in file_lines[8:]:
        values = line.strip().split()
        content.extend([int(val) for val in values if val])
    
    content_array = np.array(content)
    
    # Verify that spectrum contains 4096 channels
    if len(content_array) != 4096:
        raise ValueError(f"Spectrum must contain 4096 channels, found: {len(content_array)}")
    
    return content_array

def process_metadata(metadata_dict, detector_count):
    """
    Process and normalize extracted metadata.
    
    Args:
        metadata_dict (dict): Dictionary of raw metadata
        detector_count (int): Detector identifier
        
    Returns:
        dict: Dictionary of processed metadata
    """
    # Expected columns in final dataset
    expected_columns = [
        "File name", "Detector", "Declared enrichment", "live counting times", 
        "real counting times", "Detector quanti", "FWHM at 185 keV (keV)"
    ]
    
    # Extract declared enrichment
    if "Declared enrichment" in metadata_dict:
        metadata_dict["Declared enrichment"] = metadata_dict["Declared enrichment"].split(" ")[0]
    
    # Separate live/real counting times
    if "Live/real counting times(s)" in metadata_dict:
        live_time, real_time = metadata_dict["Live/real counting times(s)"].split("/")
        metadata_dict["live counting times"] = live_time
        metadata_dict["real counting times"] = real_time
        
    # Add detector identifier
    metadata_dict["Detector quanti"] = detector_count
    
    # Create final dictionary with only necessary columns
    processed_dict = {}
    for col in expected_columns:
        if col in metadata_dict:
            processed_dict[col] = metadata_dict[col]
        else:
            print(f"Missing column: {col}")
            print(f"Available columns: {list(metadata_dict.keys())}")
            sys.exit(f"Fatal error: missing information {col}")
    
    return processed_dict

def load_and_process_spectra(base_path):
    """
    Load and process all spectrum files from the base directory.
    
    Args:
        base_path (str): Path to directory containing spectra
        
    Returns:
        pd.DataFrame: DataFrame containing all processed spectra
    """
    dataframes = []
    
    # Iterate through all detectors
    for detector_count, detector_type in enumerate(os.listdir(f"{base_path}/U")):
        print(f"Processing detector: {detector_type}")
        
        # Iterate through all files for this detector
        for filename in os.listdir(f"{base_path}/U/{detector_type}"):
            print(f"  Processing file: {filename}")
            file_path = f"{base_path}/U/{detector_type}/{filename}"
            
            # Parse file
            result = parse_spectrum_file(file_path)
            if result is None:
                continue
                
            metadata_dict, file_lines = result
            
            # Extract spectrum data
            try:
                spectrum_data = extract_spectrum_data(file_lines)
            except ValueError as e:
                print(f"Error in file {filename}: {e}")
                sys.exit("Fatal error")
            
            # Process metadata
            processed_metadata = process_metadata(metadata_dict, detector_count)
            processed_metadata["content"] = [spectrum_data]
            
            # Create DataFrame for this file
            df_file = pd.DataFrame.from_dict(processed_metadata)
            dataframes.append(df_file)
    
    # Concatenate all DataFrames
    final_df = pd.concat(dataframes, ignore_index=True, join="inner")
    final_df = final_df.dropna()
    
    return final_df

def clean_and_convert_data(df):
    """
    Clean and convert data to appropriate types.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print(df.columns)

    # Convert counting times to integers
    df.iloc[:, 3] = [int(val.split(" ")[0]) for val in df.iloc[:, 3]]
    df.iloc[:, 4] = [int(val.split(" ")[0]) for val in df.iloc[:, 4]]
    
    # Convert enrichment to float

    df.iloc[:, 2] = [float(val.split(" ")[0]) for val in df.iloc[:, 2]]

    return df

def filter_and_normalize_data(df):
    """
    Filter data and apply normalization.
    
    Args:
        df (pd.DataFrame): DataFrame to filter and normalize
        
    Returns:
        pd.DataFrame: Filtered and normalized DataFrame
    """
    # Filter: enrichment < 90% and real counting time > 1000s
    filtered_df = df[
        (df["Declared enrichment"] < 90) & 
        (df["real counting times"] > 1000)
    ].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    
    # Select relevant columns
    columns_to_keep = [
        'File name', 'Declared enrichment', 'Detector', 'Detector quanti', 
        'content', 'real counting times', 'live counting times'
    ]
    
    # Check that all columns exist
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    
    final_df = filtered_df[available_columns].copy()
    print(final_df.columns)

    # Variable normalization
    scaler_standard = preprocessing.StandardScaler()
    scaler_minmax = preprocessing.MinMaxScaler()
    
    # Normalize declared enrichment (StandardScaler)
    final_df.loc[:, 'Declared enrichment scaled'] = scaler_standard.fit_transform(
        final_df[['Declared enrichment']]
    ).flatten()
    
    # Normalize counting times (MinMaxScaler on log)
    final_df.loc[:, 'real counting times scaled'] = scaler_minmax.fit_transform(
        np.log1p(final_df[['real counting times']])
    ).flatten()
    
    final_df.loc[:, 'live counting times scaled'] = scaler_minmax.fit_transform(
        np.log1p(final_df[['live counting times']])
    ).flatten()
    
    return final_df

def create_histograms(df, save_path="./"):
    """
    Create histograms to visualize variable distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        save_path (str): Path where to save histograms
    """
    # Columns to exclude from histograms
    exclude_columns = ['File name', 'content']
    
    for column in df.columns:
        if column not in exclude_columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df[column], bins=30, edgecolor="black", alpha=0.7, color='skyblue')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{save_path}hist_{column.replace(' ', '_')}_reduced.png", dpi=300)
            plt.close()

def stratified_train_val_split(df, train_ratio=0.8, random_seed=4):
    """
    Perform stratified split based on detector type.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    random.seed(random_seed)
    
    train_dataset = pd.DataFrame(columns=df.columns.tolist())
    val_dataset = pd.DataFrame()
    
    # Get unique detector types
    detector_types = df['Detector quanti'].value_counts().index
    print("Detector type distribution:")
    print(df['Detector quanti'].value_counts())
    
    # Iterate through each detector type
    for detector_type in detector_types:
        detector_subset = df[df['Detector quanti'] == detector_type]
        print(f"Samples for detector type {detector_type}: {len(detector_subset)}")
        
        # Create shuffled indices for this subset
        indices = list(range(len(detector_subset)))
        random.shuffle(indices)
        
        # Split indices into train and validation
        split_point = int(np.floor(train_ratio * len(detector_subset)))
        print(f"Train samples: {split_point}, Val samples: {len(detector_subset) - split_point}")
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        # Add samples to respective datasets
        train_dataset = pd.concat([train_dataset, detector_subset.iloc[train_indices]], ignore_index=True)
        val_dataset = pd.concat([val_dataset, detector_subset.iloc[val_indices]], ignore_index=True)
    
    return train_dataset, val_dataset

def normalize_spectra(train_df, val_df, test_df, save_path="./"):
    """
    Normalize spectra by applying log(1+x) then normalization by max amplitude.
    Store global parameters efficiently in a separate metadata file.
    
    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames of different sets
        save_path (str): Path to save metadata file
        
    Returns:
        tuple: DataFrames with normalized spectra and normalization config
    """

    # Extract spectra
    train_spectra = np.array([spectrum for spectrum in train_df["content"].values])
    val_spectra = np.array([spectrum for spectrum in val_df["content"].values])
    test_spectra = np.array([spectrum for spectrum in test_df["content"].values])
    
    # Apply log(1+x)
    train_log_spectra = np.log1p(train_spectra)
    val_log_spectra = np.log1p(val_spectra)
    test_log_spectra = np.log1p(test_spectra)
    
    # Calculate maximum amplitudes
    train_max_amp = np.max(train_log_spectra, axis=1)[:, np.newaxis]
    val_max_amp = np.max(val_log_spectra, axis=1)[:, np.newaxis]
    test_max_amp = np.max(test_log_spectra, axis=1)[:, np.newaxis]
    
    # Calculate global bounds for normalization
    global_max = max(np.max(train_max_amp), np.max(val_max_amp), np.max(test_max_amp))
    global_min = min(np.min(train_max_amp), np.min(val_max_amp), np.min(test_max_amp))
    
    # Normalize spectra by their maximum amplitude
    train_normalized = train_log_spectra / train_max_amp
    val_normalized = val_log_spectra / val_max_amp
    test_normalized = test_log_spectra / test_max_amp
    
    # Min-max normalization of maximum amplitudes
    train_max_scaled = (train_max_amp - global_min) / (global_max - global_min)
    val_max_scaled = (val_max_amp - global_min) / (global_max - global_min)
    test_max_scaled = (test_max_amp - global_min) / (global_max - global_min)
    
    # Create normalization metadata
    normalization_metadata = {
        "global_min": float(global_min),
        "global_max": float(global_max),
        "energy": np.arange(0, 4096).tolist(),  # 4096 channels for this data format
        "normalization_method": "log1p_then_max_amplitude_scaling",
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df) + len(val_df) + len(test_df)
        },
        "processing_steps": [
            "1. Apply log(1+x) transformation to spectra",
            "2. Calculate max amplitude per spectrum",
            "3. Normalize each spectrum by its max amplitude", 
            "4. Scale max amplitudes using global min-max normalization"
        ]
    }
    
    # Save metadata to file
    metadata_path = f"{save_path}/normalization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(normalization_metadata, f, indent=2)
    print(f"Normalization metadata saved to: {metadata_path}")
    
    # Update DataFrames (WITHOUT storing redundant global values)
    datasets = [
        (train_df.copy(), train_normalized, train_max_amp, train_max_scaled),
        (val_df.copy(), val_normalized, val_max_amp, val_max_scaled),
        (test_df.copy(), test_normalized, test_max_amp, test_max_scaled)
    ]
    
    normalized_datasets = []
    for df, normalized_spectra, max_amp, max_scaled in datasets:
        df["content"] = [spectrum.tolist() for spectrum in normalized_spectra]
        df["max_amplitude"] = max_amp.flatten()  # Individual max amplitude
        df["max_amplitude_scaled"] = max_scaled.flatten()  # Scaled individual max
        # DO NOT store global_min/global_max in each row - use metadata file instead
        
        # Optional: Store reference to metadata file in DataFrame attributes
        df.attrs['normalization_metadata_file'] = metadata_path
        df.attrs['global_min'] = float(global_min)
        df.attrs['global_max'] = float(global_max)
        
        normalized_datasets.append(df)
    
    return tuple(normalized_datasets), normalization_metadata

def save_datasets(datasets, filenames):
    """
    Save datasets in JSON format.
    
    Args:
        datasets (tuple): Tuple containing DataFrames to save
        filenames (list): List of filenames
    """
    for dataset, filename in zip(datasets, filenames):
        dataset.to_json(filename)
        print(dataset.columns)
        print(f"Dataset saved: {filename}")

def main():
    """
    Main function orchestrating the entire processing workflow.
    """
    # Configuration
    base_path = os.getcwd()
    print(f"Working directory: {base_path}")
    
    # =============================================================================
    # 1. DATA LOADING AND INITIAL PROCESSING
    # =============================================================================
    
    print("\n=== Loading spectra ===")
    df_raw = load_and_process_spectra(base_path)
    
    # 2. Data cleaning and conversion
    print("\n=== Data cleaning ===")
    df_clean = clean_and_convert_data(df_raw)
    
    # 3. Save raw data
    df_clean.to_json("dataset_before_normalization.json")
    
    # 4. Filtering and normalization
    print("\n=== Filtering and normalization ===")
    df_filtered = filter_and_normalize_data(df_clean)
    
    # 5. Save filtered data
    df_filtered.to_json("dataset_before_normalization_reduced.json")
    
    # 6. Create histograms
    print("\n=== Creating histograms ===")
    create_histograms(df_filtered)
    
    # =============================================================================
    # 2. DATASET SPLITTING
    # =============================================================================
    
    print("\nPerforming stratified train/validation split...")
    
    # Perform stratified split based on detector type
    train_dataset, val_dataset = stratified_train_val_split(df_filtered, train_ratio=0.8, random_seed=4)
    train_dataset.reset_index(inplace=True, drop=True)
    
    # Further split validation set into validation and test sets
    print("Splitting validation set into validation and test sets...")
    val_indices = np.arange(len(val_dataset))
    np.random.shuffle(val_indices)
    
    # Split validation set in half
    split_point = len(val_dataset) // 2
    val_ixs, test_ixs = val_indices[:split_point], val_indices[split_point:]
    
    test_dataset = val_dataset.iloc[test_ixs].reset_index(drop=True)
    val_dataset = val_dataset.iloc[val_ixs].reset_index(drop=True)
    
    # Save unnormalized datasets
    save_datasets(
        (train_dataset, val_dataset, test_dataset),
        ["train_before_normalization_reduced.json", 
         "val_before_normalization_reduced.json", 
         "test_before_normalization_reduced.json"]
    )
    
    # =============================================================================
    # 3. DATA NORMALIZATION
    # =============================================================================
    
    print("\nApplying normalization to spectral data...")
    
    # Apply normalization
    (train_norm, val_norm, test_norm), norm_metadata = normalize_spectra(
        train_dataset, val_dataset, test_dataset, save_path="./"
    )
    
    # =============================================================================
    # 4. FINAL DATASET PREPARATION
    # =============================================================================
    
    # Final save
    print("\n=== Final save ===")
    save_datasets(
        (train_norm, val_norm, test_norm),
        ["train.json", 
         "val.json", 
         "test.json"]
    )
    
    print(f"Global normalization bounds: min={norm_metadata['global_min']:.6f}, max={norm_metadata['global_max']:.6f}")
    print("\n=== Processing completed successfully ===")

if __name__ == "__main__":
    main()