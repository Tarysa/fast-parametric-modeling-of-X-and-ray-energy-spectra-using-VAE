import numpy as np
import pandas as pd
import os
import random
import json

def load_and_process_spectra_sheet(filename, sheet_name, detector_type):
    """
    Load and process spectral data from a specific Excel sheet.
    
    Args:
        filename (str): Path to the Excel file
        sheet_name (str): Name of the sheet to read
        detector_type (str): Type of detector (e.g., 'hpge', 'labr')
        
    Returns:
        pd.DataFrame: Processed dataframe with spectral data and metadata
    """
    # Load Excel sheet with multi-level headers and index
    print(filename)
    df = pd.read_excel(filename, sheet_name, header=[0, 1], index_col=[0, 1], engine='openpyxl')
    
    # Clean up the dataframe structure
    df = df.reset_index(level=0, drop=True)  # Remove first level of index
    df = df.droplevel(1, axis=1)  # Remove second level of columns
    df = df.dropna(axis=0, how="all")  # Remove rows that are completely NaN
    df = df.dropna(axis=1, how="all")  # Remove columns that are completely NaN
    df = df.transpose()  # Transpose so samples are rows and energy channels are columns
    
    # Add detector type information
    df["type"] = [detector_type] * len(df)
    
    # Parse enrichment and attenuation from index names
    info = df.index.tolist()
    info = [sample_id.split(" ") for sample_id in info]
    
    # Extract enrichment percentages (remove '%' symbol and convert to decimal)
    enrichment = [float(elem[0][:-1]) / 100.0 for elem in info]
    
    # Extract attenuation values (handle 'Attenuation' header case)
    attenuation = [
        float(elem[3]) if elem[3] != "Attenuation" else 0.0 
        for elem in info
    ]
    
    df["enrichment"] = enrichment
    df["attenuation"] = attenuation
    
    return df


def stratified_train_val_split(df, train_ratio=0.8, random_seed=4):
    """
    Perform stratified split based on detector type and attenuation values.
    
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
    detector_types = df['type_quanti'].value_counts().index
    print("Detector type distribution:")
    print(df['type_quanti'].value_counts())
    
    # Iterate through each detector type
    for detector_type in detector_types:
        detector_subset = df[df['type_quanti'] == detector_type]
        print(f"Samples for detector type {detector_type}: {len(detector_subset)}")
        
        # Get unique attenuation values for this detector type
        attenuation_values = detector_subset['attenuation'].value_counts().index
        print("Attenuation distribution:")
        print(detector_subset['attenuation'].value_counts())
        
        # Iterate through each attenuation value
        for attenuation_val in attenuation_values:
            print(f"Processing attenuation: {attenuation_val}")
            
            # Get subset with specific detector type and attenuation
            subset = detector_subset[detector_subset['attenuation'] == attenuation_val]
            print(f"Samples in subset: {len(subset)}")
            
            # Create shuffled indices for this subset
            indices = list(range(len(subset)))
            random.shuffle(indices)
            
            # Split indices into train and validation
            split_point = int(np.floor(train_ratio * len(subset)))
            print(f"Train samples: {split_point}, Val samples: {len(subset) - split_point}")
            
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            
            # Add samples to respective datasets
            train_dataset = pd.concat([train_dataset, subset.iloc[train_indices]], ignore_index=True)
            val_dataset = pd.concat([val_dataset, subset.iloc[val_indices]], ignore_index=True)
    
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
    train_spectra = np.array([spectrum for spectrum in train_df[train_df.columns.tolist()[1:-4]].values])
    val_spectra = np.array([spectrum for spectrum in val_df[val_df.columns.tolist()[1:-4]].values])
    test_spectra = np.array([spectrum for spectrum in test_df[test_df.columns.tolist()[1:-4]].values])
    
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
    
    normalization_metadata = {
        "global_min": float(global_min),
        "global_max": float(global_max)
    }

    metadata_path = f"{save_path}/normalization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(normalization_metadata, f, indent=2)
    print(f"Normalization metadata saved to: {metadata_path}")
    

    # Update DataFrames with minimal metadata
    datasets = [
        (train_df.copy(), train_normalized, train_max_amp, train_max_scaled),
        (val_df.copy(), val_normalized, val_max_amp, val_max_scaled),
        (test_df.copy(), test_normalized, test_max_amp, test_max_scaled)
    ]

    normalized_datasets = []
    for df, normalized_spectra, max_amp, max_scaled in datasets:
        df["content"] = [spectrum.tolist() for spectrum in normalized_spectra]
        df["max_amplitude"] = max_amp.flatten()
        df["max_amplitude_scaled"] = max_scaled.flatten()
        
        normalized_datasets.append(df)
    
    return tuple(normalized_datasets)

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
    """Main preprocessing pipeline for spectral data."""
    
    # =============================================================================
    # 1. DATA LOADING AND INITIAL PROCESSING
    # =============================================================================
    
    # Get the first Excel file from the data directory
    data_dir = os.path.join(os.getcwd(), "data")
    filename = os.listdir(data_dir)[0]
    filepath = os.path.join("data", filename)
    
    print(f"Processing file: {filename}")
    
    # Load and process HPGe detector data
    print("Loading HPGe spectral data...")
    df_hpge = load_and_process_spectra_sheet(filepath, "HPGe_sim_spectra", "hpge")
    
    # Load and process LaBr detector data  
    print("Loading LaBr spectral data...")
    df_labr = load_and_process_spectra_sheet(filepath, "LaBr_sim_spectra", "labr")
    
    # Combine both detector datasets
    df = pd.concat([df_hpge, df_labr], axis=0, ignore_index=True)
    df = df.reset_index()
    
    # Create quantitative encoding for detector types
    df["type_quanti"] = pd.factorize(df["type"].values)[0]
        
    # Save initial processed dataset
    df.to_json("dataset before normalization.json")
    print("Saved raw dataset to 'dataset before normalization.json'")
    
    # =============================================================================
    # 2. DATASET SPLITTING
    # =============================================================================
    
    print("\nPerforming stratified train/validation split...")
    
    # Perform stratified split based on detector type and attenuation
    train_dataset, val_dataset = stratified_train_val_split(df, train_ratio=0.8, random_seed=4)
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
        ["train_before_normalization.json", 
         "val_before_normalization.json", 
         "test_before_normalization.json"]
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
    
    # 9. Efficient spectrum normalization
    print("\n=== Spectrum normalization ===")
    
    # 10. Final save (without redundant global values)
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