import pandas as pd
import numpy as np
import torch
import json
from torch.utils import data


class Dataset_dataset1(data.Dataset):
    """
    PyTorch Dataset class for handling spectral data with conditions.
    
    This dataset handles spectral content with associated conditions including type,
    enrichment, and attenuation parameters.
    """
    
    def __init__(self, json_file, config, before_norm=False, not_training=False):
        """
        Initialize the dataset.
        
        Args:
            json_file (str): Path to JSON file containing the data
            config: Configuration object with dataset parameters
            train_max (bool): Whether to include normalization parameters in training
            before_norm (bool): Whether data is before normalization
        """
        # Load data from JSON file
        df = pd.read_json(json_file)

        # Store configuration parameters
        self.not_training = not_training
        with open("../datasets/MC/normalization_metadata.json") as json_file:
            self.meta_data = json.load(json_file)
        self.min_amp_dataset = self.meta_data["global_min"]
        self.max_amp_dataset = self.meta_data["global_max"]
             
        # Process content column if it doesn't exist
        if "content" not in df.columns:
                # Extract content from columns (excluding last 4 columns) 
                df["content"] = [
                    elem.tolist() for elem in np.array([
                        elem for elem in df[df.columns.tolist()[1:-4]].values
                    ])]
        
        # Extract energy values from column names (excluding first and last 9 columns)
        self.energy = [float(elem) for elem in df.columns.tolist()[1:-9]]
        
        # Set main data arrays
        self.data = df["content"]
        self.type = df['type_quanti']
        self.enrichment = df['enrichment']
        self.attenuation = df["attenuation"]

        if not(before_norm):
            self.norm_param = df["max_amplitude_scaled"]
        
        # Define condition parameters
        self.condition_name = ["type", "enrichment", "attenuation"]
        self.condition_type = ['quali', 'quanti', 'quanti']  # qualitative, quantitative types
        self.condition = df[self.condition_name]

        # Create mapping between quantitative and qualitative detector names
        self.detector_dict = dict(zip(df["type_quanti"], df["type"]))
        self.detector_name = [self.detector_dict[key] for key in self.detector_dict.keys()]
        
        # Store configuration and derived parameters
        self.config = config
        self.signal_length = len(self.data.iloc[0])
        self.inputs_class = 3  # Number of condition classes
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (x, y) or (x, y, norm_param) depending on not_training setting
                x: Spectral data as FloatTensor
                y: Conditions [type, enrichment, attenuation] as FloatTensor
                norm_param: Normalization parameter (if not_training=True)
        """
        # Get spectral data and conditions
        x = self.data.iloc[idx]
        y = np.array([
            self.type.iloc[idx], 
            self.enrichment.iloc[idx], 
            self.attenuation.iloc[idx]
        ])
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        if self.not_training:
            norm_param = self.norm_param.iloc[idx]
            inputs = x, y, norm_param
        else:
            inputs = x, y
        return inputs
    
    def undo_normalization(self, x, norm_param):
        """
        Reverse the normalization applied to the data.
        
        Args:
            x: Normalized data
            norm_param: Normalization parameter used
            
        Returns:
            Denormalized data in original scale
        """
        # Scale normalization parameter if using scaled max
        norm_param = norm_param * (self.max_amp_dataset - self.min_amp_dataset) + self.min_amp_dataset
        
        # Reverse normalization: multiply by norm_param then reverse log transformation
        denormalize_x = x * norm_param
        denormalize_x = np.exp(denormalize_x) - 1
        
        return denormalize_x


class Dataset_dataset3(data.Dataset):
    """
    PyTorch Dataset class for handling detector-based spectral data.
    
    This dataset handles spectral data with detector information, enrichment levels,
    and counting times.
    """
    
    def __init__(self, json_file, config, before_norm = False, not_training=False):
        """
        Initialize the dataset.
        
        Args:
            json_file (str): Path to JSON file containing the data
            config: Configuration object with dataset parameters  
            train_max (bool): Whether to include normalization parameters
        """
        print(f"Loading data from: {json_file}")
        
        # Load data from JSON file
        df = pd.read_json(json_file)
        print(f"Dataset columns: {df.columns}")
        
        # Set main data and configuration
        self.data = df["content"]
        self.signal_length = len(self.data.iloc[0])
        self.config = config
        self.not_training = not_training

        with open("../datasets/ESARDA/dataset U/normalization_metadata.json") as json_file:
            self.meta_data = json.load(json_file)
        self.min_amp_dataset = self.meta_data["global_min"]
        self.max_amp_dataset = self.meta_data["global_max"]
                
        if not(before_norm):
            self.norm_param = df["max_amplitude_scaled"]

        # Set up conditions and normalization based on training mode
        if not_training:            
            # Define additional condition columns for non-training mode
            self.other_condition_name = [
                'File name', 
                'Detector', 
                'Declared enrichment', 
                'real counting times'
            ]
            self.other_condition = df[self.other_condition_name]
            
            # Main condition columns (quantitative versions)
            self.condition_name = [
                'Detector quanti', 
                'Declared enrichment scaled', 
                'real counting times scaled'
            ]
            
            # Create mapping between quantitative and qualitative detector names
            self.detector_dict = dict(zip(df["Detector quanti"], df["Detector"]))
            self.detector_name = [self.detector_dict[key] for key in self.detector_dict.keys()]
        else:
            #  For training, only use quantitative conditions
            self.condition_name = [
                'Detector quanti', 
                'Declared enrichment scaled', 
                'real counting times scaled'
            ]
        
        # Set condition data and metadata
        self.condition = df[self.condition_name]
        self.condition_type = ['quali', 'quanti', 'quanti']  # qualitative, quantitative types
        self.inputs_class = len(self.condition_name)
        
        # Generate energy scale (0.075 keV per channel)
        self.energy = [float(i * 0.075) for i in range(len(self.data.iloc[0]))]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (x, y) or (x, y, norm_param) depending on train_max setting
                x: Spectral data as FloatTensor
                y: Conditions as FloatTensor  
                norm_param: Normalization parameter (if train_max=True)
        """
        # Get spectral data and conditions
        x = np.array(self.data.iloc[idx])
        y = np.array(self.condition.iloc[idx])
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
                
        if self.not_training:
            norm_param = self.norm_param.iloc[idx]
            inputs = x, y, norm_param
        else:
            inputs = x, y
        return inputs
    
    def undo_normalization(self, x, norm_param):
        """
        Reverse the normalization applied to the data.
        
        Args:
            x: Normalized data
            norm_param: Normalization parameter used
            
        Returns:
            Denormalized data in original scale
        """
        # Scale normalization parameter if using scaled max
        norm_param = norm_param * (self.max_amp_dataset - self.min_amp_dataset) + self.min_amp_dataset
        
        # Reverse normalization: multiply by norm_param then reverse log transformation  
        denormalize_x = x * norm_param
        denormalize_x = np.exp(denormalize_x) - 1
        
        return denormalize_x