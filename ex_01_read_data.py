import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.

    Args:
        data_path (Path): Path to the CSV data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with unlabeled data removed.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
        ValueError: If the data is empty after removing unlabeled data and dropping NaN values.
    """
    try: 
        df = pd.read_csv(data_path)
    except: 
        raise FileNotFoundError
    df = remove_unlabeled_data(df)
    df.dropna(inplace = True)
    df = df.apply(pd.to_numeric, errors='coerce')
    if(df.empty == True or df.isnull().values.any()):
        raise ValueError
    return df

def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).

    Args:
        data (pd.DataFrame): Input DataFrame containing a 'labels' column.

    Returns:
        pd.DataFrame: DataFrame with unlabeled data removed.
    """

    data.drop(data[data['labels'] == -1].index, inplace = True)
    return data


def jls_extract_def():
    
    return 


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert DataFrame to numpy arrays, separating labels, experiment IDs, and features.

    Args:
        data (pd.DataFrame): Input DataFrame containing 'labels', 'exp_ids', and feature columns.

    Returns:
        tuple: A tuple containing:
            - labels (np.ndarray): Array of labels
            - exp_ids (np.ndarray): Array of experiment IDs
            - data (np.ndarray): Combined array of current and voltage features
    """
    # separate labels and experiment id
    labels = data['labels'].to_numpy()
    exp_ids = data['exp_ids'].to_numpy()
    
    #get current and voltage columns
    current_col = [col for col in data.columns if col.startswith('I')]
    voltage_col = [col for col in data.columns if col.startswith('V')]
    current_data = data[current_col].to_numpy()
    voltage_data = data[voltage_col].to_numpy()

    final = np.stack([current_data, voltage_data], axis = -1)

    return labels, exp_ids, final


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the first dimension of a 3D array.
    
    Args:
        data (np.ndarray): Input array of shape (n_samples, timesteps, features)
        sequence_length (int): Length of each window
    
    Returns:
        np.ndarray: Windowed data of shape (n_windows, sequence_length*timesteps, features)
    """
    
    n_samples, timesteps, features = data.shape
    window_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=(sequence_length,timesteps, features), axis=[0,1,2])

    n_windows, z, y, seq_len, timesteps, features = window_view.shape

    # Reshape from (n_windows, sequence_length, timesteps, features) -> (n_windows, sequence_length * timesteps, features)
    reshaped = window_view.reshape(z*n_windows, seq_len * timesteps, y*features).copy()
    
    return reshaped

def get_welding_data(path: Path, n_samples: int | None = None, return_sequences: bool = False, sequence_length: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load welding data from CSV or cached numpy files.

    If numpy cache files don't exist, loads from CSV and creates cache files.
    If cache files exist, loads directly from them.

    Args:
        path (Path): Path to the CSV data file.
        n_samples (int | None): Number of samples to sample from the data. If None, all data is returned.
        return_sequences (bool): If True, return sequences of length sequence_length.
        sequence_length (int): Length of sequences to return.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of welding data features
            - np.ndarray: Array of labels
            - np.ndarray: Array of experiment IDs
    """
    try:
        labels = np.load(path/"labels.npy")
        exp_ids = np.load(path/"exp_ids.npy")
        data = np.load(path/"data.npy")
    except:
        labels, exp_ids, data = convert_to_np(load_data(path))
        np.save("labels", labels)
        np.save("exp_ids", exp_ids)
        np.save("data", data)
    if type(n_samples) == int:
        indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
        data = data[indices]
        labels = labels[indices]
        exp_ids = exp_ids[indices]
    if return_sequences == True:
        data = create_sliding_windows_first_dim(data, sequence_length)
        # Using sliding_window_view
        labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=sequence_length)
        exp_ids = np.lib.stride_tricks.sliding_window_view(exp_ids, window_shape=sequence_length)
    

    return data, labels, exp_ids
    pass