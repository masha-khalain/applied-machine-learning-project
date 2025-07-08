import numpy as np
import pandas as pd
from scipy.signal import detrend, windows


def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with the fast fourier transformation.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequencies for each signal, shape: (num_samples,).
    """
    dominant_freqs = [] 
    for signal in x:
        # Applies detrending and windowing to prepare signals for frequency analysis
        signal = detrend(signal)
        windowed = signal * windows.hamming(len(signal))
        # Computes Fast Fourier Transform (FFT) to convert signals to frequency domain
        fft_vals = np.fft.rfft(windowed)
        # Calculates Power Spectral Density (PSD)
        psd = np.abs(fft_vals) ** 2
        # Identifies and returns the dominant frequency for each signal
        freqs = np.fft.rfftfreq(len(signal), d=1/fs)
        dominant_freq = freqs[np.argmax(psd)]
        dominant_freqs.append(dominant_freq)
    return np.array(dominant_freqs)
    pass


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extract 20 different features from the data.
    Args:
        data (np.ndarray): The data to extract features from.
        labels (np.ndarray): The labels of the data.
    Returns:
        pd.DataFrame: The extracted features.
    """
    features = []

    fs = 1000  

    for i in range(data.shape[0]):
        c = data[i, 0, :]
        v = data[i, 1, :]
        power = v * c

        # Determine ftt and psd of v and c
        v_fft = np.fft.rfft(v * np.hamming(len(v)))
        c_fft = np.fft.rfft(c * np.hamming(len(c)))
        v_psd = np.abs(v_fft) ** 2
        c_psd = np.abs(c_fft) ** 2
        freqs = np.fft.rfftfreq(len(v), d=1/fs)

        
        v_centroid = np.sum(freqs * v_psd) / np.sum(v_psd)
        c_centroid = np.sum(freqs * c_psd) / np.sum(c_psd)

        row = {
            'v_mean': np.mean(v),
            'v_std': np.std(v),
            'v_min': np.min(v),
            'v_max': np.max(v),
            'v_median': np.median(v),
            'v_rms': np.sqrt(np.mean(v**2)),

            'c_mean': np.mean(c),
            'c_std': np.std(c),
            'c_min': np.min(c),
            'c_max': np.max(c),
            'c_median': np.median(c),
            'c_rms': np.sqrt(np.mean(c**2)),

            'p_mean': np.mean(power),
            'p_std': np.std(power),
            'p_max': np.max(power),
            'p_rms': np.sqrt(np.mean(power**2)),
            'p_median': np.median(power),

            'v_dominant_freq': find_dominant_frequencies(np.array([v]), fs)[0],
            'c_dominant_freq': find_dominant_frequencies(np.array([c]), fs)[0],
            'vc_corr': np.corrcoef(v, c)[0, 1],

            'label': labels[i]
        }

        features.append(row)
    return pd.DataFrame(features)
    pass


