import numpy as np
import librosa
import textgrids
import os
import python_speech_features
from tqdm import tqdm

# Function for reading labels from .TextGrig file:
def readLabels(path, sample_rate):
        
    labeled_list  = []
    grid = textgrids.TextGrid(path)

    for interval in grid['silences']:
        if interval.text == "-" or interval.text == " ":
            label = 0
        else:
            label = 1

        dur = interval.dur
        dur_samples = int(np.round(dur * sample_rate)) # sec -> num of samples
        
        for i in range(dur_samples):
            labeled_list.append(label)

    return np.array(labeled_list)

def load_files(audio_path, audio_extension):
    """
    Recursively loads audio files from a specified directory.

    Args:
        audio_path (str): The root directory to search for audio files.
        audio_extension (str, optional): The audio file extension to filter 
                                        for (default is ".wav").

    Returns:
        list: A sorted list of full paths to the found audio files.
    Raises:
        FileNotFoundError: If the specified audio_path does not exist.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path '{audio_path}' not found.")

    audio_files = []

    for root, _, files in os.walk(audio_path):
        for file in files:
            if file.endswith(audio_extension):
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)

def max_signal_length(audio_files):
    """
    Determines the maximum signal length among a list of audio files.

    Args:
        audio_files (list): A list of paths to audio files.

    Returns:
        int: The maximum signal length found among the audio files.
    
    Raises:
        ValueError: If the input list is empty.
        IOError: If any audio file cannot be loaded.
    """
    
    if not audio_files:
        raise ValueError("Audio file list cannot be empty")
    
    max_length = 0
    for audio_file in audio_files:
        try:
            signal, _ = librosa.load(audio_file)
            max_length = max(max_length, len(signal))  
        except Exception as e:  # Catch potential loading errors
            raise IOError(f"Error loading audio file '{audio_file}': {e}")

    return max_length

def object_padding(object, length):
    """
    Pad the object to the given length

    Args:
        object (np.array): time series object
        max_length (int): Desired length to pad/truncate signals to.
    Returns:
        object (np.array): processed time series object
    """
    if len(object) < length:
        padding_length = length - len(object)
        # Pad at the end
        object = np.pad(object, (0, padding_length), mode="constant")
    else:
        object = object[:length]
    return np.array(object)

def fbank_features_extraction(audio_files, max_length, preemphasis_coef=0.97, window_length=0.025, window_step=0.01, window_function=np.hamming, num_nfft=551, num_features=40):
    """
    Extracts log Mel-filterbank (fbank) features from a list of audio files.

    Args:
        audio_files (list): List of paths to audio files.
        max_length (int): Desired length to pad/truncate signals to.
        preemphasis_coef (float): Pre-emphasis filter coefficient (default: 0.97).
        window_length (float): Length of the analysis window in seconds (default: 0.025).
        window_step (float): Step between successive windows in seconds (default: 0.01).
        window_function (callable): Window function to apply (default: np.hamming).
        nfft (int): Number of FFT points (default: 551).
        num_features (int): Number of Mel filters (default: 40).

    Returns:
        np.ndarray: 2D array of shape (num_files, num_frames, num_features + 1) 
                    where num_features + 1 represents the log energy feature.
    """
    # Filter Bank
    fbank_features = list()
    for i in tqdm(range(len(audio_files))):
        # Load the signal and sample rate
        signal, sample_rate = librosa.load(audio_files[i])
        # Audio padding
        signal = object_padding(signal, max_length)
        # Extract features
        # features_fbank (Mel-frequency cepstral coefficiens): captures the spectral information of the audio signal in a way that mimics human auditory perception
        # feature_energy: the overall energy of the audio signal within a specific frequency range
        features_fbank, feature_energy = python_speech_features.base.fbank(signal=signal,
                                                                        samplerate=sample_rate,
                                                                        winlen=window_length,
                                                                        winstep=window_step,
                                                                        nfilt=num_features,
                                                                        nfft=num_nfft,
                                                                        lowfreq=0,
                                                                        highfreq=None,
                                                                        preemph=preemphasis_coef,
                                                                        winfunc=window_function)
        # Log fbank and log energy
        features_logfbank = np.log(features_fbank)
        feature_logenergy = np.log(feature_energy)
        # Merge logfbank and log energy:
        features = np.hstack((feature_logenergy.reshape(feature_logenergy.shape[0], 1), features_logfbank))
        # Storing the fbank features for each audio
        fbank_features.append(features)
    # Return the features in numpy array format
    return np.array(fbank_features)

def supervised_features_extraction(audio_files, annotation_files, max_length, preemphasis_coef=0.97, window_length=0.025, window_step=0.01, window_function=np.hamming, num_nfft=551, num_features=40):
    """
    Extracts log Mel-filterbank (fbank) features from a list of audio files.

    Args:
        audio_files (list): List of paths to audio files.
        annotation_files (list): List of paths to annotation files
        max_length (int): Desired length to pad/truncate signals to.
        preemphasis_coef (float): Pre-emphasis filter coefficient (default: 0.97).
        window_length (float): Length of the analysis window in seconds (default: 0.025).
        window_step (float): Step between successive windows in seconds (default: 0.01).
        window_function (callable): Window function to apply (default: np.hamming).
        nfft (int): Number of FFT points (default: 551).
        num_features (int): Number of Mel filters (default: 40).

    Returns:
        np.ndarray: 2D array of shape (num_files, num_frames, num_features + 1) 
                    where num_features + 1 represents the log energy feature.
    """
    # Filter Bank
    fbank_features = list()
    labels = list()
    for i in tqdm(range(len(audio_files))):
        # Load the signal and sample rate
        signal, sample_rate = librosa.load(audio_files[i])
        signal = object_padding(signal, max_length)
        truth_labels = readLabels(path=annotation_files[i], sample_rate=sample_rate)
        truth_labels = object_padding(truth_labels, max_length)
        # Extract features
        # features_fbank (Mel-frequency cepstral coefficiens): captures the spectral information of the audio signal in a way that mimics human auditory perception
        # feature_energy: the overall energy of the audio signal within a specific frequency range
        features_fbank, feature_energy = python_speech_features.base.fbank(signal=signal,
                                                                        samplerate=sample_rate,
                                                                        winlen=window_length,
                                                                        winstep=window_step,
                                                                        nfilt=num_features,
                                                                        nfft=num_nfft,
                                                                        lowfreq=0,
                                                                        highfreq=None,
                                                                        preemph=preemphasis_coef,
                                                                        winfunc=window_function)
        # Log fbank and log energy
        features_logfbank = np.log(features_fbank)
        feature_logenergy = np.log(feature_energy)
        # Merge logfbank and log energy:
        features = np.hstack((feature_logenergy.reshape(feature_logenergy.shape[0], 1), features_logfbank))
        
        # Reshape labels for each group of features:
        temp_label = python_speech_features.sigproc.framesig(sig=truth_labels, 
                                                                    frame_len=window_length * sample_rate, 
                                                                    frame_step=window_step * sample_rate, 
                                                                    winfunc=np.ones)
        label = np.zeros(temp_label.shape[0])
        label = np.array([1 if np.sum(temp_label[j], axis=0) > temp_label.shape[0] / 2 else 0 for j in range(temp_label.shape[0])])
        # Storing the fbank features and label for each audio
        fbank_features.append(features)
        labels.append(label)
    # Return the features in numpy array format
    return np.array(fbank_features), np.array(labels)
