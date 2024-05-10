import numpy as np
import librosa
from audio_data_processing_task import load_and_preprocess  # Make sure this import path is correct

def test_load_and_preprocess():
    # Path to a known audio file
    file_path = 'data/air_conditioner/177726-0-0-11.wav'
    expected_shape = (88200,)  # Expected data shape
    expected_sr = 22050       # Expected sample rate

    # Execute the function
    sample_data, sample_rate = load_and_preprocess(file_path)

    # Assert conditions
    assert sample_data.shape == expected_shape, "The shape of the processed data is incorrect."
    assert sample_rate == expected_sr, "The sample rate returned is incorrect."
