# renesasadc

The `renesasadc` package provides tools for classifying and processing audio data, specifically designed for handling common urban sounds. It includes functionality for classifying audio into four categories: air conditioner, car horn, engine idling, and siren using SVM and RF (Random Forest) models. Additionally, it includes a utility to process audio data from a structured directory.

## Installation

To install the package, use the following pip command:

```bash
pip install git+https://github.com/tanaynayak/renesasadc.git@main
```

Ensure that you have Python 3.8 or later installed on your system.

## Usage

### Audio Classification

To classify an audio file into one of the four predefined categories, use the `classify_audio` function from the `audio_classifier` module. Here's an example:

```python
from renesasadc.audio_classifier import classify_audio

file_path = 'data/car_horn/185436-1-6-0.wav'
classification_results = classify_audio(file_path)
print(classification_results)
```

This function will return the classification results for the specified audio file, indicating the predictions made by both the SVM and RF models.

### Running Data Processing Task

To execute a comprehensive audio data processing task, use the `run_job` function from the `audio_data_processing_task` module. This function requires a `data` folder in the root directory of your project, structured with labeled subfolders for each class containing WAV audio files.

```python
from renesasadc.audio_data_processing_task import run_job

run_job()
```

This function will process all audio files within the structured directory, applying necessary preprocessing and data handling as per the internal logic of the package.

