import numpy as np
import librosa
from joblib import load

# Load models and label encoder
svm_model = load('svm_model.joblib')
rf_model = load('rf_model.joblib')
label_encoder = load('label_encoder.joblib')

def extract_features(file_path, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_chroma=12, n_mels=40):
    data, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    chroma = librosa.feature.chroma_stft(y=data, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=data, hop_length=hop_length)
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    features = np.hstack((np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(spectral_centroid, axis=1), np.mean(zcr, axis=1), np.mean(mel_spectrogram_db, axis=1)))
    return features.reshape(1, -1) 

def classify_audio(file_path):
    features = extract_features(file_path)
    svm_prediction = svm_model.predict(features)
    rf_prediction = rf_model.predict(features)
    svm_class = label_encoder.inverse_transform(svm_prediction)[0]
    rf_class = label_encoder.inverse_transform(rf_prediction)[0]

    return {
        "svm_prediction": svm_class,
        "rf_prediction": rf_class
    }

