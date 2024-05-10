import os
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
from joblib import dump


meta_data = pd.read_csv('data/meta.csv')

def plot_audio_files(row):
    filepath = os.path.join('data', row['label'], row['file'])
    data, sr = librosa.load(filepath, sr=None)  # Load with original sample rate

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(data, sr=sr)
    plt.title('Waveform of {}'.format(row['file']))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectrogram of {}'.format(row['file']))
    plt.colorbar()
    plt.show()

def load_and_preprocess(file_path, duration=4, sr=22050):
    data, sr = librosa.load(file_path, sr=sr, duration=duration)
    
    pre_emphasis = 0.97
    emphasized_data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
    
    return emphasized_data, sr

def extract_features(file_path, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13, n_chroma=12, n_mels=40):
    data, sr = load_and_preprocess(file_path, sr=sr)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=data, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data, hop_length=hop_length)

    # Mel-scaled Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Aggregating all features
    features = np.hstack((
        np.mean(mfcc, axis=1), 
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid, axis=1), 
        np.mean(zcr, axis=1), 
        np.mean(mel_spectrogram_db, axis=1)
    ))

    return features

# for label in meta_data['label'].unique():
#     example_row = meta_data[meta_data['label'] == label].iloc[0]
#     plot_audio_files(example_row)






# sample_data, sample_rate = load_and_preprocess(train.iloc[0]['path'])
# print("Processed data shape:", sample_data.shape)
# print("Sample rate:", sample_rate)





if __name__ == "__main__":
    meta_data['path'] = meta_data.apply(lambda row: os.path.join('data', row['label'], row['file']), axis=1)

    train, test = train_test_split(meta_data, test_size=0.2, random_state=42, stratify=meta_data['label'])

    print("Training Set Size:", len(train))
    print("Testing Set Size:", len(test))
    train_features = train['path'].apply(lambda x: extract_features(x))

    X_train = np.array(list(train_features))
    y_train = np.array(train['label'])

    print("Feature matrix shape:", X_train.shape)
    print("Labels array shape:", y_train.shape)


    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded)

    #Linear SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    #Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    dump(svm_model, 'svm_model.joblib')
    dump(rf_model, 'rf_model.joblib')
    dump(label_encoder, 'label_encoder.joblib')

    y_pred_svm = svm_model.predict(X_val)
    y_pred_rf = rf_model.predict(X_val)

    print("SVM Classification Report:")
    print(classification_report(y_val, y_pred_svm, target_names=label_encoder.classes_))

    print("Random Forest Classification Report:")
    print(classification_report(y_val, y_pred_rf, target_names=label_encoder.classes_))

    # Confusion Matrices
    cm_svm = confusion_matrix(y_val, y_pred_svm)
    cm_rf = confusion_matrix(y_val, y_pred_rf)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_svm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("SVM Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_rf, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Random Forest Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    
    test_features = [extract_features(path) for path in test['path']]

    X_test = np.array(test_features)
    y_test = np.array(test['label'])

    y_test_encoded = label_encoder.transform(y_test)


    
    y_pred_svm = svm_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    print("SVM Test Classification Report:")
    print(classification_report(y_test_encoded, y_pred_svm, target_names=label_encoder.classes_))

    print("Random Forest Test Classification Report:")
    print(classification_report(y_test_encoded, y_pred_rf, target_names=label_encoder.classes_))

    cm_svm = confusion_matrix(y_test_encoded, y_pred_svm)
    cm_rf = confusion_matrix(y_test_encoded, y_pred_rf)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_svm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("SVM Test Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_rf, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Random Forest Test Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

