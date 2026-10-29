import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "../dataset/ravdess"


def extract_features(path):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


X, y = [], []

# Walk through all subfolders
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            features = extract_features(file_path)

            # RAVDESS emotion code: 03 = happy, 04 = sad
            emotion_code = file.split("-")[2]
            label = 1 if emotion_code == "04" else 0

            X.append(features)
            y.append(label)

print("Total samples:", len(X))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "depression_model.pkl")
print("Model saved successfully")
