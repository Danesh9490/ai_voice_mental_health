import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

from mental_state import analyze_mental_state

# ================= LOAD MODELS =================
emotion_model = load_model("emotion_cnn.h5")
depression_model = joblib.load("depression_model.pkl")

# ================= SETTINGS =================
audio_path = "test.wav"
sample_rate = 22050


# ================= FEATURE EXTRACTION =================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=sample_rate)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel])


# ================= EMOTION LABELS =================
emotion_labels = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise"
]


# ================= PREDICTION =================
features = extract_features(audio_path)

# Emotion CNN expects 2D input
emotion_input = features.reshape(1, -1)

emotion_prediction = emotion_model.predict(emotion_input)
emotion_index = np.argmax(emotion_prediction)
emotion = emotion_labels[emotion_index]

# Depression model
depression_prediction = depression_model.predict([features])[0]

# ================= MENTAL STATE ANALYSIS =================
mental_state = analyze_mental_state(emotion, depression_prediction)


# ================= OUTPUT =================
print("\n====== AI Mental Health Assessment ======")
print("Detected Emotion:", emotion)
print("Depression Prediction:", depression_prediction)
print("Mental Health State:", mental_state)
print("=========================================")