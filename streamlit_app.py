import os
import tempfile

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from joblib import load
import streamlit as st
from tensorflow.keras.models import load_model

# =========================================================
# 0. PAGE CONFIG  (must be first Streamlit call)
# =========================================================
st.set_page_config(
    page_title="Emergency Audio Classifier",
    page_icon="🚑",
    layout="centered",
)

# =========================================================
# 1. CONFIG / LOAD ARTIFACTS
# =========================================================
ARTIFACT_DIR = "artifacts"          # same as training script
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_audio_model.h5")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.joblib")

SAMPLE_RATE = 22050
N_MFCC = 40   # must match training script


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = load(SCALER_PATH)
    label_encoder = load(ENCODER_PATH)
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()
classes = list(label_encoder.classes_)

# =========================================================
# 2. FEATURE EXTRACTION (same as training!)
#    MFCC + delta + delta2, mean + std
# =========================================================
def extract_features_from_data(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # 1st and 2nd order deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(x):
        # x shape: (n_mfcc, time) -> concat mean + std along time
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

    feat = np.concatenate([
        stats(mfcc),
        stats(delta),
        stats(delta2),
    ])  # dimension should be 240
    return feat.astype("float32")


def predict_audio(file_bytes, file_ext=".wav"):
    # keep real extension so librosa can decode mp3 or wav
    if not file_ext.startswith("."):
        file_ext = "." + file_ext

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # load audio
    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    os.remove(temp_path)

    # features
    feat_vec = extract_features_from_data(y, sr=sr)
    feat_vec = feat_vec.reshape(1, -1)
    feat_vec = scaler.transform(feat_vec)

    # predict
    probs = model.predict(feat_vec)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    return pred_class, probs, y, sr


# =========================================================
# 3. STREAMLIT UI
# =========================================================
st.title("🚨 Emergency / Traffic Audio Classifier")
st.write(
    "Upload an audio file (`.wav` or `.mp3`) containing sounds like "
    "ambulance, firetruck, traffic etc."
)

uploaded_file = st.file_uploader(
    "Upload an audio file (.wav or .mp3)",
    type=["wav", "mp3"],
)

if uploaded_file is not None:
    # playback
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Audio"):
        with st.spinner("Analyzing..."):
            file_bytes = uploaded_file.read()
            _, ext = os.path.splitext(uploaded_file.name)
            pred_class, probs, audio, sr = predict_audio(file_bytes, file_ext=ext)

        # ------- Prediction -------
        st.success(f"Predicted class: **{pred_class}**")

        # ------- Probabilities -------
        st.subheader("Class probabilities")
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        st.bar_chart(prob_dict)

        # ------- Waveform -------
        st.subheader("Waveform")
        fig_wav, ax = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig_wav.tight_layout()
        st.pyplot(fig_wav)

        # ------- MFCC image -------
        st.subheader("MFCCs")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        fig_mfcc, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
        ax.set_title("MFCC")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        fig_mfcc.tight_layout()
        st.pyplot(fig_mfcc)

else:
    st.info("👆 Upload a `.wav` or `.mp3` file to get started.")

st.markdown("---")

