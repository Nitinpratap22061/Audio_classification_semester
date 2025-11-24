import os
import tempfile
from datetime import datetime

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from joblib import load
import streamlit as st
from tensorflow.keras.models import load_model
from twilio.rest import Client

# =========================================================
# 0. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emergency Audio Classifier",
    page_icon="🚑",
    layout="centered",
)

# =========================================================
# 1. CONFIG / PATHS
# =========================================================

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_audio_model.h5")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "label_encoder.joblib")

SAMPLE_RATE = 22050
N_MFCC = 40

# =========================================================
# TWILIO CONFIG  (REPLACE OR USE ENV VARIABLES)
# =========================================================

TWILIO_SID = "ACc4218ec549fb01f032bd6ee0ca75e3da"
TWILIO_AUTH = "6b70606f0425a6e540bd0fec0b80f87d"
TWILIO_FROM = "+12765944583"
TWILIO_TO = "+916205815679"


def send_sms(pred_class):
    """Send SMS alert via Twilio."""
    client = Client(TWILIO_SID, TWILIO_AUTH)
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message_body = (
        f"🚨 Detection Alert!\n"
        f"Type: {pred_class}\n"
        f"Time: {detection_time}"
    )

    message = client.messages.create(
        body=message_body,
        from_=TWILIO_FROM,
        to=TWILIO_TO
    )
    return message.sid


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = load(SCALER_PATH)
    label_encoder = load(ENCODER_PATH)
    return model, scaler, label_encoder


model, scaler, label_encoder = load_artifacts()
classes = list(label_encoder.classes_)

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features_from_data(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(x):
        return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

    feat = np.concatenate([
        stats(mfcc),
        stats(delta),
        stats(delta2),
    ])
    return feat.astype("float32")


def predict_audio(file_bytes, file_ext=".wav"):
    if not file_ext.startswith("."):
        file_ext = "." + file_ext

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
    os.remove(temp_path)

    feat_vec = extract_features_from_data(y, sr=sr)
    feat_vec = feat_vec.reshape(1, -1)
    feat_vec = scaler.transform(feat_vec)

    probs = model.predict(feat_vec)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    return pred_class, probs, y, sr


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🚨 Emergency / Traffic Audio Classifier")
st.write("Upload a `.wav` or `.mp3` file containing emergency or traffic sounds.")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔍 Analyze Audio"):
        with st.spinner("Analyzing..."):
            file_bytes = uploaded_file.read()
            _, ext = os.path.splitext(uploaded_file.name)

            pred_class, probs, audio, sr = predict_audio(
                file_bytes, file_ext=ext
            )

        # Prediction result
        st.success(f"Predicted class: **{pred_class}**")

        # Send SMS
        try:
            sms_id = send_sms(pred_class)
            st.info("📨 SMS alert sent successfully!")
        except Exception as e:
            st.error(f"⚠ SMS sending failed: {str(e)}")

        # Probability chart
        st.subheader("Class probabilities")
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        st.bar_chart(prob_dict)

        # Waveform plot
        st.subheader("Waveform")
        fig_wav, ax = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig_wav.tight_layout()
        st.pyplot(fig_wav)

        # MFCC visualization
        st.subheader("MFCCs")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        fig_mfcc, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
        ax.set_title("MFCC")
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        fig_mfcc.tight_layout()
        st.pyplot(fig_mfcc)

else:
    st.info("👆 Upload a `.wav` or `.mp3` file to begin.")

st.markdown("---")

is in this code can we do in message location of detection also send 
