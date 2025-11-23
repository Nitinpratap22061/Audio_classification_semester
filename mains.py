import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from joblib import dump  # for saving scaler & label encoder

# =========================================================
# 1. CONFIG
# =========================================================

DATA_DIR = r"C:\Users\Nitin Kumar\Desktop\audio_classification_project\UrbanSound8k"

SAMPLE_RATE = 22050
N_MFCC = 40
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 120

OUTPUT_DIR = "artifacts"   # everything (plots, reports, models) saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. LIST CATEGORIES
# =========================================================
categories = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
]

print("Classes found:", categories)

if len(categories) == 0:
    raise RuntimeError("No class folders found in DATA_DIR. Check the path.")

# =========================================================
# 3. FEATURE EXTRACTION FUNCTION
#      -> MFCC mean + std + Δ + ΔΔ
# =========================================================
def extract_mfcc(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Load an audio file, compute MFCCs + deltas, and return a rich feature vector.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)  # mono
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 1st and 2nd order deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # statistics: mean + std for each
        def stats(x):
            return np.concatenate([np.mean(x, axis=1), np.std(x, axis=1)])

        feat = np.concatenate([
            stats(mfcc),
            stats(delta),
            stats(delta2),
        ])  # total dims = n_mfcc*2*3 = 240

        return feat.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Error processing {file_path}: {e}")
        return None

# =========================================================
# 4. BUILD DATASET
# =========================================================
features = []
labels = []

print("\nExtracting MFCC features...")

for label in categories:
    class_folder = os.path.join(DATA_DIR, label)
    files = os.listdir(class_folder)

    for fname in tqdm(files, desc=f"Processing {label}"):
        if not fname.lower().endswith(".wav"):
            continue  # ignore .png or anything else

        fpath = os.path.join(class_folder, fname)
        feat_vec = extract_mfcc(fpath)

        if feat_vec is not None:
            features.append(feat_vec)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

print("\nFeature matrix shape:", X.shape)
print("Labels shape:", y.shape)

if X.shape[0] == 0:
    raise RuntimeError("No features extracted. Check that .wav files exist.")

# show class balance
unique, counts = np.unique(y, return_counts=True)
print("\nSamples per class:")
for cls, c in zip(unique, counts):
    print(f"  {cls}: {c}")

# =========================================================
# 5. ENCODE LABELS & TRAIN/TEST SPLIT
# =========================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)
print("\nEncoded classes:", list(label_encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# =========================================================
# 6. SCALE FEATURES
# =========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
print("Saved scaler and label encoder to", OUTPUT_DIR)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# =========================================================
# 7. CLASS WEIGHTS (fix majority-class bias)
# =========================================================
class_weights_vec = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights_vec)}
print("\nClass weights:", class_weights)

# =========================================================
# 8. BUILD ANN MODEL
# =========================================================
input_dim = X_train.shape[1]  # should be 240

model = Sequential([
    Dense(512, activation="relu", input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0008),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================================
# 9. CALLBACKS
# =========================================================
best_model_path = os.path.join(OUTPUT_DIR, "best_audio_model.h5")

checkpoint = ModelCheckpoint(
    best_model_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=18,
    restore_best_weights=True,
    verbose=1
)

# =========================================================
# 10. TRAIN
# =========================================================
history = model.fit(
    X_train,
    y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_cat),
    callbacks=[checkpoint, early_stop],
    class_weight=class_weights,
    verbose=1,
    shuffle=True,
)

np.save(os.path.join(OUTPUT_DIR, "history.npy"), history.history)
print("Saved training history.")

# =========================================================
# 11. EVALUATE & SAVE METRICS
# =========================================================
print("\nLoading best saved model...")
best_model = load_model(best_model_path)

test_loss, test_acc = best_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")

y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cls_report = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_
)
print("\nClassification Report:")
print(cls_report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(cls_report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"),
           cm, delimiter=",", fmt="%d")

fig_cm, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig_cm.tight_layout()
fig_cm.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close(fig_cm)

# =========================================================
# 12. PLOT TRAINING CURVES & SAVE
# =========================================================
fig_acc, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["accuracy"], label="Train Accuracy")
ax.plot(history.history["val_accuracy"], label="Val Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
ax.grid(True)
fig_acc.tight_layout()
fig_acc.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close(fig_acc)

fig_loss, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"], label="Train Loss")
ax.plot(history.history["val_loss"], label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
ax.grid(True)
fig_loss.tight_layout()
fig_loss.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close(fig_loss)

print(f"\nAll artifacts saved in: {os.path.abspath(OUTPUT_DIR)}")
