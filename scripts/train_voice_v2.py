"""
ORIEN — Voice Emotion LSTM Trainer v2
=======================================
Architecture: 1D-CNN + BiLSTM on raw MFCC feature vectors
Data:         RAVDESS WAV files (dataset/voice/Actor_*/*.wav)
Classes:      8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
Hardware:     CPU-only, 16GB RAM

Why this approach beats MFCC-PNG + MobileNetV2:
  - MFCC features are 1D time-series → suited for LSTM/1D-CNN
  - MobileNetV2 ImageNet weights are irrelevant to spectrogram textures
  - Raw-feature approach: 40 MFCCs × 130 frames = (130,40) time-series input
"""

import os, sys, time, json, warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']  = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
warnings.filterwarnings('ignore')

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent.absolute()
DS         = ROOT / "dataset" / "voice"
MODELS_OUT = ROOT / "models" / "vmax" / "voice_cloud"
MODELS_OUT.mkdir(parents=True, exist_ok=True)
SAVE_PATH  = MODELS_OUT / "voice_cloud_optimal.keras"

# RAVDESS emotion labels (1-indexed in filename)
CLASSES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N_MFCC   = 40
MAX_FRAMES = 130   # ~3 seconds at 22050 Hz with hop_length=512
BATCH_SIZE = 16    # Reduced from 32 to prevent BiLSTM gradient OOM
EPOCHS     = 80

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(wav_path: Path):
    """Extract MFCC time-series from a WAV file as (MAX_FRAMES, N_MFCC) array."""
    import librosa
    y, sr = librosa.load(str(wav_path), duration=3.0, sr=22050)
    # MFCC: shape (N_MFCC, time_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=512)
    # Delta features add temporal dynamics
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, delta, delta2]).T   # (time_frames, N_MFCC*3)

    # Pad or truncate to MAX_FRAMES
    T = feat.shape[0]
    if T < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - T, feat.shape[1]), dtype=np.float32)
        feat = np.vstack([feat, pad])
    else:
        feat = feat[:MAX_FRAMES]
    return feat.astype(np.float32)

def load_ravdess():
    print(f"[DATA] Scanning {DS} for Actor folders...")
    actors = sorted([d for d in DS.iterdir() if d.is_dir() and d.name.startswith("Actor_")])
    print(f"  Found {len(actors)} actors")

    X, y = [], []
    skipped = 0
    for actor in actors:
        for wav in sorted(actor.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) < 3:
                skipped += 1
                continue
            try:
                emo_code = int(parts[2])   # 1-indexed
                label    = emo_code - 1    # 0-indexed
                if not (0 <= label < 8):
                    skipped += 1
                    continue
                feat = extract_features(wav)
                X.append(feat)
                y.append(label)
            except Exception as e:
                skipped += 1

        print(f"  [OK] {actor.name} | total so far: {len(X)}")

    X = np.array(X, dtype=np.float32)   # (N, MAX_FRAMES, N_MFCC*3)
    y = np.array(y, dtype=np.int32)
    print(f"\n  Dataset: {X.shape} | {len(y)} samples | skipped={skipped}")

    from collections import Counter
    dist = Counter(y.tolist())
    print(f"  Class dist: { {CLASSES[k]: v for k,v in sorted(dist.items())} }")

    return X, y, dist

def compute_class_weights(dist):
    total = sum(dist.values())
    n_cls = len(CLASSES)
    cw = {k: total / (n_cls * v) for k, v in dist.items()}
    print(f"  Class weights: { {CLASSES[k]: f'{v:.2f}' for k,v in sorted(cw.items())} }")
    return cw

# ── Normalize ─────────────────────────────────────────────────────────────────
def normalize_features(X_tr, X_va):
    """Standard-score normalization based on training statistics."""
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_tr - mean) / std, (X_va - mean) / std, mean, std

# ── Model ─────────────────────────────────────────────────────────────────────
def build_voice_model(input_shape, num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, optimizers

    L2 = 1e-4
    inp = layers.Input(shape=input_shape, name="mfcc_input")

    # 1D-CNN blocks — reduced filters to halve OOM risk
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.35)(x)

    # BiLSTM — reduced units to prevent gradient OOM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                          kernel_regularizer=regularizers.l2(L2)))(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=False,
                                          kernel_regularizer=regularizers.l2(L2)))(x)
    x = layers.Dropout(0.40)(x)

    # Dense head
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name="ORIEN_VOICE_BILSTM_v2")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    params = model.count_params()
    print(f"  [BUILD] BiLSTM Voice Model | params={params:,} ({params*4/1e6:.1f} MB)")
    return model

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Voice BiLSTM Trainer v2  (RAVDESS MFCC time-series)")
    print(f"  TF {tf.__version__} | CPU-only | batch={BATCH_SIZE} | epochs={EPOCHS}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    X, y, dist = load_ravdess()
    cw         = compute_class_weights(dist)

    # Stratified split: 85% train, 15% val
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"\n  Split: train={len(X_tr)} | val={len(X_va)}")

    # Normalize
    X_tr, X_va, mean, std = normalize_features(X_tr, X_va)
    np.save(str(MODELS_OUT / "voice_mean.npy"), mean)
    np.save(str(MODELS_OUT / "voice_std.npy"),  std)
    print(f"  Feature shape: {X_tr.shape}  (samples, frames, features)")

    model = build_voice_model(X_tr.shape[1:], len(CLASSES))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=7, factor=0.5, min_lr=1e-7,
            monitor='val_loss', verbose=1
        ),
    ]

    print(f"\n[TRAIN] Starting {EPOCHS}-epoch BiLSTM training...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    best_acc = max(history.history.get('val_accuracy', [0.0]))
    duration = time.time() - t0

    # Save class mapping
    json.dump(CLASSES, open(str(MODELS_OUT / "classes.json"), "w"))

    print(f"\n{'='*65}")
    print(f"  FINAL: best val_accuracy = {best_acc:.4%}")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Time:  {duration:.0f}s ({duration/60:.1f} min)")
    if best_acc >= 0.75:
        print("  STATUS: ✅ EXCELLENT (above RAVDESS SOTA baseline)")
    elif best_acc >= 0.60:
        print("  STATUS: ✅ GOOD")
    elif best_acc >= 0.40:
        print("  STATUS: ⚠️ FAIR — try more epochs")
    else:
        print("  STATUS: ❌ LOW")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
