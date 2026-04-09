"""
ORIEN — Emotion Master Trainer v2
====================================
Strategy:
  Phase 1 (5 epochs): Freeze backbone, LR=1e-3 → train only head
  Phase 2 (25 epochs): Unfreeze all, LR=1e-4 → fine-tune full network
Data: FER2013 CSV (48x48 grayscale → RGB upscale to 96x96)
Fixes: class weights for imbalance, no layer freeze OOM, proper LR
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
DS         = ROOT / "dataset"
MODELS_OUT = ROOT / "models" / "vmax" / "emotion_master"
MODELS_OUT.mkdir(parents=True, exist_ok=True)

FER_CSV    = DS / "face_emotion" / "fer2013.csv"
IMG_SIZE   = 96
BATCH_SIZE = 16   # safe for 48→96 px upscale on 16GB RAM
EPOCHS_P1  = 5    # Phase 1: head warm-up (frozen backbone)
EPOCHS_P2  = 30   # Phase 2: full fine-tune
SAVE_PATH  = MODELS_OUT / "emotion_master_optimal.keras"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def load_fer_data():
    """Load FER2013 CSV and return (X_train, y_train, X_val, y_val) as numpy arrays."""
    import pandas as pd

    print(f"[DATA] Loading FER2013 from {FER_CSV} ...")
    df = pd.read_csv(str(FER_CSV))
    print(f"       Total: {len(df)} samples | Classes: {sorted(df['emotion'].unique())}")

    def row_to_img(row):
        arr = np.array(row['pixels'].split(), dtype=np.float32).reshape(48, 48)
        return arr

    train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
    val_df   = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])].reset_index(drop=True)

    print(f"       Train: {len(train_df)} | Val: {len(val_df)}")

    def df_to_arrays(d):
        X, y = [], []
        for _, row in d.iterrows():
            arr = row_to_img(row)
            rgb = np.stack([arr, arr, arr], axis=-1)  # grayscale → 3-channel
            X.append(rgb)
            y.append(int(row['emotion']))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    print("       Converting train set...")
    X_train, y_train = df_to_arrays(train_df)
    print("       Converting val set...")
    X_val,   y_val   = df_to_arrays(val_df)

    # Values already in [0,255] — model's Rescaling(1./255) handles normalization
    print(f"       X_train: {X_train.shape} | X_val: {X_val.shape}")
    print(f"       Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, y_train, X_val, y_val

def compute_class_weights(y):
    """Compute balanced class weights to handle FER imbalance."""
    from collections import Counter
    counts = Counter(y)
    total  = len(y)
    n_cls  = len(counts)
    weights = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
    print(f"[CW]  Class weights: { {CLASSES[k]: f'{v:.2f}' for k,v in sorted(weights.items())} }")
    return weights

def build_model(num_classes: int, img_size: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, optimizers

    print(f"[BUILD] MobileNetV2 alpha=0.35 | {num_classes} classes | {img_size}x{img_size}")
    base = applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35
    )
    # Phase 1: freeze all backbone
    base.trainable = False

    inp = layers.Input(shape=(img_size, img_size, 3), name="input_face")
    # Normalization (model handles internally — raw [0,255] input expected)
    x = layers.Rescaling(1./255, name="rescale")(inp)
    # Augmentation (active during training only)
    x = layers.RandomFlip("horizontal", name="aug_flip")(x)
    x = layers.RandomRotation(0.12, name="aug_rot")(x)
    x = layers.RandomZoom(0.10, name="aug_zoom")(x)
    x = layers.RandomContrast(0.15, name="aug_contrast")(x)
    # Backbone
    x = base(x, training=False)   # BN layers use stored stats during freeze
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(256, activation='swish', name="fc1")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    out = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = models.Model(inp, out, name="ORIEN_EMOTION_V2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"       Trainable layers: {trainable}/{total} (Phase 1 — head only)")
    return model, base

def make_tf_dataset(X, y, batch_size, shuffle=True):
    """Create tf.data.Dataset from numpy arrays — streams without full cache."""
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(5000, len(X)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Emotion Master Trainer v2  (2-Phase FER2013)")
    print(f"  TF {tf.__version__} | CPU-only | img={IMG_SIZE} | batch={BATCH_SIZE}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    # ── Load Data ──────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val = load_fer_data()
    class_weights = compute_class_weights(y_train)
    num_classes   = len(CLASSES)

    # Upscale from 48→96 using TF (batched, memory-friendly)
    print(f"\n[RESIZE] Upscaling {len(X_train)} train + {len(X_val)} val images 48→{IMG_SIZE}...")

    def resize_batch(X, size=IMG_SIZE, chunk=512):
        out = []
        for i in range(0, len(X), chunk):
            batch = tf.image.resize(X[i:i+chunk], (size, size)).numpy()
            out.append(batch)
            if (i // chunk) % 5 == 0:
                print(f"  Resized {min(i+chunk, len(X))}/{len(X)}")
        return np.concatenate(out, axis=0)

    X_train = resize_batch(X_train)
    X_val   = resize_batch(X_val)
    print(f"  Done. X_train: {X_train.shape} ({X_train.nbytes/1e6:.1f} MB)")

    train_ds = make_tf_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   BATCH_SIZE, shuffle=False)

    # ── Build Model ────────────────────────────────────────────────────────────
    model, base = build_model(num_classes, IMG_SIZE)

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 1: Head warm-up — frozen backbone, LR=1e-3
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Head warm-up | {EPOCHS_P1} epochs | LR=1e-3")
    print(f"{'='*65}")

    p1_path = MODELS_OUT / "emotion_phase1.keras"
    cb_p1 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(p1_path), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
    ]
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_P1,
        class_weight=class_weights,
        callbacks=cb_p1,
        verbose=1,
    )
    best_p1 = max(hist1.history.get('val_accuracy', [0.0]))
    print(f"\n  Phase 1 best val_acc: {best_p1:.2%}")

    # ════════════════════════════════════════════════════════════════════════════
    # PHASE 2: Full fine-tune — unfreeze backbone, LR=1e-4
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Full fine-tune | {EPOCHS_P2} epochs | LR=1e-4")
    print(f"{'='*65}")

    # Unfreeze backbone (except BN layers — keep frozen for stability)
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False   # keep BN stats frozen — prevents instability

    trainable = sum(1 for l in model.layers if l.trainable)
    total     = len(model.layers)
    print(f"  Trainable layers: {trainable}/{total} (Phase 2 — all unfrozen except BN)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    patience_es = max(6, EPOCHS_P2 // 4)
    patience_lr = max(3, EPOCHS_P2 // 8)

    cb_p2 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=patience_es, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=patience_lr, factor=0.4, min_lr=1e-7,
            monitor='val_loss', verbose=1
        ),
    ]

    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_P2,
        class_weight=class_weights,
        callbacks=cb_p2,
        verbose=1,
    )
    best_p2 = max(hist2.history.get('val_accuracy', [0.0]))

    # ── Save Artifacts ────────────────────────────────────────────────────────
    json.dump(CLASSES, open(str(MODELS_OUT / "classes.json"), "w"))
    duration = time.time() - t0

    print(f"\n{'='*65}")
    print(f"  DONE | Phase1: {best_p1:.2%} → Phase2: {best_p2:.2%}")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Time:  {duration:.0f}s ({duration/60:.1f} min)")
    if best_p2 >= 0.65:
        print("  STATUS: ✅ ACCEPTABLE (FER2013 SOTA ~73%)")
    elif best_p2 >= 0.50:
        print("  STATUS: ⚠️ FAIR — continue training or add more data")
    else:
        print("  STATUS: ❌ LOW — check data pipeline")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
