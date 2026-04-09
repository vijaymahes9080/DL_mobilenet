"""
ORIEN — Emotion CNN Trainer v5 (Ultra-Optimized)
====================================================
Key Fixes vs v4:
  - Architecture: Switched from heavy Conv2D to Depthwise Separable Convolutions (SeparableConv2D).
  - Why: Separable convolutions reduce parameter count and multiply-add operations by ~8x per layer.
  - Result: Step time on CPU should drop from ~2.3 seconds to ~500ms, vastly accelerating training
    without significantly sacrificing the model's accuracy ceiling.
  
Target: 60-70% val_acc on FER2013 | CPU-only, 16GB RAM friendly
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
IMG_SIZE   = 48      
BATCH_SIZE = 128     
EPOCHS     = 80
SAVE_PATH  = MODELS_OUT / "emotion_master_optimal.keras"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ── Load FER2013 ──────────────────────────────────────────────────────────────
def load_fer():
    import pandas as pd
    print(f"[DATA] Loading {FER_CSV.name} ...")
    df = pd.read_csv(str(FER_CSV))

    def pixels_to_arr(px_str):
        return np.array(px_str.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)

    train_df = df[df['Usage'] == 'Training'].reset_index(drop=True)
    val_df   = df[df['Usage'].isin(['PublicTest','PrivateTest'])].reset_index(drop=True)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    X_tr = np.array([pixels_to_arr(r) for r in train_df['pixels']], dtype=np.float32)
    y_tr = train_df['emotion'].values.astype(np.int32)
    X_va = np.array([pixels_to_arr(r) for r in val_df['pixels']],   dtype=np.float32)
    y_va = val_df['emotion'].values.astype(np.int32)

    from collections import Counter
    dist = Counter(y_tr.tolist())
    
    total = len(y_tr)
    n_cls = len(CLASSES)
    cw    = {k: min(3.0, total / (n_cls * v)) for k, v in dist.items()}
    return X_tr, y_tr, X_va, y_va, cw

# ── Augmentation (numpy, fast) ────────────────────────────────────────────────
def augment_numpy(X: np.ndarray, y: np.ndarray) -> tuple:
    X_flipped = X[:, :, ::-1, :]      
    X_aug = np.concatenate([X, X_flipped], axis=0)
    y_aug = np.concatenate([y, y],          axis=0)
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    act = 'relu'
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face")
    x = layers.Rescaling(1./255.)(inp)

    # First layer is standard Conv2D because input channels = 1 (Separable doesn't help here)
    x = layers.Conv2D(32, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    
    # Block 1
    x = layers.SeparableConv2D(64, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.20)(x)

    # Block 2
    x = layers.SeparableConv2D(128, 3, padding='same', activation=act)(x)
    x = layers.SeparableConv2D(128, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.SeparableConv2D(256, 3, padding='same', activation=act)(x)
    x = layers.SeparableConv2D(256, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # Head mapping — Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=act)(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    m = models.Model(inp, out, name="ORIEN_FER_v5")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    params = m.count_params()
    print(f"  [BUILD] FER CNN v5 (Separable) | params={params:,} ({params*4/1e6:.1f} MB)")
    return m

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Emotion CNN v5  (Ultra-Optimized Separable CNN)")
    print(f"  TF {tf.__version__} | CPU | batch={BATCH_SIZE} | epochs={EPOCHS}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    X_tr, y_tr, X_va, y_va, cw = load_fer()

    print(f"[AUG]  Pre-augmenting {len(X_tr)} → ", end="", flush=True)
    X_tr, y_tr = augment_numpy(X_tr, y_tr)
    print(f"{len(X_tr)} samples")

    model = build_model(len(CLASSES))

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=12, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-7,
            monitor='val_loss', verbose=1
        ),
    ]

    print(f"\n[TRAIN] Launching accelerated {EPOCHS}-epoch run ...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=cbs,
        shuffle=True,
        verbose=1,
    )

    best_acc = max(history.history.get('val_accuracy', [0.0]))
    duration = time.time() - t0

    json.dump(CLASSES, open(str(MODELS_OUT / "classes.json"), "w"))

    print(f"\n{'='*65}")
    print(f"  FINAL: val_accuracy = {best_acc:.4%}")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Time:  {duration:.0f}s ({duration/60:.1f} min)")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
