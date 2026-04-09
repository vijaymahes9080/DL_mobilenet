"""
ORIEN — Emotion CNN Trainer v6 (Hyper-Optimized)
====================================================
Key Fixes vs v5 for MAXIMUM CPU SPEED:
  - Batch Size: Increased to 512 (Massive parallel vectorization for CPU)
  - Pre-augmentation Removed: Training natively on the 28k base samples without flipping, 
    halving the epoch steps, but balancing with more epochs (35 expected)
  - JIT Compilation (XLA): Enabled locally to fuse kernels and accelerate TF ops on CPU
  - Network Thinning: Reduced max filters to 128 and removed the 4th block to hit 
    <60 seconds per epoch.
  
Target: 35 Epochs in ~30 Minutes | CPU-only, 16GB RAM friendly
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
BATCH_SIZE = 512     # 4x larger batch for massive CPU BLAS speedups
EPOCHS     = 35      # Target exactly 35 epochs as requested
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

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    act = 'relu'
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face")
    x = layers.Rescaling(1./255.)(inp)

    # First layer Conv2D
    x = layers.Conv2D(32, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 1 - Separable
    x = layers.SeparableConv2D(64, 3, padding='same', activation=act)(x)
    x = layers.SeparableConv2D(64, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.20)(x)

    # Block 2 - Separable
    x = layers.SeparableConv2D(128, 3, padding='same', activation=act)(x)
    x = layers.SeparableConv2D(128, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Head mapping — Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=act)(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    m = models.Model(inp, out, name="ORIEN_FER_v6")
    
    # JIT Compile massively speeds up CPU execution by fusing ops
    m.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    
    params = m.count_params()
    print(f"  [BUILD] FER CNN v6 (Hyper-Optimized) | params={params:,} ({params*4/1e6:.1f} MB)")
    return m

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Emotion CNN v6  (XLA Enabled, Batch 512, Thinned CNN)")
    print(f"  TF {tf.__version__} | CPU | batch={BATCH_SIZE} | epochs={EPOCHS}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    X_tr, y_tr, X_va, y_va, cw = load_fer()

    # Shuffle training data
    idx = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    model = build_model(len(CLASSES))

    # Note: No Early Stopping this time to guarantee hitting all 35 target epochs
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-7,
            monitor='val_loss', verbose=1
        ),
    ]

    print(f"\n[TRAIN] Launching 30-minute target run over {EPOCHS} epochs ...")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
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
