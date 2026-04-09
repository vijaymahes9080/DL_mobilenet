"""
ORIEN — Emotion CNN Trainer v7 (20-Minute Masterpiece)
======================================================
Key Fixes to Guarantee < 20 Minutes Total Training on CPU:
  1. Resolution Scaling: Reduced from 48x48 to 32x32 (2.25x fewer pixels), drastically
     cutting MACs per layer without destroying facial spatial awareness.
  2. Validation Pruning: `validation_freq=5` bypasses validation phase (which was eating 
     ~40 seconds per epoch). We now validate only 7 times total instead of 35.
  3. Algorithmic Scheduler: Switched to `CosineDecay` learning rate which doesn't
     need per-epoch validation checks, allowing us to drop `ReduceLROnPlateau`.
  4. Stability: Disabled XLA (jit_compile=False) to prevent background OOM crashes.
  
Target: 35 Epochs in EXACTLY ~15-20 Minutes | CPU-only
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
IMG_SIZE   = 32      # Scaled down to 32x32 for lightning fast convolution matrix math
BATCH_SIZE = 256     # 256 is the "safe sweet spot" for Windows CPU BLAS
EPOCHS     = 35      
SAVE_PATH  = MODELS_OUT / "emotion_master_optimal.keras"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ── Load FER2013 ──────────────────────────────────────────────────────────────
def load_fer():
    import pandas as pd
    import cv2
    print(f"[DATA] Loading {FER_CSV.name} ...")
    df = pd.read_csv(str(FER_CSV))

    def pixels_to_arr(px_str):
        # 48x48 base
        base = np.array(px_str.split(), dtype=np.float32).reshape(48, 48)
        # Interpolate to 32x32 for extreme speed training
        resized = cv2.resize(base, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return resized.reshape(IMG_SIZE, IMG_SIZE, 1)

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
def build_model(num_classes, steps_per_epoch):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    act = 'relu'
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face")
    x = layers.Rescaling(1./255.)(inp)

    # First Standard Conv
    x = layers.Conv2D(32, 3, padding='same', activation=act)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Block 1 - Separable (Ultra-lightweight depthwise ops)
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

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=act)(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    m = models.Model(inp, out, name="ORIEN_FER_v7")
    
    # Cosine Decay handles LR mathematically (no validation checking needed)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.5e-3,
        decay_steps=steps_per_epoch * EPOCHS,
        alpha=0.01
    )
    
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False  # Disabled to prevent crash seen in v6
    )
    
    params = m.count_params()
    print(f"  [BUILD] FER CNN v7 | params={params:,} ({params*4/1e6:.1f} MB)")
    return m

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Emotion CNN v7  (The 20-Min Constraint Masterpiece)")
    print(f"  TF {tf.__version__} | CPU | Res: {IMG_SIZE}x{IMG_SIZE} | batch={BATCH_SIZE} | epochs={EPOCHS}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    X_tr, y_tr, X_va, y_va, cw = load_fer()

    idx = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    steps_per_epoch = len(X_tr) // BATCH_SIZE
    model = build_model(len(CLASSES), steps_per_epoch)

    # Save only at validation breaks to save disk I/O time
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        )
    ]

    print(f"\n[TRAIN] Launching 20-minute constraint run...")
    print(f"[TRAIN] Note: Validation will only run every 5 epochs to drastically speed up completion.")
    
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        validation_freq=5,  # The ultimate time-saver
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
