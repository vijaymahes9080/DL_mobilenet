"""
ORIEN — Emotion Master CNN Trainer v3
========================================
Architecture: Lightweight custom CNN (no pretrained weights)
              Designed for FER2013 grayscale 48x48 images → 7 classes
Strategy:     Direct CSV load, class-weighted loss, ReduceLR, EarlyStop
Hardware:     CPU-only, 16GB RAM, batch_size=64

Note on FER2013 SOTA: Best published results are ~73-75% val accuracy.
This trainer targets 60-70% which is realistic on CPU without augmentation tricks.
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
IMG_SIZE   = 48     # Keep native FER resolution — no upscale needed
BATCH_SIZE = 64     # Large batch is fine at 48x48 on 16GB
EPOCHS     = 60     # Custom CNN trains fast per epoch — allow long convergence
SAVE_PATH  = MODELS_OUT / "emotion_master_optimal.keras"

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ── Data ──────────────────────────────────────────────────────────────────────
def load_fer():
    import pandas as pd
    print(f"[DATA] Loading {FER_CSV.name}...")
    df = pd.read_csv(str(FER_CSV))

    def to_array(pixels_str):
        arr = np.array(pixels_str.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)
        return arr  # [0,255] values — normalised inside model

    train_rows = df[df['Usage'] == 'Training'].reset_index(drop=True)
    val_rows   = df[df['Usage'].isin(['PublicTest','PrivateTest'])].reset_index(drop=True)

    print(f"  Train: {len(train_rows)} | Val: {len(val_rows)}")

    X_tr = np.array([to_array(r) for r in train_rows['pixels']], dtype=np.float32)
    y_tr = train_rows['emotion'].values.astype(np.int32)
    X_va = np.array([to_array(r) for r in val_rows['pixels']],   dtype=np.float32)
    y_va = val_rows['emotion'].values.astype(np.int32)

    from collections import Counter
    dist = Counter(y_tr.tolist())
    print(f"  Class dist: { {CLASSES[k]: v for k,v in sorted(dist.items())} }")

    # Class weights (balanced)
    total = len(y_tr)
    n_cls = len(CLASSES)
    cw = {k: total / (n_cls * v) for k, v in dist.items()}
    print(f"  Class weights: { {CLASSES[k]: f'{v:.2f}' for k,v in sorted(cw.items())} }")

    return X_tr, y_tr, X_va, y_va, cw

# ── Model ─────────────────────────────────────────────────────────────────────
def build_fer_cnn(num_classes: int):
    """
    Proven FER2013 architecture: VGG-style deep CNN.
    References: Mollahosseini et al., DeepEmotion, multiple Kaggle winners.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    L2 = 1e-4
    act = 'elu'

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="face_input")
    x = layers.Rescaling(1./255.0)(inp)

    # Block 1
    x = layers.Conv2D(64, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(128, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Conv2D(128, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.30)(x)

    # Block 3
    x = layers.Conv2D(256, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Conv2D(256, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)

    # Block 4
    x = layers.Conv2D(512, 3, padding='same', activation=act,
                      kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.40)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation=act,
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(256, activation=act,
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out, name="ORIEN_FER_CNN_v3")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    params = model.count_params()
    print(f"  [BUILD] Custom FER CNN | params={params:,} ({params*4/1e6:.1f} MB)")
    return model

# ── Augmentation ──────────────────────────────────────────────────────────────
def make_dataset(X, y, batch_size, augment=False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        def aug(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.85, 1.15)
            img = tf.pad(img, [[2,2],[2,2],[0,0]])
            img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 1])
            return img, label
        ds = ds.shuffle(min(5000, len(X)), reshuffle_each_iteration=True)
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    t0 = time.time()

    print("\n" + "="*65)
    print("  ORIEN Emotion CNN Trainer v3  (FER2013 Native 48x48)")
    print(f"  TF {tf.__version__} | CPU-only | batch={BATCH_SIZE} | epochs={EPOCHS}")
    print("="*65)

    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    X_tr, y_tr, X_va, y_va, cw = load_fer()

    train_ds = make_dataset(X_tr, y_tr, BATCH_SIZE, augment=True)
    val_ds   = make_dataset(X_va, y_va, BATCH_SIZE, augment=False)

    model = build_fer_cnn(len(CLASSES))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(SAVE_PATH), save_best_only=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5, factor=0.5, min_lr=1e-7,
            monitor='val_loss', verbose=1
        ),
    ]

    print(f"\n[TRAIN] Starting {EPOCHS}-epoch training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    best_acc  = max(history.history.get('val_accuracy', [0.0]))
    duration  = time.time() - t0

    json.dump(CLASSES, open(str(MODELS_OUT / "classes.json"), "w"))
    print(f"\n{'='*65}")
    print(f"  FINAL: best val_accuracy = {best_acc:.4%}")
    print(f"  Model: {SAVE_PATH}")
    print(f"  Time:  {duration:.0f}s ({duration/60:.1f} min)")
    if best_acc >= 0.65:
        print("  STATUS: ✅ ACCEPTABLE (FER2013 SOTA ~73%)")
    elif best_acc >= 0.50:
        print("  STATUS: ⚠️ FAIR — continue training")
    else:
        print("  STATUS: ❌ LOW — check pipeline")
    print(f"{'='*65}")

if __name__ == "__main__":
    main()
