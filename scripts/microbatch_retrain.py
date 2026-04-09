"""
ORIEN — Micro-Batch Retrainer (Phase 2)
=========================================
Target: emotion_master (7 classes) + voice (8 classes)
Hardware: 16GB RAM, CPU-only
Strategy: batch_size=2, disk streaming, no cache, MobileNetV2-alpha-0.35

Usage:
    python scripts/microbatch_retrain.py --modality emotion
    python scripts/microbatch_retrain.py --modality voice
    python scripts/microbatch_retrain.py --modality both
"""

import os, sys, time, argparse, json
os.environ['TF_ENABLE_ONEDNN_OPTS']  = '0'
os.environ['CUDA_VISIBLE_DEVICES']   = '-1'   # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import numpy as np

def get_args():
    p = argparse.ArgumentParser(description="ORIEN Micro-Batch Retrainer")
    p.add_argument("--modality", choices=["emotion", "voice", "both"], default="both",
                   help="Which modality to retrain (default: both)")
    p.add_argument("--epochs",     type=int, default=15,
                   help="Training epochs (default: 15)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Batch size — keep at 1 or 2 for CPU OOM safety (default: 2)")
    return p.parse_args()

# ── Paths ────────────────────────────────────────────────────────────────────
from pathlib import Path
ROOT       = Path(__file__).parent.parent.absolute()
DS         = ROOT / "dataset"
MODELS_OUT = ROOT / "models" / "vmax"

EMOTION_TRAIN = DS / "face_emotion" / "train"   # 7 classes
VOICE_CLASSES = DS / "voice_cloud" / "classes"  # 8 classes (MFCC PNGs)

IMG_SIZE = 96   # Balanced: quality vs. RAM

# ── Model Builder ─────────────────────────────────────────────────────────────
def build_model(name: str, img_size: int, num_classes: int):
    """Lightweight MobileNetV2 alpha=0.35 — designed for CPU / 16GB RAM."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications

    print(f"  [BUILD] {name} — {num_classes} classes @ {img_size}x{img_size}")
    base = applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35                # smallest MobileNetV2 variant — saves ~60% RAM vs default
    )
    base.trainable = True
    # Freeze lower 40 layers (stem + early blocks) to save memory/compute
    for layer in base.layers[:40]:
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1./255),          # Internal normalization — no manual /255 needed externally
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10),
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ], name=f"ORIEN_{name.upper()}_MICROBATCH")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── Dataset Loader ─────────────────────────────────────────────────────────────
def load_dataset(path: Path, img_size: int, batch_size: int):
    """Stream from disk — NO cache — to prevent OOM."""
    import tensorflow as tf

    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    class_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if len(class_dirs) < 2:
        raise ValueError(f"Need >= 2 class subdirectories in {path}, found {len(class_dirs)}: {[d.name for d in class_dirs]}")

    print(f"  [DATA] Found {len(class_dirs)} classes in {path.name}: {[d.name for d in class_dirs]}")

    ds_pair = tf.keras.utils.image_dataset_from_directory(
        str(path),
        validation_split=0.15,
        subset="both",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int',        # sparse_categorical for memory efficiency
        shuffle=True,
    )
    train_ds, val_ds = ds_pair

    # IMPORTANT: extract class_names BEFORE applying transforms
    # (shuffle/prefetch wraps into _PrefetchDataset which loses the attribute)
    class_names = train_ds.class_names

    # DO NOT cache — stream from disk to prevent RAM OOM
    train_ds = train_ds.shuffle(buffer_size=500, reshuffle_each_iteration=True) \
                       .prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

# ── Trainer ───────────────────────────────────────────────────────────────────
def train_modality(name: str, data_path: Path, args) -> dict:
    """Full training cycle for one modality."""
    import tensorflow as tf

    print("\n" + "="*65)
    print(f"  TRAINING: {name.upper()} | batch={args.batch_size} | epochs={args.epochs}")
    print("="*65)
    t0 = time.time()

    save_dir = MODELS_OUT / name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}_optimal.keras"

    try:
        train_ds, val_ds, class_names = load_dataset(data_path, IMG_SIZE, args.batch_size)
        num_classes = len(class_names)

        model = build_model(name, IMG_SIZE, num_classes)
        model.summary(print_fn=lambda x: print("   " + x) if "Total" in x else None)

        # Patience scales with epochs to avoid premature stopping on slow CPU
        patience_es  = max(4, args.epochs // 3)
        patience_lr  = max(2, args.epochs // 6)

        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                str(save_path), save_best_only=True,
                monitor='val_accuracy', verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=patience_es, restore_best_weights=True,
                monitor='val_accuracy', verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=patience_lr, factor=0.5, min_lr=1e-6,
                monitor='val_loss', verbose=1
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=cbs,
            verbose=1,
        )

        best_acc = max(history.history.get('val_accuracy', [0.0]))
        duration = time.time() - t0

        # Save class mapping
        json_path = save_dir / "classes.json"
        json.dump(class_names, open(str(json_path), "w"))

        print(f"\n  [DONE] {name.upper()} => best val_accuracy = {best_acc:.4%} | {duration:.0f}s")
        print(f"         Model: {save_path}")
        print(f"         Classes: {json_path}")

        tf.keras.backend.clear_session()
        return {"modality": name, "accuracy": best_acc, "time": f"{duration:.0f}s",
                "status": "TRAINED" if best_acc > 0 else "LOW_ACC",
                "classes": class_names, "num_classes": num_classes}

    except Exception as e:
        import traceback
        print(f"\n  [ERROR] {name.upper()} failed: {e}")
        traceback.print_exc()
        tf.keras.backend.clear_session()
        return {"modality": name, "accuracy": 0.0, "time": "0s",
                "status": f"ERROR: {str(e)[:80]}", "classes": [], "num_classes": 0}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    print("\n" + "="*65)
    print("  ORIEN Micro-Batch Retrainer (Phase 2)")
    print(f"  Batch size: {args.batch_size} | Epochs: {args.epochs}")
    print("="*65)

    import tensorflow as tf
    print(f"  TF Version : {tf.__version__}")
    print(f"  CPUs       : {tf.config.threading.get_inter_op_parallelism_threads()} inter-op threads")
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

    target_map = {
        "emotion": [("emotion_master", EMOTION_TRAIN)],
        "voice":   [("voice_cloud",    VOICE_CLASSES)],
        "both":    [("emotion_master", EMOTION_TRAIN),
                    ("voice_cloud",    VOICE_CLASSES)],
    }

    tasks = target_map[args.modality]
    results = []
    t_global = time.time()

    for (mod_name, data_path) in tasks:
        res = train_modality(mod_name, data_path, args)
        results.append(res)

    # ── Final Report ──────────────────────────────────────────────────────────
    total_time = time.time() - t_global
    print("\n" + "="*65)
    print("  ORIEN MICRO-BATCH RETRAIN — FINAL REPORT")
    print("="*65)
    for r in results:
        verdict = "✅ READY" if r["accuracy"] >= 0.95 else ("⚠️ NEEDS MORE" if r["accuracy"] > 0 else "❌ FAILED")
        print(f"  {r['modality']:20s} | Acc: {r['accuracy']:.2%} | {r['time']:>6s} | {verdict}")

    print(f"\n  Total training time: {total_time:.0f}s")

    # Save report
    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    report_lines = [
        "# ORIEN Micro-Batch Retrain Report (Phase 2)\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "| Modality | Accuracy | Time | Status |\n",
        "|---|---|---|---|\n",
    ]
    for r in results:
        report_lines.append(f"| {r['modality']} | {r['accuracy']:.2%} | {r['time']} | {r['status']} |\n")
    report_path = MODELS_OUT / "MICROBATCH_RETRAIN_REPORT.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    print(f"\n  Report: {report_path}")
    print("="*65)

if __name__ == "__main__":
    main()
