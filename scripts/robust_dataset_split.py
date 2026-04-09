import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import json

# Setup Paths
BASE_DIR = Path("d:/current project/DL")
DATASET_DIR = BASE_DIR / "dataset"
TRAINING_DIR = BASE_DIR / "training"
SPLIT_DIR = TRAINING_DIR / "splits"

# Config
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

def setup_dirs():
    """Create the structured directory system."""
    for split in SPLIT_RATIOS.keys():
        for modality in ["vision", "emotion", "behavior", "voice"]:
            (SPLIT_DIR / split / modality).mkdir(parents=True, exist_ok=True)
    print("[*] Directory structure established.")

def split_emotion():
    """Splits FER2013 data based on 'Usage' or stratified split."""
    csv_path = DATASET_DIR / "face_emotion" / "fer2013.csv"
    if not csv_path.exists():
        print("[!] FER2013 CSV not found. Skipping...")
        return
    
    print("[*] Splitting Emotion (FER2013)...")
    df = pd.read_csv(csv_path)
    
    # Deduplication to prevent leakage (as per user request: "Remove duplicates")
    original_len = len(df)
    df = df.drop_duplicates(subset=['pixels'])
    print(f"  - Deduplicated: {original_len} -> {len(df)} samples")
    
    # FER2013 actually has Usage: Training, PublicTest, PrivateTest
    # We will map them to our train, val, test for consistency
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']
    
    # If standard split doesn't match 70/15/15, we could re-split, 
    # but FER2013 usage is industry standard for this dataset.
    
    def save_emotion_split(df_split, name):
        pixels = df_split['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48) / 255.0).tolist()
        labels = df_split['emotion'].tolist()
        np.save(SPLIT_DIR / name / "emotion" / "data.npy", np.array(pixels))
        np.save(SPLIT_DIR / name / "emotion" / "labels.npy", np.array(labels))
        print(f"  - {name}: {len(labels)} samples")

    save_emotion_split(train_df, "train")
    save_emotion_split(val_df, "val")
    save_emotion_split(test_df, "test")

def split_behavior():
    """Splits behavioral features with stratification and no leakage."""
    labels_path = DATASET_DIR / "behavior" / "public_labels.csv"
    # We use the full features if available, or recreate from labels
    if not labels_path.exists():
        print("[!] Behavioral labels not found. Skipping...")
        return
    
    print("[*] Splitting Behavioral...")
    df = pd.read_csv(labels_path)
    
    # For now, we'll split filenames and labels.
    # Training expects features, but for splitting we ensure the filenames are isolated.
    train_idx, temp_idx = train_test_split(df.index, test_size=0.30, stratify=df['is_illegal'], random_state=RANDOM_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=df.loc[temp_idx, 'is_illegal'], random_state=RANDOM_SEED)
    
    df.loc[train_idx].to_csv(SPLIT_DIR / "train" / "behavior" / "labels.csv", index=False)
    df.loc[val_idx].to_csv(SPLIT_DIR / "val" / "behavior" / "labels.csv", index=False)
    df.loc[test_idx].to_csv(SPLIT_DIR / "test" / "behavior" / "labels.csv", index=False)
    
    print(f"  - Splits completed: Train({len(train_idx)}), Val({len(val_idx)}), Test({len(test_idx)})")

def split_vision():
    """Splits Vision data (LFW) ensuring no identity leakage."""
    vision_dir = DATASET_DIR / "vision_preprocessed"
    if not vision_dir.exists():
        # Fallback to dataset/face
        vision_dir = DATASET_DIR / "face"
        
    if not vision_dir.exists():
        print("[!] Vision data not found. Skipping...")
        return

    print("[*] Splitting Vision (LFW)...")
    identities = [d for d in vision_dir.iterdir() if d.is_dir()]
    
    # Split by identity to prevent leakage
    train_ids, temp_ids = train_test_split(identities, test_size=0.30, random_state=RANDOM_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_SEED)
    
    def copy_ids(ids, split_name):
        dest_root = SPLIT_DIR / split_name / "vision"
        count = 0
        for id_path in ids:
            target = dest_root / id_path.name
            if not target.exists():
                try:
                    # Try symlink first
                    os.symlink(id_path.absolute(), target, target_is_directory=True)
                except OSError:
                    # Fallback to copy if symlink fails (e.g. no permissions on Windows)
                    shutil.copytree(id_path, target, dirs_exist_ok=True)
            count += len(list(id_path.glob("*")))
        print(f"  - {split_name}: {len(ids)} identities, {count} images")

    copy_ids(train_ids, "train")
    copy_ids(val_ids, "val")
    copy_ids(test_ids, "test")

def split_voice():
    """Splits Voice data (RAVDESS) ensuring no speaker leakage."""
    voice_dir = DATASET_DIR / "voice"
    if not voice_dir.exists():
        print("[!] Voice data not found. Skipping...")
        return

    print("[*] Splitting Voice (RAVDESS)...")
    voice_files = list(voice_dir.rglob("*.wav"))
    if not voice_files:
        print("[!] No voice files found.")
        return

    # RAVDESS actor ID is the last field in the filename
    # e.g., 03-01-01-01-01-01-01.wav -> Actor 01
    actor_to_files = {}
    for vf in voice_files:
        actor_id = vf.stem.split('-')[-1]
        if actor_id not in actor_to_files:
            actor_to_files[actor_id] = []
        actor_to_files[actor_id].append(vf)
    
    unique_actors = list(actor_to_files.keys())
    train_actors, temp_actors = train_test_split(unique_actors, test_size=0.30, random_state=RANDOM_SEED)
    val_actors, test_actors = train_test_split(temp_actors, test_size=0.50, random_state=RANDOM_SEED)

    def save_voice_split(actors, split_name):
        import librosa
        dest_dir = SPLIT_DIR / split_name / "voice"
        mfccs = []
        labels = []
        for actor in actors:
            for vf in actor_to_files[actor]:
                try:
                    # Logic same as preprocessor for consistency
                    y, sr = librosa.load(vf, duration=3.0)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    if mfcc.shape[1] < 130:
                        mfcc = np.pad(mfcc, ((0,0), (0, 130 - mfcc.shape[1])))
                    else:
                        mfcc = mfcc[:, :130]
                    mfccs.append(mfcc)
                    # Emotion is the 3rd field (0-index 2)
                    labels.append(int(vf.stem.split('-')[2]))
                except Exception as e:
                    continue
        
        np.save(dest_dir / "data.npy", np.array(mfccs))
        np.save(dest_dir / "labels.npy", np.array(labels))
        print(f"  - {split_name}: {len(actors)} actors, {len(labels)} samples")

    save_voice_split(train_actors, "train")
    save_voice_split(val_actors, "val")
    save_voice_split(test_actors, "test")

def generate_report():
    """Generates a summary of the splitting process."""
    report = {
        "status": "SUCCESS",
        "modalities": ["vision", "emotion", "behavior", "voice"],
        "distribution": {}
    }
    
    for split in SPLIT_RATIOS.keys():
        report["distribution"][split] = {}
        for mod in ["vision", "emotion", "behavior", "voice"]:
            mod_path = SPLIT_DIR / split / mod
            if mod == "vision":
                report["distribution"][split][mod] = len(list(mod_path.iterdir()))
            elif mod in ["emotion", "voice"]:
                if (mod_path / "labels.npy").exists():
                    report["distribution"][split][mod] = len(np.load(mod_path / "labels.npy"))
            elif mod == "behavior":
                if (mod_path / "labels.csv").exists():
                    report["distribution"][split][mod] = len(pd.read_csv(mod_path / "labels.csv"))

    with open(TRAINING_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"[*] Split report generated at {TRAINING_DIR / 'split_report.json'}")


if __name__ == "__main__":
    setup_dirs()
    split_emotion()
    split_behavior()
    split_vision()
    split_voice()
    generate_report()
