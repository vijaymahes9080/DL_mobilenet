import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

def print_p(msg):
    print(msg)
    sys.stdout.flush()

# Setup Paths
BASE_DIR = Path("d:/current project/DL")
SPLIT_DIR = BASE_DIR / "training" / "splits"

def audit_vision():
    print_p("[*] Auditing Vision Leakage...")
    splits = ["train", "val", "test"]
    id_sets = {}
    
    for split in splits:
        vision_path = SPLIT_DIR / split / "vision"
        if not vision_path.exists(): continue
        identities = {d.name for d in vision_path.iterdir() if d.is_dir()}
        id_sets[split] = identities
        print_p(f"  - {split}: {len(identities)} identities.")

    # Check Identity overlap
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    has_error = False
    for s1, s2 in pairs:
        overlap = id_sets[s1] & id_sets[s2]
        if overlap:
            print_p(f"  [!] CRITICAL: Identity overlap between {s1} and {s2}: {list(overlap)[:5]}...")
            has_error = True
    
    if not has_error:
        print_p("  [+] No Identity overlap detected across Vision splits.")

def audit_emotion():
    print_p("[*] Auditing Emotion Leakage...")
    splits = ["train", "val", "test"]
    pixel_hashes = {}
    
    for split in splits:
        data_path = SPLIT_DIR / split / "emotion" / "data.npy"
        if data_path.exists():
            data = np.load(data_path)
            # Sample 10% for large datasets to speed up, or all for FER (35k is fine)
            hashes = {hashlib.md5(x.tobytes()).hexdigest() for x in data}
            pixel_hashes[split] = hashes
            print_p(f"  - {split}: {len(hashes)} unique pixel hashes.")

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        if s1 in pixel_hashes and s2 in pixel_hashes:
            overlap = pixel_hashes[s1] & pixel_hashes[s2]
            if overlap:
                print_p(f"  [!] WARNING: {len(overlap)} samples overlap between {s1} and {s2}.")

def audit_behavior():
    print_p("[*] Auditing Behavior Leakage...")
    splits = ["train", "val", "test"]
    file_sets = {}
    
    for split in splits:
        labels_path = SPLIT_DIR / split / "behavior" / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            filenames = set(df['filename'].tolist())
            file_sets[split] = filenames
            print_p(f"  - {split}: {len(filenames)} unique sessions.")

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        if s1 in file_sets and s2 in file_sets:
            overlap = file_sets[s1] & file_sets[s2]
            if overlap:
                print_p(f"  [!] CRITICAL: Session overlap between {s1} and {s2}: {list(overlap)[:5]}...")

if __name__ == "__main__":
    audit_vision()
    audit_emotion()
    audit_behavior()
    print_p("[*] AUDIT COMPLETE.")

