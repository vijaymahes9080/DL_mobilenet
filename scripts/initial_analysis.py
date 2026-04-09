import os
import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\current project\DL\dataset\behavior")
LABELS_PATH = ROOT / "public_labels.csv"
TRAIN_DIR = ROOT / "training_files"
TEST_DIR = ROOT / "test_files"

def find_session_files():
    mapping = {}
    for d in [TRAIN_DIR, TEST_DIR]:
        if not d.exists(): continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.startswith("session_"):
                    mapping[f] = os.path.join(root, f)
    return mapping

def analyze_labels():
    df = pd.read_csv(LABELS_PATH)
    print(f"Total labeled sessions: {len(df)}")
    print("Class Distribution:")
    print(df['is_illegal'].value_counts(normalize=True))
    
    file_map = find_session_files()
    df['path'] = df['filename'].map(file_map)
    
    missing = df[df['path'].isna()]
    if not missing.empty:
        print(f"Warning: {len(missing)} files not found on disk.")
    else:
        print("All labeled files found.")
        
    # Check session samples
    sample_path = df['path'].dropna().iloc[0]
    sample_df = pd.read_csv(sample_path)
    print("\nSample Session Data:")
    print(sample_df.head())
    print(f"Columns: {sample_df.columns.tolist()}")

if __name__ == "__main__":
    analyze_labels()
