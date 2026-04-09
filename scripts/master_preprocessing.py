import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE
import cv2
import librosa
import warnings
from tqdm import tqdm
import shutil

warnings.filterwarnings('ignore')

# Workspace Paths
ROOT = Path(__file__).parent.parent.absolute()
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "training"

def preprocess_vision_to_files():
    """Resizes all face images to 128x128 and saves as JPG in a preprocessed folder."""
    face_dir = DATASET_DIR / "face" / "faces"
    target_dir = DATASET_DIR / "vision_preprocessed"
    
    if not face_dir.exists():
        print("Warning: Face directory not found. Skipping vision.")
        return
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    
    print(f"[*] Preprocessing Vision Files to: {target_dir}")
    image_files = []
    for ext in ["*.jpg", "*.pgm", "*.png"]:
        image_files.extend(list(face_dir.rglob(ext)))
        
    print(f"Found {len(image_files)} shards.")
    
    size = (128, 128)
    for img_path in tqdm(image_files, desc="Vision File Preprocessing"):
        try:
            # Maintain class directory structure
            class_name = img_path.parent.name
            class_target_dir = target_dir / class_name
            class_target_dir.mkdir(parents=True, exist_ok=True)
            
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(size)
            
            # Save as jpg to target dir
            save_name = img_path.stem + ".jpg"
            img_resized.save(class_target_dir / save_name, "JPEG")
        except Exception as e:
            pass
            
    print(f"Vision preprocessing complete. Files saved in {target_dir}")

def process_emotion():
    """Processes the full FER2013 dataset."""
    csv_path = DATASET_DIR / "face_emotion" / "fer2013.csv"
    if not csv_path.exists():
        print("Warning: FER2013 labels not found.")
        return
    
    print(f"[*] Processing Full Emotion Data from: {csv_path.name}")
    df = pd.read_csv(csv_path)
    
    pixels = df['pixels'].tolist()
    width, height = 48, 48
    
    faces = []
    for pixel_sequence in tqdm(pixels, desc="Emotion Preprocessing"):
        try:
            face = np.fromstring(pixel_sequence, sep=' ', dtype=np.int32)
            face = face.reshape(width, height).astype(np.float32) / 255.0
            faces.append(face)
        except:
            continue
    
    if faces:
        res = np.array(faces)
        save_path = OUTPUT_DIR / "emotion_data_full.npy"
        np.save(save_path, res)
        print(f"Emotion data saved to {save_path}. Shape: {res.shape}")

def extract_kinematic_features(csv_path):
    """Extracts behavioral features from a single session log."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5: return None
        
        # Kinematics
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['client timestamp'].diff().replace(0, 0.001).fillna(0.001)
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2).fillna(0)
        df['vel'] = df['dist'] / df['dt']
        
        features = [
            df['vel'].mean(),
            df['vel'].max(),
            df['vel'].std() if len(df) > 1 else 0, # Jitter index
            df['dist'].sum(),
            len(df) / (df['client timestamp'].max() - df['client timestamp'].min() + 0.1)
        ]
        return features
    except Exception as e:
        return None

def process_behavior():
    """Processes full behavioral dataset with SMOTE balancing."""
    labels_path = DATASET_DIR / "behavior" / "public_labels.csv"
    behavior_root = DATASET_DIR / "behavior"
    
    if not labels_path.exists():
        print("Warning: Behavioral labels not found.")
        return
    
    labels_df = pd.read_csv(labels_path)
    print(f"[*] Mapping {len(labels_df)} sessions to feature space...")
    
    print("[*] Indexing physical session files...")
    file_map = {f.name: str(f) for f in behavior_root.rglob("session_*")}
    print(f"Found {len(file_map)} physical files available.")
    
    feature_list = []
    target_list = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Behavioral Preprocessing"):
        fname = row['filename']
        if fname in file_map:
            feats = extract_kinematic_features(file_map[fname])
            if feats:
                feature_list.append(feats)
                target_list.append(row['is_illegal'])
    
    if not feature_list:
        print("No behavioral features extracted.")
        return

    X = np.array(feature_list)
    y = np.array(target_list)
    
    print(f"[*] Class Distribution Before SMOTE: {np.bincount(y)}")
    
    if len(np.unique(y)) > 1:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"[*] Class Distribution After SMOTE: {np.bincount(y_res)}")
    else:
        X_res, y_res = X, y
    
    res_df = pd.DataFrame(X_res, columns=['vel_avg', 'vel_max', 'jitter', 'dist_total', 'density'])
    res_df['is_illegal'] = y_res
    
    save_path = OUTPUT_DIR / "behavioral_features_full.csv"
    res_df.to_csv(save_path, index=False)
    print(f"Behavioral features saved to {save_path}. Count: {len(res_df)}")

def process_voice():
    """Extracts MFCCs for all audio samples."""
    voice_dir = DATASET_DIR / "voice"
    if not voice_dir.exists():
        print("Warning: Voice directory not found.")
        return
    
    audio_files = list(voice_dir.rglob("*.wav"))
    print(f"[*] Extracting MFCCs from {len(audio_files)} audio files...")
    
    mfcc_all = []
    for audio_path in tqdm(audio_files, desc="Voice Preprocessing"):
        try:
            y_audio, sr = librosa.load(audio_path, duration=3.0)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            if mfcc.shape[1] < 130:
                mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])))
            else:
                mfcc = mfcc[:, :130]
            mfcc_all.append(mfcc)
        except Exception as e:
            pass
            
    if mfcc_all:
        res = np.array(mfcc_all)
        save_path = OUTPUT_DIR / "voice_mfcc_full.npy"
        np.save(save_path, res)
        print(f"Voice MFCC data saved to {save_path}. Shape: {res.shape}")

if __name__ == "__main__":
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
        
    print("ORIEN Master Preprocessing: FULL CONTROL MODE")
    process_emotion()
    process_behavior()
    process_voice()
    preprocess_vision_to_files()
    print("ALL MODALITIES SYNERGIZED")
