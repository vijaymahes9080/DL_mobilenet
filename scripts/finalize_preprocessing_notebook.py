import nbformat as nbf
from pathlib import Path

notebook_path = Path('d:/current project/DL/training/DATA_PREPROCESSING.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Expand Vision to full dataset (Remove :5 limit)
for cell in nb.cells:
    if 'image_files = list(face_dir.rglob("*.jpg"))[:5]' in cell.source:
        cell.source = cell.source.replace('[:5]', '')
        # Also remove the plotting logic for 13000 images to avoid notebook bloat, 
        # or just plot the first 5 but process all.
        cell.source = cell.source.replace(
            'for i, img_path in enumerate(image_files):',
            'for i, img_path in enumerate(image_files):\n        # Process all but only plot first 5\n        if i >= 5: \n            img = Image.open(img_path).convert("RGB")\n            img_resized = img.resize(output_size)\n            img_array = np.array(img_resized) / 255.0\n            processed_samples.append(img_array)\n            continue'
        )

# 2. Expand Emotion to full dataset (Remove :1000 limit)
for cell in nb.cells:
    if 'for pixel_sequence in pixels[:1000]:' in cell.source:
        cell.source = cell.source.replace('[:1000]', '')

# 3. Fix Behavioral (Replace mock with real feature extraction)
behavior_code = """
def extract_kinematic_features(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 5: return None
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['client timestamp'].diff().replace(0, 0.001)
        df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['vel'] = df['dist'] / df['dt']
        return [df['vel'].mean(), df['vel'].max(), df['vel'].std() if len(df)>1 else 0, df['dist'].sum(), len(df)/(df['client timestamp'].max()-df['client timestamp'].min()+0.1)]
    except: return None

def process_behavior():
    labels_path = DATASET_DIR / "behavior" / "public_labels.csv"
    train_dir = DATASET_DIR / "behavior" / "training_files"
    if not labels_path.exists(): return None, None
    labels_df = pd.read_csv(labels_path)
    file_map = {f: os.path.join(root, f) for root, _, files in os.walk(train_dir) for f in files if f.startswith("session_")}
    features, targets = [], []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        if row['filename'] in file_map:
            f = extract_kinematic_features(file_map[row['filename']])
            if f: features.append(f); targets.append(row['is_illegal'])
    X, y = np.array(features), np.array(targets)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    df_res = pd.DataFrame(X_res, columns=['vel_avg', 'vel_max', 'jitter', 'dist_total', 'interaction_density'])
    df_res['is_illegal'] = y_res
    return df_res, y_res

behavior_features, behavior_labels = process_behavior()
"""
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Mapping 816 sessions' in cell.source:
        cell.source = behavior_code

# 4. Fix Voice (Batch MFCC extraction)
voice_code = """
import librosa
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, duration=3.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < 130: mfcc = np.pad(mfcc, ((0,0), (0, 130 - mfcc.shape[1])))
    else: mfcc = mfcc[:, :130]
    return mfcc

voice_sample_dir = DATASET_DIR / "voice"
voice_files = list(voice_sample_dir.rglob("*.wav"))
mfcc_all = []
print(f"[*] Batch processing {len(voice_files)} voice samples...")
for vf in tqdm(voice_files):
    try: mfcc_all.append(extract_mfcc(vf))
    except: pass
mfcc_all = np.array(mfcc_all)
print(f"✅ Voice MFCC data prepared. Shape: {mfcc_all.shape}")
"""
for cell in nb.cells:
    if cell.cell_type == 'code' and 'extract_mfcc(sample_file)' in cell.source:
        cell.source = voice_code

# 5. Update Save Block for full data filenames
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Saving preprocessed artifacts' in cell.source:
        cell.source = cell.source.replace('emotion_data_batch.npy', 'emotion_data_full.npy')
        cell.source = cell.source.replace('behavioral_features_balanced.csv', 'behavioral_features_full.csv')
        cell.source += "\\nif 'mfcc_all' in locals():\\n    np.save(OUTPUT_DIR / 'voice_mfcc_full.npy', mfcc_all)\\n    print('✅ Saved voice_mfcc_full.npy')"

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook localized and expanded for FULL DATASET processing.")
