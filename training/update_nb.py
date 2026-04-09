import nbformat as nbf
from pathlib import Path

notebook_path = Path('d:/current project/DL/training/DATA_PREPROCESSING.ipynb')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Create Emotion Section
emotion_markdown = nbf.v4.new_markdown_cell("## 🎭 Section 1.5: Emotion Preprocessing (FER2013)\nProcessing standard facial expression pixels for neural tone mapping.")
emotion_code = nbf.v4.new_code_cell("""def process_emotion():
    csv_path = DATASET_DIR / \"face_emotion\" / \"fer2013.csv\"
    if not csv_path.exists():
        print(\"⚠️ FER2013 labels not found.\")
        return None
    
    print(f\"[*] Processing Emotion Data from: {csv_path.name}\")
    # Using low_memory=False to avoid DtypeWarning
    df = pd.read_csv(csv_path)
    
    # FER2013 pixels are space-separated strings in 48x48 format
    pixels = df['pixels'].tolist()
    width, height = 48, 48
    
    faces = []
    # Processing first 1000 samples for verification synergy
    for pixel_sequence in pixels[:1000]:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = face / 255.0
        faces.append(face)
    
    faces = np.asarray(faces)
    print(f\"✅ Processed {len(faces)} emotion samples. Shape: {faces.shape}\")
    return faces

emotion_data = process_emotion()""")

# Insert before Section 2 (which should be behavioral)
# Let's find index where Section 2 starts
insert_idx = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and 'Section 2: Behavioral' in cell.source:
        insert_idx = i
        break

if insert_idx != -1:
    nb.cells.insert(insert_idx, emotion_markdown)
    nb.cells.insert(insert_idx + 1, emotion_code)
else:
    nb.cells.append(emotion_markdown)
    nb.cells.append(emotion_code)

# Update Export block to include emotion_data
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Saving preprocessed artifacts' in cell.source:
        cell.source = cell.source.replace(
            "print(\"[*] Saving preprocessed artifacts...\")",
            "print(\"[*] Saving preprocessed artifacts...\")\n\n# Save Emotion Data\nif 'emotion_data' is not None and 'emotion_data' in locals():\n    np.save(OUTPUT_DIR / \"emotion_data_batch.npy\", emotion_data)\n    print(f\"✅ Emotion data saved to: {OUTPUT_DIR / 'emotion_data_batch.npy'}\")"
        )
        cell.source = cell.source.replace(
            "print(f\"Vision Shards: {len(vision_data) if 'vision_data' in locals() else 0} verified\")",
            "print(f\"Vision Shards: {len(vision_data) if 'vision_data' in locals() else 0} verified\")\nprint(f\"Emotion Samples: {len(emotion_data) if 'emotion_data' in locals() and emotion_data is not None else 0} verified\")"
        )

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated with Emotion Preprocessing section.")
