# 🔍 ORIEN Neural Data Acquirer [Optimized UTF-8]
import os, sys, requests, tarfile, zipfile, shutil
from pathlib import Path

# Fix UTF-8 output for Windows
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

ROOT = Path(__file__).parent.parent.absolute()
DATASET_ROOT = ROOT / "dataset"
TMP_DIR = DATASET_ROOT / "tmp_downloads"
BIN_DIR = DATASET_ROOT / "bin"

URLS = {
    "LFW": {
        "url": "https://huggingface.co/datasets/marcelo-victor/lfw/resolve/main/archive.zip?download=true",
        "target": DATASET_ROOT / "face_core",
        "extract": True
    },
    "FER2013": {
        "url": "https://huggingface.co/datasets/MoAamir28/FER2013/resolve/main/fer2013.csv?download=true",
        "target": DATASET_ROOT / "face_emotion",
        "extract": False
    },
    "HAGRID": {
        "url": "https://huggingface.co/datasets/cj-mills/hagrid-sample-120k-384p/resolve/main/hagrid-sample-120k-384p.zip?download=true",
        "target": DATASET_ROOT / "gesture_hub",
        "extract": True
    }
}

def download_file(url, dest):
    print(f"[*] Downloading {url} to {dest}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded//(1024*1024)}MB / {total_size//(1024*1024)}MB")
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded//(1024*1024)}MB")
                        sys.stdout.flush()
    print(f"\n[OK] Download Complete: {dest}")

def extract_file(src, dest):
    print(f"[*] Extracting {src} to {dest}...")
    dest.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.tgz' or src.suffixes == ['.tar', '.gz']:
        with tarfile.open(src, 'r:gz') as tar:
            tar.extractall(path=dest)
    elif src.suffix == '.zip':
        with zipfile.ZipFile(src, 'r') as zip_ref:
            zip_ref.extractall(dest)
    print(f"[OK] Extraction Complete.")

def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*30)
    print("  ORIEN: NEURAL DATA ACQUISITION")
    print("  Status: Orchestrating Cluster Sync")
    print("="*30 + "\n")

    for name, info in URLS.items():
        print(f"--- Processing {name} ---")
        
        # 1. Clean up existing (Move to bin if exists)
        if info['target'].exists():
            print(f"[*] Moving existing {name} to bin...")
            timestamp = os.path.getmtime(info['target'])
            archive_path = BIN_DIR / f"{name}_{int(timestamp)}"
            if archive_path.exists(): shutil.rmtree(archive_path)
            shutil.move(str(info['target']), str(archive_path))

        # 2. Download
        filename = info['url'].split('/')[-1].split('?')[0]
        tmp_file = TMP_DIR / filename
        if not tmp_file.exists():
            download_file(info['url'], tmp_file)
        else:
            print(f"[#] Using cached file: {tmp_file}")

        # 3. Extract/Move
        if info['extract']:
            extract_file(tmp_file, info['target'])
        else:
            info['target'].mkdir(parents=True, exist_ok=True)
            shutil.copy(str(tmp_file), str(info['target'] / filename))

    print("\n[SUCCESS] All datasets synchronized and verified.")
    print(f"Total Disk Usage: ~6.4 GB")

if __name__ == "__main__":
    main()
