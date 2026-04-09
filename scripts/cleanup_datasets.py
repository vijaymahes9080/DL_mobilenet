import os, shutil, sys
from pathlib import Path

# 💎 ORIEN | DATASET SYNERGY CLEANER
# Removes archives and redundant data.

ROOT = Path(r"D:\current project\DL\dataset")

def purge(path):
    p = Path(path)
    if p.exists():
        if p.is_dir():
            print(f"  [PURGE] Removing redundant DIR: {p}")
            shutil.rmtree(p, ignore_errors=True)
        else:
            print(f"  [PURGE] Removing archive/file: {p}")
            os.remove(p)
    else:
        print(f"  [SKIP] Not found: {p}")

def run_cleanup():
    print("\n" + "*"*60)
    print("  ORIEN: DEEP PROJECT HYGIENE & RECLAMATION")
    print("  Goal: Purge Redundant Archives, Logs & Junk")
    print("*"*60 + "\n")
    
    # [1] Purge all Archives & Parquet Sources (since they are unpacked)
    EXTS = ('.zip', '.tar.gz', '.tgz', '.parquet', '.rar', '.pyc', '.tmp')
    for root, _, files in os.walk(ROOT):
        for f in files:
            if f.lower().endswith(EXTS):
                purge(os.path.join(root, f))
    
    # [2] Purge Project-Level Junk
    proj_root = ROOT.parent
    junk_files = ["training_live.log", "training_error.log", "test_imports.py", "test_pil.py", "dl_pipeline.py", "finalization_plan.md"]
    for f in junk_files:
        p = proj_root / f
        if p.exists():
            print(f"  [PURGE] Redundant Project Core: {p}")
            p.unlink()

    # [3] Specific Directory Redundancies
    redundant_dirs = [
        ROOT / "face" / "lfwcrop_grey",
        ROOT / "face_orl" / "ORL-Dataset-Face-Recognition-main",
        ROOT / "gesture" / "hagrid-classification-512p-no-gesture-150k"
    ]
    for d in redundant_dirs:
        if d.exists(): purge(d)

    print("\nDeep Synergy Reclamation Complete.")

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    run_cleanup()
