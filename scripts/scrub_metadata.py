import os
import re

SCRUB_MAP = {
    r"V\d+\.\d+": "Release",
    r"Optimized": "Optimized",
    r"Advanced": "Advanced",
    r"Final": "Final",
    r"System": "System",
    r"Deployment": "Deployment",
    r"\[Optimized.*?\]": "",
    r"\[Current_Active_State\]": "Stable",
    r"\[Operational\]": "2026-04-08",
}

def scrub_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        orig = content
        for pattern, replacement in SCRUB_MAP.items():
            content = re.sub(pattern, replacement, content)
            
        if content != orig:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Scrubbed: {filepath}")
    except Exception as e:
        print(f"Failed to scrub {filepath}: {e}")

if __name__ == "__main__":
    for root, dirs, files in os.walk("."):
        if ".git" in dirs: dirs.remove(".git")
        if ".venv_training" in dirs: dirs.remove(".venv_training")
        if "node_modules" in dirs: dirs.remove("node_modules")
        
        for file in files:
            if file.endswith((".py", ".bat", ".md", ".txt", ".json", ".html")):
                scrub_file(os.path.join(root, file))
