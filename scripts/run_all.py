import os, subprocess, threading, time, sys, webbrowser

# 💎 ORIEN | CRYSTAL Release MASTER LAUNCHER
# ========================================
# Path-Safe, ASGI-Correct, Multi-Threaded Sync.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_backend():
    print("ORIEN | Starting Neural Brain (WS:8000)...")
    backend_dir = os.path.join(ROOT, "backend")
    # ⚠️ Absolute pathing for uvicorn app-dir to fix "Module Not Found"
    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0", "--port", "8000",
        "--app-dir", backend_dir, "--log-level", "warning"
    ]
    subprocess.run(cmd)

def run_frontend():
    print("ORIEN | Starting Atmospheric HUD (8080)...")
    frontend_dir = os.path.join(ROOT, "frontend")
    os.chdir(frontend_dir)
    subprocess.run([sys.executable, "-m", "http.server", "8080"])

def open_portal():
    print("ORIEN | Portal Establishing...")
    time.sleep(4) # Allow time for Keras models to lazy-load if needed
    webbrowser.open("http://localhost:8000")
    print("ORIEN | HUD Sync ACTIVE.")

if __name__ == "__main__":
    if sys.platform == "win32": sys.stdout.reconfigure(encoding='utf-8')

    # Start Threads
    threading.Thread(target=run_backend, daemon=True).start()
    threading.Thread(target=open_portal, daemon=True).start()

    print("\n" + "💎"*30)
    print("  ORIEN CRYSTAL Release | SYNERGY BOOT")
    print("  Status: Neural Fibers Aligned.")
    print("💎"*30 + "\n")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nORIEN | Powering down.")
        sys.exit(0)
