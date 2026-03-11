import subprocess
import sys

if __name__ == "__main__":
    print("🚀 Starting Fake News Detector...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app.py",
        "--server.port=8501",
        "--server.headless=false"
    ])
