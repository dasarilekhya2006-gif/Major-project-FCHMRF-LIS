"""
Startup entry point for the fcHMRF-LIS FastAPI backend.

Run from anywhere with:
    C:\Python314\python.exe run_server.py
"""

import sys
import os
from pathlib import Path

# Always add the project root to sys.path first,
# so all `from backend.xxx import yyy` imports resolve correctly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root so relative paths inside
# the backend (storage dirs, config, etc.) resolve correctly.
os.chdir(PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT / "backend")],
    )
