"""CLI script to launch the React frontend dev server."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch React Research Dashboard Dev Server")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind (default: 3000)")
    parser.add_argument("--host", action="store_true", help="Expose to local network")
    return parser.parse_args()


def main():
    args = parse_args()
    if not FRONTEND_DIR.exists():
        print(f"Error: Frontend directory not found at {FRONTEND_DIR}")
        sys.exit(1)

    print(f"Starting React Research Dashboard from {FRONTEND_DIR}")
    print(f"Connecting to FastAPI backend at http://127.0.0.1:8000")

    cmd = ["npm", "run", "dev", "--", "--port", str(args.port)]
    if args.host:
        cmd.append("--host")

    try:
        # Run npm command in frontend directory
        subprocess.run(cmd, cwd=str(FRONTEND_DIR), check=True, shell=True)
    except KeyboardInterrupt:
        print("\nStopping frontend server.")
    except Exception as e:
        print(f"\nCould not run npm directly: {e}")
        print("Please ensure Node.js and npm are installed, then run:")
        print(f"  cd {FRONTEND_DIR}")
        print("  npm install")
        print("  npm run dev")


if __name__ == "__main__":
    main()
