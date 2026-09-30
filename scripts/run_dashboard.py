"""CLI launcher script for the Phase 10 Interactive Neuroimaging Dashboard."""

import argparse
from pathlib import Path
import subprocess
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger

logger = get_logger("run_dashboard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the interactive neuroimaging visualization and analysis dashboard."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to serve the dashboard on (default: 8501).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address to bind (default: localhost).",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(log_level="INFO")
    args = parse_args()

    app_path = Path(__file__).resolve().parent.parent / "src" / "visualization" / "dashboard_app.py"
    if not app_path.exists():
        logger.error(f"Dashboard entry point not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--browser.gatherUsageStats",
        "false",
    ]

    logger.info("=" * 80)
    logger.info("LAUNCHING INTERACTIVE NEUROIMAGING RESEARCH DASHBOARD")
    logger.info(f"  Entry Point: {app_path}")
    logger.info(f"  URL:         http://{args.host}:{args.port}")
    logger.info("=" * 80)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user.")
    except Exception as e:
        logger.error(f"Failed to launch Streamlit dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
