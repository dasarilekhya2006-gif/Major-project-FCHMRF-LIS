"""CLI script to launch the FastAPI backend server."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.core.config import settings
from src.core.logging import get_logger

logger = get_logger("run_backend")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch FastAPI Backend for Neuroimaging Analysis")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting {settings.app_name} on http://{args.host}:{args.port}")
    logger.info(f"Interactive OpenAPI Swagger documentation at http://{args.host}:{args.port}/docs")
    logger.info(f"Storage directory: {settings.storage_dir}")

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
