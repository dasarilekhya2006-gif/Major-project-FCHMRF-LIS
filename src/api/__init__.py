"""API module providing REST endpoints and FastAPI server application."""

from src.api.app import create_app, app

__all__ = ["create_app", "app"]
