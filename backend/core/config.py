"""Application configuration and environment settings for FastAPI backend."""

import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central backend configuration with environment variable overrides."""

    app_name: str = "Adaptive fcHMRF-LIS Neuroimaging Backend"
    api_v1_prefix: str = "/api/v1"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Base workspace directories
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    storage_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))
    )

    # Security & Upload constraints
    max_upload_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "250")))
    allowed_extensions: List[str] = Field(default_factory=lambda: [".nii", ".nii.gz"])

    # CORS configuration
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if origin.strip()
        ]
    )

    # Scientific defaults
    default_nominal_alpha: float = 0.05
    default_fchmrf_beta: float = 0.8
    default_fchmrf_lambda_dl: float = 0.5
    default_fchmrf_max_iterations: int = 25

    def ensure_directories(self) -> None:
        """Create storage root directory if it does not exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
