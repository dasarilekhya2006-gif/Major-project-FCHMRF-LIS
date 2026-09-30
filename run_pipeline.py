#!/usr/bin/env python3
"""Reproducible root CLI entry point for the Integrated Neuroimaging Research Pipeline.

Usage examples:
    # Full end-to-end pipeline execution on demo volume:
    python run_pipeline.py --experiment-id demo_exp

    # Full pipeline with Bayesian Optimization adaptive parameter selection:
    python run_pipeline.py --experiment-id demo_adaptive --adaptive

    # Pipeline on custom NIfTI input:
    python run_pipeline.py --input /path/to/brain.nii.gz --experiment-id custom_exp

    # Re-run ignoring cached intermediate artifacts:
    python run_pipeline.py --experiment-id demo_exp --force-recompute
"""

import sys
from pathlib import Path

# Ensure project root is first in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_pipeline import main

if __name__ == "__main__":
    main()
