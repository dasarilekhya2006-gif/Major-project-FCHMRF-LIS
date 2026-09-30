"""REST API routes for model serving, synthetic data generation, and interactive exploration."""

import io
import time
import uuid
from typing import Dict
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.logging import get_logger
from src.data.schemas import SyntheticBenchmarkSample
from src.data.synthetic import generate_synthetic_spatial_dataset
from src.preprocessing.spatial_graph import build_knn_spatial_adjacency
from src.statistical_analysis.mixture_models import fit_two_group_mixture_em
from src.fchmrf_lis.graph_potentials import compute_unary_potentials, compute_bilateral_weights
from src.fchmrf_lis.mean_field import run_mean_field_inference
from src.fchmrf_lis.lis_calculator import compute_lis
from src.fchmrf_lis.spatial_fdr import control_spatial_fdr_lis
from src.deep_learning.predictor import ROIPriorPredictor
from src.adaptive_parameter_selection.mf_em import MeanFieldEMSelector
from src.evaluation.benchmark import BenchmarkRunner
from src.api.schemas import (
    HealthResponse,
    SyntheticGenerateRequest,
    SyntheticGenerateResponse,
    InferenceRequest,
    InferenceResponse,
    BenchmarkRunResponse,
    MethodBenchmarkResult,
)

logger = get_logger(__name__)
router = APIRouter()

# In-memory session store for prototype exploration
SAMPLE_CACHE: Dict[str, SyntheticBenchmarkSample] = {}
DISCOVERY_CACHE: Dict[str, np.ndarray] = {}


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check API service health and version."""
    return HealthResponse()


@router.post("/synthetic/generate", response_model=SyntheticGenerateResponse)
def generate_sample(req: SyntheticGenerateRequest) -> SyntheticGenerateResponse:
    """Generate a reproducible synthetic 3D neuroimaging volume."""
    sample_id = str(uuid.uuid4())[:8]
    sample = generate_synthetic_spatial_dataset(
        grid_size=tuple(req.grid_size),
        snr=req.snr,
        num_clusters=req.num_clusters,
        seed=req.seed,
    )
    SAMPLE_CACHE[sample_id] = sample

    return SyntheticGenerateResponse(
        sample_id=sample_id,
        shape=list(sample.z_map.shape),
        snr=sample.snr,
        true_active_voxels=int(np.sum(sample.ground_truth)),
        total_voxels=int(sample.z_map.size),
        max_zscore=float(np.max(sample.z_map)),
        min_zscore=float(np.min(sample.z_map)),
    )


@router.post("/inference", response_model=InferenceResponse)
def run_inference(req: InferenceRequest) -> InferenceResponse:
    """Execute Adaptive DL-Assisted fcHMRF-LIS inference pipeline on cached sample."""
    sample_id = req.sample_id
    if not sample_id or sample_id not in SAMPLE_CACHE:
        # Generate default sample if none provided
        sample_id = "default"
        SAMPLE_CACHE[sample_id] = generate_synthetic_spatial_dataset(
            grid_size=(48, 48, 24), snr=2.0, seed=42
        )

    sample = SAMPLE_CACHE[sample_id]
    start_time = time.time()

    z_map = sample.z_map
    z_1d = z_map.ravel()
    coords = np.argwhere(np.ones(z_map.shape, dtype=bool)).astype(np.float32)
    indices, dists = build_knn_spatial_adjacency(coords, k_neighbors=18)
    diffs = np.abs(z_1d[:, None] - z_1d[indices])
    weights = compute_bilateral_weights(dists, diffs)

    mixture = fit_two_group_mixture_em(z_1d, fix_null=True)

    if req.use_dl_prior:
        predictor = ROIPriorPredictor()
        dl_prior_1d = predictor.predict(z_map).ravel()
    else:
        dl_prior_1d = np.zeros_like(z_1d)

    beta = req.beta
    lambda_dl = req.lambda_dl

    if req.adaptive_tuning and req.use_dl_prior:
        selector = MeanFieldEMSelector(max_em_iter=10, beta_init=beta, lambda_init=lambda_dl)
        beta, lambda_dl, mixture = selector.fit(z_1d, indices, weights, dl_prior_1d, mixture)

    u0, u1 = compute_unary_potentials(
        z_1d, mixture, dl_prior_1d if req.use_dl_prior else None, lambda_dl=lambda_dl
    )
    q1, _, _ = run_mean_field_inference(u0, u1, indices, weights, beta=beta)
    lis = compute_lis(q1)

    rejected, lis_cutoff, est_fdr = control_spatial_fdr_lis(lis, alpha=req.nominal_alpha)
    elapsed = time.time() - start_time

    # Cache discoveries for slice visualization
    DISCOVERY_CACHE[sample_id] = rejected.reshape(z_map.shape)

    return InferenceResponse(
        sample_id=sample_id,
        nominal_alpha=req.nominal_alpha,
        num_discoveries=int(np.sum(rejected)),
        discovery_percentage=float(100.0 * np.mean(rejected)),
        lis_cutoff_threshold=lis_cutoff,
        estimated_fdr=est_fdr,
        beta_used=float(beta),
        lambda_dl_used=float(lambda_dl),
        execution_time_seconds=round(elapsed, 4),
        method="Adaptive-DL-Assisted fcHMRF-LIS" if req.use_dl_prior else "Standard fcHMRF-LIS",
    )


@router.post("/benchmark", response_model=BenchmarkRunResponse)
def run_benchmark(
    snr: float = Query(2.0, ge=0.5, le=5.0),
    nominal_alpha: float = Query(0.05, ge=0.01, le=0.2),
) -> BenchmarkRunResponse:
    """Run full comparative benchmark across BH, Storey, locfdr, and fcHMRF variants."""
    sample = generate_synthetic_spatial_dataset(grid_size=(40, 40, 20), snr=snr, seed=42)
    runner = BenchmarkRunner(nominal_alpha=nominal_alpha, seed=42)
    eval_dict = runner.run_single_comparison(sample)

    results = []
    for method_name, m in eval_dict.items():
        results.append(
            MethodBenchmarkResult(
                method=method_name,
                empirical_fdr=m.empirical_fdr,
                power_tpr=m.tpr,
                fndr=m.fndr,
                f1_score=m.f1_score,
                dice=m.dice,
                mcc=m.mcc,
                num_discoveries=m.num_discoveries,
            )
        )

    return BenchmarkRunResponse(nominal_alpha=nominal_alpha, snr=snr, results=results)


@router.get("/slice")
def get_slice_image(
    sample_id: str = Query("default"),
    slice_idx: Optional[int] = Query(None),
    show_discoveries: bool = Query(True),
) -> Response:
    """Generate and stream a PNG image of an axial slice with discovery overlay."""
    if sample_id not in SAMPLE_CACHE:
        SAMPLE_CACHE[sample_id] = generate_synthetic_spatial_dataset(
            grid_size=(48, 48, 24), snr=2.0, seed=42
        )

    sample = SAMPLE_CACHE[sample_id]
    z_map = sample.z_map
    if slice_idx is None:
        slice_idx = z_map.shape[2] // 2
    slice_idx = int(np.clip(slice_idx, 0, z_map.shape[2] - 1))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.rot90(z_map[:, :, slice_idx]), cmap="bone")

    if show_discoveries and sample_id in DISCOVERY_CACHE:
        disc = DISCOVERY_CACHE[sample_id][:, :, slice_idx]
        if np.any(disc):
            overlay = np.zeros((*disc.shape, 4), dtype=np.float32)
            overlay[np.rot90(disc)] = [1.0, 0.2, 0.2, 0.6]  # Semi-transparent red
            ax.imshow(overlay)

    ax.set_title(f"Axial Slice z={slice_idx}", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
