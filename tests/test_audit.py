"""Phase 17 Research-Code Audit Test Suite.

Verifies mathematical accuracy, numerical stability, reproducibility, security,
and integration across all audit checkpoints:
1. Degenerate mixture model EM initialization safety.
2. Bilateral kernel chunked message passing scale normalization.
3. Baseline Local FDR deterministic reproducibility.
4. Bayesian optimization robustness to GP fit failures.
5. Storage service directory traversal prevention via strict path containment.
6. Visualization service 3D surface mesh tuple handling.
7. Loss evaluation logit numerical handling.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

# Modules under audit
from src.statistical_analysis.mixture_models import fit_two_group_mixture_em
from src.fchmrf_lis.dense_kernel import dense_bilateral_message_passing, compute_dense_bilateral_matrix
from src.evaluation.baselines import LocalFDRCorrection
from src.adaptive_selection.search_space import create_fchmrf_search_space
from src.adaptive_selection.bayesian_optimizer import BayesianOptimizer, AcquisitionFunction
from src.adaptive_selection.objectives import BaseObjective, ObjectiveResult
from backend.services.storage_service import StorageService
from backend.services.visualization_service import visualization_service
from src.deep_learning.trainer import ROIPriorTrainer
from src.deep_learning.losses import CombinedBceDiceLoss


class TestMixtureModelAudit:
    """Audit tests for two-group Gaussian mixture models."""

    def test_degenerate_z_scores_initialization(self):
        """EM initialization must not produce NaNs when all z-scores are identical."""
        # Array of identical values
        z_degenerate = np.zeros(100, dtype=np.float64)
        params = fit_two_group_mixture_em(z_degenerate, max_iter=5, fix_null=False)

        assert not np.isnan(params.mu1), "mu1 should not be NaN on degenerate input"
        assert not np.isnan(params.sigma1), "sigma1 should not be NaN on degenerate input"
        assert params.sigma1 >= 0.5, "sigma1 should satisfy minimum standard deviation floor"


class TestBilateralKernelAudit:
    """Audit tests for fcHMRF bilateral message passing scale consistency."""

    def test_chunked_message_passing_row_normalization(self):
        """Chunked message passing must maintain row sum = 1.0 energy normalization."""
        N = 200
        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 50, size=(N, 3))
        z = rng.normal(0, 1, size=N)
        s = rng.choice([-1.0, 1.0], size=N)

        # 1. Precomputed matrix normalized mode
        K_pre = compute_dense_bilateral_matrix(
            coordinates=coords,
            z_scores=z,
            sigma_spatial=2.5,
            sigma_intensity=1.5,
            w_spatial=1.0,
            w_pure_spatial=0.2,
        )
        row_sums = np.sum(K_pre, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        K_norm = K_pre / row_sums
        expected_messages = K_norm @ s

        # 2. Chunked message passing mode with small chunk size
        actual_messages = dense_bilateral_message_passing(
            coordinates=coords,
            z_scores=z,
            belief_diff=s,
            sigma_spatial=2.5,
            sigma_intensity=1.5,
            w_spatial=1.0,
            w_pure_spatial=0.2,
            chunk_size=50,
        )

        # Both modes must produce mathematically identical normalized messages
        np.testing.assert_allclose(actual_messages, expected_messages, atol=1e-5)
        # Message magnitudes must stay bounded in [-1.0, 1.0]
        assert np.all(np.abs(actual_messages) <= 1.0001)


class TestReproducibilityAudit:
    """Audit tests for deterministic baseline behavior."""

    def test_local_fdr_reproducibility(self):
        """Local FDR baseline with degenerate variance must produce identical results across runs."""
        z_flat = np.zeros((10, 10, 10), dtype=np.float64)
        p_flat = np.ones((10, 10, 10), dtype=np.float64)

        clf1 = LocalFDRCorrection(alpha=0.05)
        res1 = clf1.run(z_scores=z_flat, p_values=p_flat)

        clf2 = LocalFDRCorrection(alpha=0.05)
        res2 = clf2.run(z_scores=z_flat, p_values=p_flat)

        np.testing.assert_array_equal(res1.significant_mask, res2.significant_mask)
        np.testing.assert_allclose(res1.test_statistic_map, res2.test_statistic_map)


class DegenerateConstantObjective(BaseObjective):
    """Objective returning identical scores to force flat GP history."""

    def evaluate(self, param_dict):
        return ObjectiveResult(score=0.5, metrics={}, elapsed_seconds=0.01)


class TestBayesianOptimizerAudit:
    """Audit tests for Bayesian Optimization GP stability."""

    def test_gp_fit_failure_graceful_recovery(self):
        """Bayesian optimizer must not crash when GP fitting fails or produces flat values."""
        space = create_fchmrf_search_space()
        obj = DegenerateConstantObjective()
        opt = BayesianOptimizer(
            search_space=space,
            objective=obj,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            seed=42,
        )

        # Run 3 iterations
        result = opt.optimize(n_iterations=3, n_initial=2)
        assert len(result.trials) == 6  # 1 baseline + 2 initial + 3 BO
        assert result.best_score == 0.5


class TestStorageSecurityAudit:
    """Audit tests for storage sandbox path traversal defenses."""

    def test_path_traversal_rejection(self):
        """StorageService must reject directory traversal attempts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageService(root_dir=Path(tmp_dir))

            # Attempt invalid experiment IDs
            with pytest.raises(HTTPException) as exc_info:
                storage.get_experiment_dir("../malicious_id", must_exist=False)
            assert exc_info.value.status_code == 400

            with pytest.raises(HTTPException):
                storage.get_experiment_dir("..\\malicious_id", must_exist=False)

    def test_artifact_path_traversal_rejection(self):
        """Artifact downloads outside the experiment sandbox must be rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageService(root_dir=Path(tmp_dir))
            exp_id = "exp_test_sandbox"
            storage.create_experiment(exp_id, "test.nii.gz")

            # Attempt artifact path traversal
            with pytest.raises(HTTPException) as exc_info:
                storage.get_artifact_path(exp_id, stage="../..", filename="secret.txt")
            assert exc_info.value.status_code in (400, 404)


class DummyProbModel(nn.Module):
    """Mock model returning raw probabilities without required_divisibility."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class TestTrainerLogitAudit:
    """Audit tests for model evaluation loss computation."""

    def test_trainer_eval_with_prob_only_model(self):
        """Trainer must evaluate models returning probabilities without crashing or invalid logits."""
        model = DummyProbModel()
        trainer = ROIPriorTrainer(model=model, loss_fn=CombinedBceDiceLoss(from_logits=True), learning_rate=1e-3)

        # Create dummy dataloader
        x = torch.randn(2, 1, 8, 8, 8)
        y = torch.randint(0, 2, (2, 1, 8, 8, 8)).float()
        dataset = torch.utils.data.TensorDataset(x, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Evaluate must run cleanly
        metrics = trainer.evaluate(loader, split_name="val")
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])


class TestVisualizationMeshAudit:
    """Audit tests for 3D surface mesh extraction and unpacking."""

    def test_3d_mesh_extraction_tuple_handling(self):
        """extract_surface_mesh 6-tuple must convert cleanly into Mesh3DData."""
        from src.visualization.interactive_3d import extract_surface_mesh
        from backend.schemas.visualization import Mesh3DData

        vol = np.zeros((10, 10, 10), dtype=float)
        vol[3:7, 3:7, 3:7] = 1.0  # Cube in center

        mesh_tuple = extract_surface_mesh(vol, level=0.5, step_size=1)
        assert mesh_tuple is not None
        assert len(mesh_tuple) == 6

        x, y, z, i, j, k = mesh_tuple
        mesh_data = Mesh3DData(
            num_vertices=len(x),
            num_faces=len(i),
            x=x.tolist(),
            y=y.tolist(),
            z=z.tolist(),
            i=i.tolist(),
            j=j.tolist(),
            k=k.tolist(),
        )
        assert mesh_data.num_vertices > 0
        assert mesh_data.num_faces > 0
