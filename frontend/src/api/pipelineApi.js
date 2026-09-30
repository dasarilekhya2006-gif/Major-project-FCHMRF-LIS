import apiClient from "./client";

/**
 * Execute preprocessing stage
 */
export async function runPreprocessing(experimentId, { normalizeIntensity = true, applyBrainMask = true } = {}) {
  const response = await apiClient.post("/preprocess", {
    experiment_id: experimentId,
    normalize_intensity: normalizeIntensity,
    apply_brain_mask: applyBrainMask,
  });
  return response.data;
}

/**
 * Execute 3D U-Net ROI segmentation stage
 */
export async function runSegmentation(experimentId, { probabilityThreshold = 0.5 } = {}) {
  const response = await apiClient.post("/segment", {
    experiment_id: experimentId,
    probability_threshold: probabilityThreshold,
  });
  return response.data;
}

/**
 * Execute voxel-level statistical hypothesis testing stage
 */
export async function runStatistics(experimentId) {
  const response = await apiClient.post("/statistics", {
    experiment_id: experimentId,
  });
  return response.data;
}

/**
 * Execute fcHMRF-LIS spatial modeling and FDR control stage
 */
export async function runFchmrfLis(
  experimentId,
  { nominalAlpha = 0.05, beta = 0.8, lambdaDl = 0.5, maxIterations = 20 } = {}
) {
  const response = await apiClient.post("/fchmrf-lis", {
    experiment_id: experimentId,
    nominal_alpha: nominalAlpha,
    beta: beta,
    lambda_dl: lambdaDl,
    max_iterations: maxIterations,
  });
  return response.data;
}

/**
 * Trigger complete end-to-end pipeline asynchronously
 */
export async function runCompletePipeline(payload) {
  const response = await apiClient.post("/run-pipeline", payload);
  return response.data;
}

/**
 * Poll pipeline execution progress and current stage
 */
export async function getPipelineStatus(experimentId) {
  const response = await apiClient.get(`/pipeline/status/${experimentId}`);
  return response.data;
}
