import apiClient from "./client";

/**
 * Get layer availability and discovery counts
 */
export async function getResults(experimentId) {
  const response = await apiClient.get(`/results/${experimentId}`);
  return response.data;
}

/**
 * Get visualization configuration and bounds
 */
export async function getVisualizationMeta(experimentId) {
  const response = await apiClient.get(`/visualization/${experimentId}`);
  return response.data;
}

/**
 * Get raw 2D slice data for grayscale MRI base and temperature heatmap overlay
 */
export async function getSliceData(experimentId, orientation = "axial", sliceIdx = null) {
  const params = { orientation };
  if (sliceIdx !== null && sliceIdx !== undefined) {
    params.slice_idx = sliceIdx;
  }
  const response = await apiClient.get(`/visualization/${experimentId}/slice`, { params });
  return response.data;
}

/**
 * Get Marching Cubes 3D surface meshes
 */
export async function getMesh3D(experimentId, stepSize = 1) {
  const response = await apiClient.get(`/visualization/${experimentId}/3d-mesh`, {
    params: { step_size: stepSize },
  });
  return response.data;
}

/**
 * Construct safe download URL for research artifact
 */
export function getDownloadUrl(experimentId, stage, filename) {
  const baseURL = apiClient.defaults.baseURL || "";
  return `${baseURL}/download/${experimentId}/${stage}/${filename}`;
}
