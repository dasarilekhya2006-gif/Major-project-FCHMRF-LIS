import apiClient from "./client";

/**
 * Get full provenance, timestamps, parameters, and metadata
 */
export async function getExperimentDetails(experimentId) {
  const response = await apiClient.get(`/experiment/${experimentId}`);
  return response.data;
}

/**
 * List all past experiments stored on the backend
 */
export async function listExperiments() {
  const response = await apiClient.get("/experiments");
  return response.data;
}

/**
 * Service health probe
 */
export async function checkBackendHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}
