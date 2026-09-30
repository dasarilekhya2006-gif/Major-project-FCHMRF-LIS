import apiClient from "./client";

/**
 * Upload a NIfTI volume (.nii or .nii.gz)
 * @param {File} file
 * @param {Function} onProgress
 * @returns {Promise<{experiment_id: string, filename: string, status: string, shape: number[], voxel_spacing: number[], orientation: string}>}
 */
export async function uploadNiftiVolume(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percentCompleted);
      }
    },
  });

  return response.data;
}
