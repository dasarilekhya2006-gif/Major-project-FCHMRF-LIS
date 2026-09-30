import axios from "axios";

// Default to relative root (handled by Vite proxy in dev) or explicit env variable
const API_BASE_URL = import.meta.env.VITE_API_URL || "";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 minutes for heavy MCMC / 3D meshing operations
  headers: {
    Accept: "application/json",
  },
});

// Response interceptor for consistent error extraction
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorData = error.response?.data || {};
    const friendlyError = {
      status: error.response?.status || 500,
      code: errorData.error || "SERVER_ERROR",
      message: errorData.message || error.message || "An unexpected error occurred.",
      detail: errorData,
    };
    return Promise.reject(friendlyError);
  }
);

export default apiClient;
