import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/upload": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/preprocess": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/segment": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/statistics": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/fchmrf-lis": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/run-pipeline": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/pipeline": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/results": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/visualization": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/experiment": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/experiments": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/download": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
