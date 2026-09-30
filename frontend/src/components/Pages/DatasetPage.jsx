import React, { useState } from "react";
import {
  Database, Upload, FileCheck, AlertCircle, Loader2,
  CheckCircle2, Info,
} from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as uploadApi from "../../api/uploadApi";

function MetaRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0",
      borderBottom: "1px solid var(--border-dim)", fontSize: 12 }}>
      <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>{label}</span>
      <span style={{ color: "var(--text-primary)", fontFamily: "JetBrains Mono, monospace" }}>{value}</span>
    </div>
  );
}

export default function DatasetPage() {
  const {
    activeExperimentId, setActiveExperimentId,
    datasetMeta, setDatasetMeta,
    refreshResults, refreshExperimentsList,
    changeOrientation,
  } = useExperiment();

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const processFile = async (file) => {
    if (!file) return;
    if (!file.name.endsWith(".nii") && !file.name.endsWith(".nii.gz")) {
      setErrorMessage("Only .nii or .nii.gz NIfTI files are supported.");
      return;
    }
    setErrorMessage(null);
    setIsUploading(true);
    setUploadProgress(0);
    try {
      const res = await uploadApi.uploadNiftiVolume(file, (p) => setUploadProgress(p));
      setActiveExperimentId(res.experiment_id);
      setDatasetMeta({
        filename: res.filename,
        shape: res.shape,
        voxel_spacing: res.voxel_spacing,
        orientation: res.orientation,
        data_type: res.data_type,
        file_size: file.size,
      });
      await refreshExperimentsList();
      await refreshResults(res.experiment_id);
      changeOrientation("axial");
    } catch (err) {
      setErrorMessage(err.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (e) => processFile(e.target.files?.[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    processFile(e.dataTransfer.files?.[0]);
  };

  const formatBytes = (bytes) => {
    if (!bytes) return "—";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 700 }}>
      <div>
        <div className="section-title">Dataset</div>
        <div className="section-sub">Upload a NIfTI neuroimaging volume to begin analysis</div>
      </div>

      {/* Upload card */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Upload size={13} /> Upload NIfTI Volume</span>
          <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "JetBrains Mono, monospace" }}>
            .nii  ·  .nii.gz
          </span>
        </div>

        {!datasetMeta ? (
          <>
            <label
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              style={{
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", gap: 12, padding: "40px 24px",
                border: `2px dashed ${isDragOver ? "var(--accent)" : "var(--border-subtle)"}`,
                borderRadius: 8, cursor: isUploading ? "not-allowed" : "pointer",
                background: isDragOver ? "var(--accent-dim)" : "var(--bg-input)",
                transition: "all 0.15s",
              }}
            >
              <input
                type="file" accept=".nii,.gz" style={{ display: "none" }}
                onChange={handleFileChange} disabled={isUploading}
              />
              {isUploading ? (
                <div style={{ textAlign: "center", width: "100%", maxWidth: 280 }}>
                  <Loader2 size={28} style={{ color: "var(--accent)", marginBottom: 10 }} className="animate-spin" />
                  <div style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 10 }}>
                    Validating NIfTI headers… {uploadProgress}%
                  </div>
                  <div style={{ width: "100%", height: 4, background: "var(--border-dim)", borderRadius: 2 }}>
                    <div style={{ width: `${uploadProgress}%`, height: "100%",
                      background: "var(--accent)", borderRadius: 2, transition: "width 0.3s" }} />
                  </div>
                </div>
              ) : (
                <>
                  <div style={{
                    width: 52, height: 52, borderRadius: 12,
                    background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    color: "var(--text-muted)",
                  }}>
                    <Upload size={22} />
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-secondary)" }}>
                      Drop NIfTI file here or click to browse
                    </div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                      Supports .nii and .nii.gz formats · Validated with NiBabel
                    </div>
                  </div>
                </>
              )}
            </label>

            {errorMessage && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8, marginTop: 12,
                padding: "9px 12px", borderRadius: 6,
                background: "var(--rose-dim)", border: "1px solid rgba(244,63,94,0.3)",
                fontSize: 12, color: "var(--rose)",
              }}>
                <AlertCircle size={14} />
                <span>{errorMessage}</span>
              </div>
            )}
          </>
        ) : (
          /* Dataset loaded state */
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 8, padding: "10px 14px",
              background: "var(--emerald-dim)", border: "1px solid rgba(16,185,129,0.25)",
              borderRadius: 8, color: "var(--emerald)", fontSize: 12, fontWeight: 600,
            }}>
              <CheckCircle2 size={15} />
              Dataset validated and loaded successfully
            </div>

            <div style={{ background: "var(--bg-input)", borderRadius: 8,
              border: "1px solid var(--border-dim)", padding: "4px 12px" }}>
              <MetaRow label="Filename" value={datasetMeta.filename?.split("/").pop() || datasetMeta.filename} />
              <MetaRow label="File Size" value={formatBytes(datasetMeta.file_size)} />
              <MetaRow label="Dimensions" value={datasetMeta.shape?.join(" × ") || "—"} />
              <MetaRow label="Voxel Spacing" value={
                datasetMeta.voxel_spacing
                  ? datasetMeta.voxel_spacing.map((v) => v.toFixed(2)).join(" × ") + " mm"
                  : "—"
              } />
              <MetaRow label="Orientation" value={datasetMeta.orientation || "RAS+"} />
              <MetaRow label="Data Type" value={datasetMeta.data_type || "float32"} />
              <MetaRow label="Experiment ID" value={activeExperimentId || "—"} />
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <label style={{ cursor: "pointer" }}>
                <input type="file" accept=".nii,.gz" style={{ display: "none" }}
                  onChange={handleFileChange} disabled={isUploading} />
                <span className="btn btn-secondary">
                  <Upload size={13} />
                  Replace Dataset
                </span>
              </label>
            </div>

            {errorMessage && (
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "9px 12px", borderRadius: 6,
                background: "var(--rose-dim)", border: "1px solid rgba(244,63,94,0.3)",
                fontSize: 12, color: "var(--rose)",
              }}>
                <AlertCircle size={14} />
                <span>{errorMessage}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Info box */}
      <div style={{
        display: "flex", gap: 10, padding: "12px 14px",
        background: "var(--bg-card)", border: "1px solid var(--border-dim)",
        borderRadius: 8, fontSize: 12, color: "var(--text-muted)",
      }}>
        <Info size={14} style={{ color: "var(--accent)", flexShrink: 0, marginTop: 1 }} />
        <div>
          <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>Supported Formats: </span>
          NIfTI-1 and NIfTI-2 volumes (.nii, .nii.gz). 3D statistical maps and 4D BOLD timeseries are supported.
          Affine matrix, voxel spacing, and orientation are validated using NiBabel on upload.
          Data is never modified on disk.
        </div>
      </div>
    </div>
  );
}
