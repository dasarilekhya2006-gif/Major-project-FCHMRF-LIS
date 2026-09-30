import React, { useState } from "react";
import { SlidersHorizontal, Play, Loader2, CheckCircle2, Info } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

function Toggle({ label, checked, onChange, description }) {
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 10,
      padding: "11px 14px",
      background: "var(--bg-input)", border: "1px solid var(--border-dim)",
      borderRadius: 7, marginBottom: 8,
    }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        style={{ marginTop: 2, flexShrink: 0, accentColor: "var(--accent)" }} />
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>{label}</div>
        {description && (
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{description}</div>
        )}
      </div>
    </div>
  );
}

export default function PreprocessingPage() {
  const {
    activeExperimentId, stageStatuses,
    refreshResults,
  } = useExperiment();

  const [normalize, setNormalize] = useState(true);
  const [brainMask, setBrainMask] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);
  const [error, setError] = useState(null);

  const isCompleted = stageStatuses.preprocess === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setError(null);
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runPreprocessing(activeExperimentId, {
        normalizeIntensity: normalize,
        applyBrainMask: brainMask,
      });
      setExecTime(((performance.now() - start) / 1000).toFixed(2));
      await refreshResults(activeExperimentId);
    } catch (err) {
      setError(err.message || "Preprocessing failed.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 640 }}>
      <div>
        <div className="section-title">Preprocessing</div>
        <div className="section-sub">Signal preparation before statistical analysis</div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title"><SlidersHorizontal size={13} /> Configuration</span>
          {isCompleted && <span className="badge badge-completed"><CheckCircle2 size={9} /> Completed</span>}
        </div>

        <Toggle
          label="Intensity Normalization (Z-score)"
          description="Standardizes voxel intensities to zero mean and unit variance within the brain mask."
          checked={normalize}
          onChange={setNormalize}
        />
        <Toggle
          label="Otsu Brain Mask Extraction"
          description="Computes a binary brain mask using Otsu thresholding to exclude non-brain voxels from analysis."
          checked={brainMask}
          onChange={setBrainMask}
        />

        {!activeExperimentId && (
          <div style={{
            display: "flex", gap: 8, padding: "9px 12px", borderRadius: 6,
            background: "var(--amber-dim)", border: "1px solid rgba(245,158,11,0.25)",
            fontSize: 12, color: "var(--amber)", marginBottom: 12,
          }}>
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>Upload a dataset first to enable preprocessing.</span>
          </div>
        )}

        <button
          className="btn btn-primary btn-full"
          onClick={handleRun}
          disabled={!activeExperimentId || isRunning}
          style={{ marginTop: 4 }}
        >
          {isRunning ? (
            <><Loader2 size={13} className="animate-spin" /> Processing…</>
          ) : (
            <><Play size={13} style={{ fill: "currentColor" }} /> {isCompleted ? "Re-run Preprocessing" : "Run Preprocessing"}</>
          )}
        </button>

        {error && (
          <div style={{ fontSize: 12, color: "var(--rose)", marginTop: 8, padding: "8px 10px",
            background: "var(--rose-dim)", border: "1px solid rgba(244,63,94,0.3)", borderRadius: 6 }}>
            Error: {error}
          </div>
        )}

        {execTime && (
          <div style={{ display: "flex", justifyContent: "space-between",
            marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border-dim)",
            fontSize: 11, fontFamily: "JetBrains Mono, monospace", color: "var(--text-muted)" }}>
            <span>Execution time</span>
            <span style={{ color: "var(--text-primary)" }}>{execTime} s</span>
          </div>
        )}
      </div>

      <div style={{
        padding: "12px 14px", background: "var(--bg-card)",
        border: "1px solid var(--border-dim)", borderRadius: 8,
        fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7,
      }}>
        <strong style={{ color: "var(--text-secondary)" }}>Pipeline position:</strong> Preprocessing runs after dataset
        upload and before 3D U-Net ROI detection. Output is a standardized volume stored in-memory for
        downstream statistical analysis. Source files are never modified on disk.
      </div>
    </div>
  );
}
