import React, { useState } from "react";
import { Crosshair, Play, Loader2, CheckCircle2, Info, Cpu } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function ROIPage() {
  const {
    activeExperimentId, stageStatuses, parameters,
    setParameters, setLayers, refreshResults,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);
  const [error, setError] = useState(null);

  const isCompleted = stageStatuses.segmentation === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setError(null);
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runSegmentation(activeExperimentId, {
        probabilityThreshold: parameters.probabilityThreshold,
      });
      setExecTime(((performance.now() - start) / 1000).toFixed(2));
      await refreshResults(activeExperimentId);
      setLayers((prev) => ({ ...prev, showRoi: true }));
    } catch (err) {
      setError(err.message || "ROI detection failed.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 680 }}>
      <div>
        <div className="section-title">ROI Segmentation</div>
        <div className="section-sub">3D U-Net region-of-interest prior detection</div>
      </div>

      {/* Model info */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Cpu size={13} /> Model Status</span>
          {isCompleted && <span className="badge badge-completed"><CheckCircle2 size={9} /> Completed</span>}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1,
          background: "var(--border-dim)", border: "1px solid var(--border-dim)",
          borderRadius: 7, overflow: "hidden", fontSize: 12 }}>
          {[
            ["Model",    "Configurable 3D U-Net (lightweight)"],
            ["Input",    "Preprocessed NIfTI volume (float32)"],
            ["Output",   "Binary ROI mask + continuous probability map"],
            ["Status",   isCompleted ? "Segmentation complete" : activeExperimentId ? "Ready to run" : "Awaiting dataset"],
          ].map(([k, v]) => (
            <div key={k} style={{ padding: "9px 12px", background: "var(--bg-card)" }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: "uppercase",
                letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: 3 }}>{k}</div>
              <div style={{ color: isCompleted && k === "Status" ? "var(--emerald)" :
                !activeExperimentId && k === "Status" ? "var(--amber)" : "var(--text-secondary)" }}>
                {v}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Config */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Crosshair size={13} /> Detection Parameters</span>
        </div>

        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>
              Probability Threshold
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "JetBrains Mono, monospace",
              color: "var(--teal)" }}>
              {parameters.probabilityThreshold}
            </span>
          </div>
          <input
            type="range" min="0.1" max="0.9" step="0.05"
            value={parameters.probabilityThreshold}
            onChange={(e) =>
              setParameters((p) => ({ ...p, probabilityThreshold: parseFloat(e.target.value) }))
            }
            style={{ width: "100%", accentColor: "var(--teal)" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between",
            fontSize: 10, color: "var(--text-dim)", marginTop: 3 }}>
            <span>0.1 (permissive)</span>
            <span>0.9 (restrictive)</span>
          </div>
        </div>

        {!activeExperimentId && (
          <div style={{
            display: "flex", gap: 8, padding: "9px 12px", borderRadius: 6,
            background: "var(--amber-dim)", border: "1px solid rgba(245,158,11,0.25)",
            fontSize: 12, color: "var(--amber)", marginBottom: 12,
          }}>
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>Upload and preprocess a dataset first.</span>
          </div>
        )}

        <button
          className="btn btn-primary btn-full"
          onClick={handleRun}
          disabled={!activeExperimentId || isRunning}
        >
          {isRunning ? (
            <><Loader2 size={13} className="animate-spin" /> Detecting ROI…</>
          ) : (
            <><Play size={13} style={{ fill: "currentColor" }} /> {isCompleted ? "Re-run ROI Detection" : "Run ROI Detection"}</>
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
        <strong style={{ color: "var(--text-secondary)" }}>About the 3D U-Net ROI prior: </strong>
        The U-Net produces a continuous probability map π<sub>i</sub><sup>DL</sup> ∈ [0, 1] which
        is later blended into the fcHMRF-LIS unary potentials via the λ<sub>DL</sub> weighting parameter.
        Setting λ<sub>DL</sub> = 0 recovers the original published fcHMRF-LIS without DL modification.
      </div>
    </div>
  );
}
