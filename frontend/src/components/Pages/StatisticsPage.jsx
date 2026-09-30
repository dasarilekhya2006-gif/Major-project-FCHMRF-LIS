import React, { useState } from "react";
import { BarChart2, Play, Loader2, CheckCircle2, Info } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

function InfoRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between",
      padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontSize: 12 }}>
      <span style={{ color: "var(--text-muted)" }}>{label}</span>
      <span style={{ color: "var(--text-secondary)", fontFamily: "JetBrains Mono, monospace" }}>{value}</span>
    </div>
  );
}

export default function StatisticsPage() {
  const {
    activeExperimentId, stageStatuses,
    setLayers, refreshResults,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);
  const [error, setError] = useState(null);

  const isCompleted = stageStatuses.statistics === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setError(null);
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runStatistics(activeExperimentId);
      setExecTime(((performance.now() - start) / 1000).toFixed(2));
      await refreshResults(activeExperimentId);
      setLayers((prev) => ({ ...prev, showStat: true }));
    } catch (err) {
      setError(err.message || "Statistics computation failed.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 640 }}>
      <div>
        <div className="section-title">Statistical Analysis</div>
        <div className="section-sub">Voxel-level hypothesis testing and z-score computation</div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title"><BarChart2 size={13} /> Test Configuration</span>
          {isCompleted && <span className="badge badge-completed"><CheckCircle2 size={9} /> Completed</span>}
        </div>

        <div style={{ background: "var(--bg-input)", borderRadius: 7,
          border: "1px solid var(--border-dim)", padding: "4px 12px", marginBottom: 16 }}>
          <InfoRow label="Test Type"        value="Two-sided voxel-wise test" />
          <InfoRow label="Output Statistic" value="Z-score (standard normal)" />
          <InfoRow label="P-values"         value="Raw (uncorrected)" />
          <InfoRow label="Domain"           value="Brain-mask restricted" />
        </div>

        <p style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7, marginBottom: 14 }}>
          Computes voxel-level two-sided test statistics and raw p-values within the brain mask.
          Output z-scores are passed directly to the fcHMRF-LIS spatial inference engine.
        </p>

        {!activeExperimentId && (
          <div style={{
            display: "flex", gap: 8, padding: "9px 12px", borderRadius: 6,
            background: "var(--amber-dim)", border: "1px solid rgba(245,158,11,0.25)",
            fontSize: 12, color: "var(--amber)", marginBottom: 12,
          }}>
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>Upload a dataset and run preprocessing first.</span>
          </div>
        )}

        <button
          className="btn btn-primary btn-full"
          onClick={handleRun}
          disabled={!activeExperimentId || isRunning}
        >
          {isRunning ? (
            <><Loader2 size={13} className="animate-spin" /> Computing…</>
          ) : (
            <><Play size={13} style={{ fill: "currentColor" }} /> {isCompleted ? "Re-run Statistical Analysis" : "Run Statistical Analysis"}</>
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
    </div>
  );
}
