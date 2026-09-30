import React from "react";
import {
  BrainCircuit,
  RefreshCw,
  Settings,
  Circle,
} from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function WorkstationHeader() {
  const {
    backendOnline,
    activeExperimentId,
    experimentsList,
    selectExperiment,
    refreshExperimentsList,
    datasetMeta,
  } = useExperiment();

  return (
    <header className="ws-header">
      {/* Brand */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{
          width: 34, height: 34, borderRadius: 8,
          background: "var(--accent-dim)",
          border: "1px solid var(--accent-border)",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--accent)",
        }}>
          <BrainCircuit size={18} />
        </div>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            fcHMRF-LIS Neuroimaging Research Platform
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>
            Spatial False Discovery Rate Control in Neuroimaging
          </div>
        </div>
      </div>

      {/* Right side status strip */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {/* API Status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 5,
          padding: "4px 10px",
          borderRadius: 6,
          border: "1px solid",
          fontSize: 11, fontWeight: 600,
          background: backendOnline ? "rgba(16,185,129,0.08)" : "rgba(244,63,94,0.08)",
          borderColor: backendOnline ? "rgba(16,185,129,0.25)" : "rgba(244,63,94,0.25)",
          color: backendOnline ? "var(--emerald)" : "var(--rose)",
        }}>
          <Circle size={7} fill="currentColor" />
          <span>{backendOnline ? "API Connected" : "API Offline"}</span>
        </div>

        {/* Experiment selector */}
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          padding: "4px 10px",
          borderRadius: 6,
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-input)",
          fontSize: 11,
        }}>
          <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>EXP:</span>
          <select
            value={activeExperimentId || ""}
            onChange={(e) => e.target.value && selectExperiment(e.target.value)}
            style={{
              background: "transparent", border: "none", outline: "none",
              color: "var(--text-primary)", fontSize: 11, fontWeight: 600,
              fontFamily: "JetBrains Mono, monospace",
              cursor: "pointer", maxWidth: 160,
            }}
          >
            {experimentsList.length === 0 ? (
              <option value="">— No experiments —</option>
            ) : (
              experimentsList.map((exp) => (
                <option key={exp.experiment_id} value={exp.experiment_id}
                  style={{ background: "#111827" }}>
                  {exp.experiment_id}
                </option>
              ))
            )}
          </select>
          <button
            onClick={refreshExperimentsList}
            title="Refresh"
            style={{ background: "none", border: "none", color: "var(--text-muted)",
              cursor: "pointer", display: "flex", alignItems: "center" }}
          >
            <RefreshCw size={11} />
          </button>
        </div>

        {/* Dataset status pill */}
        <div style={{
          padding: "4px 10px",
          borderRadius: 6,
          border: "1px solid var(--border-subtle)",
          background: "var(--bg-input)",
          fontSize: 11,
          color: datasetMeta ? "var(--teal)" : "var(--text-muted)",
        }}>
          {datasetMeta
            ? `Dataset: ${datasetMeta.filename?.split("/").pop()?.slice(0, 20) || "loaded"}`
            : "No dataset loaded"}
        </div>

        {/* Settings placeholder */}
        <button style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--border-subtle)",
          borderRadius: 6,
          padding: "5px 7px",
          cursor: "pointer",
          color: "var(--text-muted)",
          display: "flex", alignItems: "center",
        }}>
          <Settings size={14} />
        </button>
      </div>
    </header>
  );
}
