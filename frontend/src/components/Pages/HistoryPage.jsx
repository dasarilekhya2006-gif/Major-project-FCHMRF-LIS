import React from "react";
import { History, RefreshCw, ArrowRight } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function HistoryPage({ onNavigate }) {
  const {
    experimentsList, activeExperimentId,
    selectExperiment, refreshExperimentsList,
  } = useExperiment();

  const handleLoad = async (expId) => {
    await selectExperiment(expId);
    onNavigate("overview");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <div className="section-title">Experiment History</div>
          <div className="section-sub">Previous runs and provenance log</div>
        </div>
        <button
          className="btn btn-secondary"
          onClick={refreshExperimentsList}
          style={{ marginTop: 4 }}
        >
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {experimentsList.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <History size={44} className="empty-state-icon" />
            <div className="empty-state-title">No Experiments Found</div>
            <div className="empty-state-body">
              Upload a NIfTI dataset to create your first experiment record.
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {experimentsList.map((exp) => {
            const isActive = exp.experiment_id === activeExperimentId;
            const stages = exp.stages || {};
            const stageKeys = ["preprocess", "segmentation", "statistics", "fchmrf_lis", "fdr"];
            const completedCount = stageKeys.filter((k) => stages[k] === "completed").length;

            return (
              <div
                key={exp.experiment_id}
                style={{
                  background: isActive ? "rgba(59,130,212,0.07)" : "var(--bg-card)",
                  border: `1px solid ${isActive ? "var(--accent-border)" : "var(--border-subtle)"}`,
                  borderRadius: 8,
                  padding: "14px 16px",
                  display: "flex", alignItems: "center",
                  justifyContent: "space-between", gap: 16,
                  transition: "all 0.15s",
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "JetBrains Mono, monospace",
                      color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis",
                      whiteSpace: "nowrap" }}>
                      {exp.experiment_id}
                    </span>
                    {isActive && <span className="badge badge-ready">Active</span>}
                    <span className="badge badge-pending" style={{ fontSize: 9 }}>
                      {exp.status || "ready"}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                    {exp.filename || "scan.nii.gz"} · Created: {new Date(exp.created_at).toLocaleString()}
                  </div>
                  {/* Stage progress dots */}
                  <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    {stageKeys.map((key) => {
                      const st = stages[key];
                      return (
                        <span
                          key={key}
                          title={`${key}: ${st || "pending"}`}
                          style={{
                            width: 8, height: 8, borderRadius: "50%",
                            background: st === "completed" ? "var(--emerald)"
                              : st === "running" ? "var(--accent)"
                              : st === "failed" ? "var(--rose)"
                              : "var(--border-strong)",
                            display: "inline-block",
                          }}
                        />
                      );
                    })}
                    <span style={{ fontSize: 10, color: "var(--text-dim)", marginLeft: 4 }}>
                      {completedCount}/{stageKeys.length} stages
                    </span>
                  </div>
                </div>

                <button
                  className="btn btn-secondary"
                  onClick={() => handleLoad(exp.experiment_id)}
                  style={{ flexShrink: 0 }}
                >
                  Load <ArrowRight size={12} />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
