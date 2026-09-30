import React from "react";
import {
  Database, SlidersHorizontal, Crosshair, BarChart2,
  Network, ShieldCheck, CheckCircle2, Circle,
  Loader2, XCircle, AlertTriangle,
} from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

const STAGE_META = [
  { key: "upload",       label: "Dataset",             Icon: Database,         color: "var(--teal)" },
  { key: "preprocess",   label: "Preprocessing",       Icon: SlidersHorizontal,color: "var(--accent)" },
  { key: "segmentation", label: "ROI Segmentation",    Icon: Crosshair,        color: "var(--accent)" },
  { key: "statistics",   label: "Statistical Analysis",Icon: BarChart2,        color: "var(--accent)" },
  { key: "fchmrf_lis",   label: "fcHMRF-LIS",          Icon: Network,          color: "var(--accent)" },
  { key: "fdr",          label: "FDR Control",         Icon: ShieldCheck,      color: "var(--emerald)" },
];

function StatusBadge({ status }) {
  if (status === "completed") return <span className="badge badge-completed"><CheckCircle2 size={9} /> Done</span>;
  if (status === "running")   return <span className="badge badge-running"><Loader2 size={9} className="animate-spin" /> Running</span>;
  if (status === "failed")    return <span className="badge badge-failed"><XCircle size={9} /> Failed</span>;
  return <span className="badge badge-pending"><Circle size={7} style={{ fill: "currentColor" }} /> Not Started</span>;
}

function StageCard({ meta, status, onClick }) {
  const { Icon } = meta;
  const isDone   = status === "completed";
  const isRun    = status === "running";
  const isFailed = status === "failed";

  let iconBg = "var(--bg-elevated)";
  let iconColor = "var(--text-muted)";
  let borderColor = "var(--border-subtle)";

  if (isDone)   { iconBg = "var(--emerald-dim)"; iconColor = "var(--emerald)"; borderColor = "rgba(16,185,129,0.3)"; }
  if (isRun)    { iconBg = "var(--accent-dim)";  iconColor = "var(--accent)";  borderColor = "var(--accent-border)"; }
  if (isFailed) { iconBg = "var(--rose-dim)";    iconColor = "var(--rose)";    borderColor = "rgba(244,63,94,0.3)"; }

  return (
    <div
      onClick={onClick}
      style={{
        background: "var(--bg-card)",
        border: `1px solid ${borderColor}`,
        borderRadius: 8,
        padding: "14px 16px",
        cursor: "pointer",
        transition: "border-color 0.15s, background 0.15s",
        display: "flex", flexDirection: "column", gap: 10,
      }}
      onMouseOver={(e) => { e.currentTarget.style.background = "var(--bg-elevated)"; }}
      onMouseOut={(e) => { e.currentTarget.style.background = "var(--bg-card)"; }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{
          width: 36, height: 36, borderRadius: 8,
          background: iconBg, display: "flex", alignItems: "center", justifyContent: "center",
          color: iconColor,
        }}>
          <Icon size={17} />
        </div>
        <StatusBadge status={status} />
      </div>
      <div>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>{meta.label}</div>
      </div>
    </div>
  );
}

export default function OverviewPage({ onNavigate }) {
  const { stageStatuses, datasetMeta, resultsData, backendOnline, pipelineStatus, activeExperimentId } = useExperiment();

  // Pipeline stages for horizontal flow
  const flowStages = [
    { key: "upload",       label: "Dataset",       Icon: Database },
    { key: "preprocess",   label: "Preprocess",    Icon: SlidersHorizontal },
    { key: "segmentation", label: "3D U-Net ROI",  Icon: Crosshair },
    { key: "statistics",   label: "Statistics",    Icon: BarChart2 },
    { key: "fchmrf_lis",   label: "fcHMRF-LIS",    Icon: Network },
    { key: "fdr",          label: "FDR Control",   Icon: ShieldCheck },
  ];

  const getFlowState = (key) => {
    if (key === "upload") return datasetMeta ? "done" : "pending";
    const s = stageStatuses[key];
    if (s === "completed") return "done";
    if (s === "running")   return "active";
    if (s === "failed")    return "failed";
    return "pending";
  };

  const discoveries = resultsData?.discoveries;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Page title */}
      <div>
        <div className="section-title">Research Overview</div>
        <div className="section-sub">Monitor pipeline state and navigate to individual stages</div>
      </div>

      {/* Pipeline failure / running banner */}
      {pipelineStatus.status === "running" && (
        <div style={{
          background: "rgba(59,130,212,0.1)",
          border: "1px solid var(--accent-border)",
          borderRadius: 8, padding: "10px 14px",
          display: "flex", alignItems: "center", gap: 8,
          fontSize: 12, color: "var(--accent)",
        }}>
          <Loader2 size={14} className="animate-spin" />
          <span><strong>Pipeline running</strong> — Stage: {pipelineStatus.stage} ({pipelineStatus.progress}%)</span>
        </div>
      )}
      {pipelineStatus.status === "failed" && (
        <div style={{
          background: "var(--rose-dim)",
          border: "1px solid rgba(244,63,94,0.3)",
          borderRadius: 8, padding: "10px 14px",
          display: "flex", alignItems: "center", gap: 8,
          fontSize: 12, color: "var(--rose)",
        }}>
          <AlertTriangle size={14} />
          <span><strong>Pipeline failed:</strong> {pipelineStatus.error || "Unknown error"}</span>
        </div>
      )}

      {/* Pipeline flow diagram */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            <Network size={13} /> Analysis Pipeline
          </span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {activeExperimentId ? `Exp: ${activeExperimentId}` : "No experiment loaded"}
          </span>
        </div>
        <div className="pipeline-flow">
          {flowStages.map((stage, idx) => {
            const state = getFlowState(stage.key);
            const prevState = idx > 0 ? getFlowState(flowStages[idx - 1].key) : "done";
            const connState = prevState === "done" ? "done" : prevState === "active" ? "active" : "";
            const { Icon } = stage;
            return (
              <React.Fragment key={stage.key}>
                {idx > 0 && <div className={`pipeline-connector${connState ? " " + connState : ""}`} />}
                <div className={`pipeline-stage${state === "done" ? " done" : state === "active" ? " active" : state === "failed" ? " failed" : ""}`}>
                  <div className="pipeline-stage-icon">
                    {state === "active" ? <Loader2 size={16} className="animate-spin" /> :
                     state === "done"   ? <CheckCircle2 size={16} /> :
                     state === "failed" ? <XCircle size={16} /> :
                     <Icon size={16} />}
                  </div>
                  <span className="pipeline-stage-label">{stage.label}</span>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Stage cards grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        <StageCard meta={STAGE_META[0]} status={datasetMeta ? "completed" : "pending"}
          onClick={() => onNavigate("dataset")} />
        <StageCard meta={STAGE_META[1]} status={stageStatuses.preprocess}
          onClick={() => onNavigate("preprocessing")} />
        <StageCard meta={STAGE_META[2]} status={stageStatuses.segmentation}
          onClick={() => onNavigate("roi")} />
        <StageCard meta={STAGE_META[3]} status={stageStatuses.statistics}
          onClick={() => onNavigate("statistics")} />
        <StageCard meta={STAGE_META[4]} status={stageStatuses.fchmrf_lis}
          onClick={() => onNavigate("fchmrf")} />
        <StageCard meta={STAGE_META[5]} status={stageStatuses.fdr}
          onClick={() => onNavigate("fdr")} />
      </div>

      {/* Quick-access summary */}
      {discoveries && (
        <div className="card">
          <div className="card-header">
            <span className="card-title"><ShieldCheck size={13} /> FDR Results Summary</span>
            <span className="badge badge-completed"><CheckCircle2 size={9} /> Available</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
            <div className="stat-card">
              <div className="stat-card-label">Proposed Discoveries</div>
              <div className="stat-card-value text-accent">
                {discoveries.proposed_discoveries?.toLocaleString() ?? "—"}
              </div>
              <div className="stat-card-sub">Adaptive fcHMRF-LIS</div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">Baseline Discoveries</div>
              <div className="stat-card-value" style={{ color: "var(--teal)" }}>
                {discoveries.baseline_discoveries?.toLocaleString() ?? "—"}
              </div>
              <div className="stat-card-sub">BH-FDR</div>
            </div>
            <div className="stat-card">
              <div className="stat-card-label">Method Agreement</div>
              <div className="stat-card-value text-emerald">
                {discoveries.agreement_dice !== undefined
                  ? (discoveries.agreement_dice * 100).toFixed(1) + "%"
                  : "—"}
              </div>
              <div className="stat-card-sub">Dice coefficient</div>
            </div>
          </div>
        </div>
      )}

      {/* No experiment state */}
      {!activeExperimentId && (
        <div className="card">
          <div className="empty-state">
            <Database size={44} className="empty-state-icon" />
            <div className="empty-state-title">No Experiment Loaded</div>
            <div className="empty-state-body">
              Upload a NIfTI dataset to create an experiment and begin analysis.
            </div>
            <button className="btn btn-primary" onClick={() => onNavigate("dataset")}>
              <Database size={13} /> Go to Dataset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
