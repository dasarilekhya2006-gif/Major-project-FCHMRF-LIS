import React from "react";
import {
  LayoutDashboard,
  Database,
  SlidersHorizontal,
  Crosshair,
  BarChart2,
  Cpu,
  Network,
  ShieldCheck,
  ScanEye,
  GitCompare,
  TableProperties,
  History,
  BookOpen,
  FlaskConical,
} from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

const NAV_GROUPS = [
  {
    label: "Workspace",
    items: [
      { id: "overview",       label: "Overview",           Icon: LayoutDashboard },
      { id: "dataset",        label: "Dataset",            Icon: Database },
    ],
  },
  {
    label: "Pipeline",
    items: [
      { id: "preprocessing",  label: "Preprocessing",      Icon: SlidersHorizontal },
      { id: "roi",            label: "ROI Segmentation",   Icon: Crosshair },
      { id: "statistics",     label: "Statistical Analysis",Icon: BarChart2 },
      { id: "parameters",     label: "Parameters",         Icon: Cpu },
      { id: "fchmrf",         label: "fcHMRF-LIS",         Icon: Network },
      { id: "fdr",            label: "FDR Results",        Icon: ShieldCheck },
    ],
  },
  {
    label: "Analysis",
    items: [
      { id: "viewer",         label: "Neuroimaging Viewer",Icon: ScanEye },
      { id: "comparison",     label: "Baseline Comparison",Icon: GitCompare },
      { id: "metrics",        label: "Metrics",            Icon: TableProperties },
    ],
  },
  {
    label: "Research",
    items: [
      { id: "history",        label: "Experiment History", Icon: History },
      { id: "methodology",    label: "Methodology",        Icon: BookOpen },
    ],
  },
];

export default function WorkstationSidebar({ activePage, onNavigate }) {
  const { stageStatuses } = useExperiment();

  // Map page IDs to pipeline stage keys for status dots
  const pageStage = {
    preprocessing: "preprocess",
    roi:           "segmentation",
    statistics:    "statistics",
    fchmrf:        "fchmrf_lis",
    fdr:           "fdr",
  };

  const getStatusDot = (pageId) => {
    const stageKey = pageStage[pageId];
    if (!stageKey) return null;
    const status = stageStatuses[stageKey];
    const colors = {
      completed: "var(--emerald)",
      running:   "var(--accent)",
      failed:    "var(--rose)",
      pending:   "var(--border-strong)",
    };
    return (
      <span style={{
        width: 6, height: 6, borderRadius: "50%",
        background: colors[status] || colors.pending,
        flexShrink: 0, marginLeft: "auto",
        display: "inline-block",
      }} />
    );
  };

  return (
    <nav className="ws-sidebar">
      {/* Brand logo area (slim) */}
      <div style={{
        padding: "14px 16px 10px",
        borderBottom: "1px solid var(--border-dim)",
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <FlaskConical size={14} style={{ color: "var(--accent)", flexShrink: 0 }} />
        <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)",
          letterSpacing: "0.05em", textTransform: "uppercase" }}>
          Research Platform
        </span>
      </div>

      {/* Navigation groups */}
      {NAV_GROUPS.map((group) => (
        <div key={group.label}>
          <div className="nav-section-label">{group.label}</div>
          {group.items.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`nav-item${activePage === id ? " active" : ""}`}
              onClick={() => onNavigate(id)}
            >
              <Icon size={14} className="nav-icon" />
              <span style={{ flex: 1, fontSize: 13 }}>{label}</span>
              {getStatusDot(id)}
            </button>
          ))}
        </div>
      ))}

      {/* Footer */}
      <div style={{ marginTop: "auto", padding: "10px 16px 12px",
        borderTop: "1px solid var(--border-dim)", fontSize: 10,
        color: "var(--text-dim)", lineHeight: 1.5 }}>
        <div style={{ fontWeight: 600 }}>fcHMRF-LIS v1.0</div>
        <div>Research Prototype Only</div>
        <div>Not for Clinical Use</div>
      </div>
    </nav>
  );
}
