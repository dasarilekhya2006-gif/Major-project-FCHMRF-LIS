import React from "react";
import { Check, Circle, Loader2, XCircle } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function PipelineStepper() {
  const { stageStatuses, pipelineStatus, activeTab, setActiveTab } = useExperiment();

  const stages = [
    { key: "upload", label: "Upload Data", tab: "2d" },
    { key: "preprocess", label: "Preprocess", tab: "2d" },
    { key: "segmentation", label: "3D U-Net ROI", tab: "2d" },
    { key: "statistics", label: "Statistics", tab: "2d" },
    { key: "fchmrf_lis", label: "fcHMRF-LIS", tab: "2d" },
    { key: "fdr", label: "Apply FDR", tab: "2d" },
    { key: "results", label: "View Results", tab: "2d" },
    { key: "comparison", label: "Compare Baseline", tab: "comparison" },
    { key: "export", label: "Export Results", tab: "export" },
  ];

  const getStageState = (stageKey) => {
    // If global pipeline running and matching stage
    if (pipelineStatus.status === "running") {
      const activeStage = pipelineStatus.stage?.toLowerCase() || "";
      if (activeStage.includes(stageKey)) {
        return "running";
      }
    }
    if (stageKey === "results" || stageKey === "comparison" || stageKey === "export") {
      return stageStatuses.fdr === "completed" ? "completed" : "pending";
    }
    return stageStatuses[stageKey] || "pending";
  };

  return (
    <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-2.5 overflow-x-auto">
      <div className="max-w-7xl mx-auto flex items-center gap-2 min-w-max">
        {stages.map((stage, idx) => {
          const state = getStageState(stage.key);

          let badgeIcon;
          let badgeClass;

          if (state === "completed") {
            badgeIcon = <Check className="w-3 h-3 text-emerald-400 stroke-[3]" />;
            badgeClass = "bg-emerald-950/60 text-emerald-300 border-emerald-800/80 hover:bg-emerald-900/60";
          } else if (state === "running") {
            badgeIcon = <Loader2 className="w-3 h-3 text-orange-400 animate-spin" />;
            badgeClass = "bg-orange-950/60 text-orange-300 border-orange-700 animate-pulse hover:bg-orange-900/60";
          } else if (state === "failed") {
            badgeIcon = <XCircle className="w-3 h-3 text-rose-400" />;
            badgeClass = "bg-rose-950/60 text-rose-300 border-rose-800 hover:bg-rose-900/60";
          } else {
            badgeIcon = <Circle className="w-2.5 h-2.5 text-slate-500 fill-slate-700/50" />;
            badgeClass = "bg-slate-950/40 text-slate-400 border-slate-800 hover:bg-slate-800/50";
          }

          return (
            <React.Fragment key={stage.key}>
              <button
                onClick={() => setActiveTab(stage.tab)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-all cursor-pointer ${badgeClass}`}
                title={`Stage: ${stage.label} (${state})`}
              >
                {badgeIcon}
                <span>{stage.label}</span>
              </button>
              {idx < stages.length - 1 && (
                <span className="text-slate-600 text-xs select-none">→</span>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
