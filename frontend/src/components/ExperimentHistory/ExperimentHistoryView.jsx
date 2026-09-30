import React from "react";
import { History, RefreshCw, CheckCircle2, AlertCircle, ArrowRight } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function ExperimentHistoryView() {
  const {
    experimentsList,
    activeExperimentId,
    selectExperiment,
    refreshExperimentsList,
  } = useExperiment();

  const handleSelect = async (expId) => {
    await selectExperiment(expId);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <History className="w-3.5 h-3.5 text-orange-400" />
          Experiment History & Provenance Log
        </h3>
        <button
          onClick={refreshExperimentsList}
          className="btn-secondary text-xs py-1 px-2.5"
        >
          <RefreshCw className="w-3 h-3" />
          <span>Refresh List</span>
        </button>
      </div>

      <div className="space-y-2">
        {experimentsList.length === 0 ? (
          <div className="text-slate-500 py-8 text-center italic">
            No previous experiments recorded. Upload a NIfTI volume to start your first run.
          </div>
        ) : (
          experimentsList.map((exp) => {
            const isSelected = exp.experiment_id === activeExperimentId;
            const stages = exp.stages || {};
            const isAllCompleted = Object.values(stages).every((s) => s === "completed");

            return (
              <div
                key={exp.experiment_id}
                className={`p-3 rounded-lg border transition-all flex items-center justify-between gap-4 ${
                  isSelected
                    ? "bg-orange-950/20 border-orange-600/80 shadow-md"
                    : "bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60"
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-slate-200">{exp.experiment_id}</span>
                    {isSelected && (
                      <span className="px-1.5 py-0.2 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30 text-[10px] font-bold">
                        ACTIVE
                      </span>
                    )}
                    <span
                      className={`text-[10px] px-1.5 py-0.2 rounded border ${
                        isAllCompleted
                          ? "bg-emerald-950/60 text-emerald-400 border-emerald-800"
                          : "bg-slate-800 text-slate-400 border-slate-700"
                      }`}
                    >
                      {exp.status || "ready"}
                    </span>
                  </div>
                  <div className="text-slate-400 text-[11px]">
                    <span className="text-slate-500">Source: </span>
                    {exp.filename || "scan.nii.gz"}
                  </div>
                  <div className="text-slate-500 text-[10px]">
                    Created: {new Date(exp.created_at).toLocaleString()}
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {/* Stages indicator */}
                  <div className="flex items-center gap-1 text-[10px]">
                    {["preprocess", "segmentation", "statistics", "fchmrf_lis", "fdr"].map((stage) => {
                      const st = stages[stage];
                      return (
                        <span
                          key={stage}
                          title={`${stage}: ${st || "pending"}`}
                          className={`w-2 h-2 rounded-full ${
                            st === "completed"
                              ? "bg-emerald-500"
                              : st === "running"
                              ? "bg-orange-500 animate-ping"
                              : "bg-slate-700"
                          }`}
                        />
                      );
                    })}
                  </div>

                  <button
                    onClick={() => handleSelect(exp.experiment_id)}
                    className="btn-secondary text-xs py-1 px-3 text-slate-200 hover:text-white"
                  >
                    <span>Load</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
