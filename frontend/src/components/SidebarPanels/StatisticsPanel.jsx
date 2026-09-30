import React, { useState } from "react";
import { BarChart2, CheckCircle2, Loader2, Play } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function StatisticsPanel() {
  const {
    activeExperimentId,
    stageStatuses,
    refreshResults,
    setLayers,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);

  const isCompleted = stageStatuses.statistics === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runStatistics(activeExperimentId);
      const duration = ((performance.now() - start) / 1000).toFixed(2);
      setExecTime(duration);
      await refreshResults(activeExperimentId);
      setLayers((prev) => ({ ...prev, showStat: true }));
    } catch (err) {
      console.error("Statistics calculation failed:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <BarChart2 className="w-3.5 h-3.5 text-amber-400" />
          4. Statistics
        </h3>
        {isCompleted && (
          <span className="badge badge-completed">
            <CheckCircle2 className="w-3 h-3" /> Done
          </span>
        )}
      </div>

      <p className="text-[11px] text-slate-400 leading-tight">
        Computes voxel-level two-sided test statistics (z-scores) and raw p-values within brain mask.
      </p>

      <button
        onClick={handleRun}
        disabled={!activeExperimentId || isRunning}
        className="btn-secondary w-full text-xs py-1.5"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
            <span>Computing Statistics...</span>
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current text-slate-400" />
            <span>{isCompleted ? "Re-run Statistics" : "Run Statistical Analysis"}</span>
          </>
        )}
      </button>

      {execTime && (
        <div className="text-[11px] font-mono text-slate-400 flex justify-between pt-1 border-t border-slate-800/60">
          <span>Execution time:</span>
          <span className="text-slate-200">{execTime} s</span>
        </div>
      )}
    </div>
  );
}
