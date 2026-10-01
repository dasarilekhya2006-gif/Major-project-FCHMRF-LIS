import React, { useState } from "react";
import { Network, CheckCircle2, Loader2, Play } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function FCHMRFLISPanel() {
  const {
    activeExperimentId,
    stageStatuses,
    refreshResults,
    parameters,
    setLayers,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);

  const isCompleted = stageStatuses.fchmrf_lis === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runFchmrfLis(activeExperimentId, {
        nominalAlpha: parameters.nominalAlpha,
        beta: parameters.beta,
        lambdaDl: parameters.lambdaDl,
        maxIterations: parameters.maxIterations,
      });
      const duration = ((performance.now() - start) / 1000).toFixed(2);
      setExecTime(duration);
      await refreshResults(activeExperimentId);
      // Turn on FDR and LIS layers
      setLayers((prev) => ({ ...prev, showFdr: true, showLis: true }));
    } catch (err) {
      console.error("fcHMRF-LIS execution failed:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Network className="w-3.5 h-3.5 text-orange-400" />
          6. fcHMRF-LIS
        </h3>
        {isCompleted && (
          <span className="badge badge-completed">
            <CheckCircle2 className="w-3 h-3" /> Done
          </span>
        )}
      </div>

      <p className="text-[11px] text-slate-400 leading-tight">
        Spatial bilateral filtering and mean-field variational inference for the Local Index of Significance.
      </p>

      <button
        onClick={handleRun}
        disabled={!activeExperimentId || isRunning}
        className="btn-primary w-full text-xs py-1.5"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-white" />
            <span>Computing Spatial LIS...</span>
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current text-white" />
            <span>{isCompleted ? "Re-run fcHMRF-LIS" : "Run fcHMRF-LIS"}</span>
          </>
        )}
      </button>

      {execTime && (
        <div className="text-[11px]  text-slate-400 flex justify-between pt-1 border-t border-slate-800/60">
          <span>Execution time:</span>
          <span className="text-slate-200">{execTime} s</span>
        </div>
      )}
    </div>
  );
}
