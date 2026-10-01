import React, { useState } from "react";
import { Crosshair, CheckCircle2, Loader2, Play, Eye } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function SegmentationPanel() {
  const {
    activeExperimentId,
    stageStatuses,
    refreshResults,
    parameters,
    setParameters,
    setLayers,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);

  const isCompleted = stageStatuses.segmentation === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runSegmentation(activeExperimentId, {
        probabilityThreshold: parameters.probabilityThreshold,
      });
      const duration = ((performance.now() - start) / 1000).toFixed(2);
      setExecTime(duration);
      await refreshResults(activeExperimentId);
      // Ensure ROI layer is turned on
      setLayers((prev) => ({ ...prev, showRoi: true }));
    } catch (err) {
      console.error("ROI segmentation failed:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
          3. 3D U-Net ROI
        </h3>
        {isCompleted && (
          <span className="badge badge-completed">
            <CheckCircle2 className="w-3 h-3" /> Done
          </span>
        )}
      </div>

      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Probability Threshold:</span>
          <span className=" text-cyan-400 font-bold">{parameters.probabilityThreshold}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={parameters.probabilityThreshold}
          onChange={(e) =>
            setParameters((prev) => ({ ...prev, probabilityThreshold: parseFloat(e.target.value) }))
          }
          className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
        />
      </div>

      <button
        onClick={handleRun}
        disabled={!activeExperimentId || isRunning}
        className="btn-secondary w-full text-xs py-1.5"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
            <span>Detecting ROI...</span>
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current text-slate-400" />
            <span>{isCompleted ? "Re-run ROI Detection" : "Run ROI Detection"}</span>
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
