import React, { useState } from "react";
import { Sliders, CheckCircle2, Loader2, Play } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function PreprocessingPanel() {
  const {
    activeExperimentId,
    stageStatuses,
    refreshResults,
    experimentDetails,
    orientation,
    sliceIdx,
  } = useExperiment();

  const [normalize, setNormalize] = useState(true);
  const [brainMask, setBrainMask] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);

  const isCompleted = stageStatuses.preprocess === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setIsRunning(true);
    const start = performance.now();
    try {
      await pipelineApi.runPreprocessing(activeExperimentId, {
        normalizeIntensity: normalize,
        applyBrainMask: brainMask,
      });
      const duration = ((performance.now() - start) / 1000).toFixed(2);
      setExecTime(duration);
      await refreshResults(activeExperimentId);
    } catch (err) {
      console.error("Preprocessing failed:", err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Sliders className="w-3.5 h-3.5 text-orange-400" />
          2. Preprocessing
        </h3>
        {isCompleted && (
          <span className="badge badge-completed">
            <CheckCircle2 className="w-3 h-3" /> Done
          </span>
        )}
      </div>

      <div className="space-y-2 text-xs">
        <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={normalize}
            onChange={(e) => setNormalize(e.target.checked)}
            className="rounded bg-slate-800 border-slate-700 text-orange-500 focus:ring-0"
          />
          <span>Intensity Normalization (Z-score)</span>
        </label>
        <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={brainMask}
            onChange={(e) => setBrainMask(e.target.checked)}
            className="rounded bg-slate-800 border-slate-700 text-orange-500 focus:ring-0"
          />
          <span>Otsu Brain Mask Extraction</span>
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={!activeExperimentId || isRunning}
        className="btn-secondary w-full text-xs py-1.5"
      >
        {isRunning ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin text-orange-400" />
            <span>Preprocessing...</span>
          </>
        ) : (
          <>
            <Play className="w-3 h-3 fill-current text-slate-400" />
            <span>{isCompleted ? "Re-run Preprocessing" : "Run Preprocessing"}</span>
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
