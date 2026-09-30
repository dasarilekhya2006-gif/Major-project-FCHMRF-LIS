import React from "react";
import { Activity, Brain, CheckCircle2, ChevronDown, Play, RefreshCw, Server, AlertCircle } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

export default function Header() {
  const {
    activeExperimentId,
    selectExperiment,
    experimentsList,
    refreshExperimentsList,
    pipelineStatus,
    setPipelineStatus,
    parameters,
    backendOnline,
  } = useExperiment();

  const handleRunPipeline = async () => {
    if (!activeExperimentId) return;
    try {
      setPipelineStatus({
        status: "running",
        stage: "preprocess",
        progress: 10,
        error: null,
      });

      await pipelineApi.runCompletePipeline({
        experiment_id: activeExperimentId,
        run_segmentation: true,
        run_statistics: true,
        run_fchmrf_lis: true,
        run_fdr: true,
        nominal_alpha: parameters.nominalAlpha,
        beta: parameters.beta,
        lambda_dl: parameters.lambdaDl,
      });
    } catch (err) {
      setPipelineStatus({
        status: "failed",
        stage: "error",
        progress: 0,
        error: err.message || "Failed to trigger pipeline.",
      });
    }
  };

  return (
    <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-30 px-4 py-2.5 shadow-md">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Title & Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center shadow-lg shadow-orange-500/20 text-white">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-semibold text-slate-100 text-sm tracking-tight">
                fcHMRF-LIS Neuroimaging Analysis
              </h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                Research Only
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Deep Learning Prior & Adaptive Spatial False Discovery Rate
            </p>
          </div>
        </div>

        {/* Experiment Selector & Global Status */}
        <div className="flex items-center gap-3">
          {/* Active Experiment Dropdown */}
          <div className="flex items-center gap-1.5 bg-slate-950/80 border border-slate-700/80 rounded-md px-2.5 py-1 text-xs">
            <span className="text-slate-400 font-medium">Exp:</span>
            <select
              value={activeExperimentId || ""}
              onChange={(e) => selectExperiment(e.target.value)}
              className="bg-transparent text-slate-200 font-mono text-xs focus:outline-none cursor-pointer"
            >
              {experimentsList.length === 0 ? (
                <option value="" disabled>No experiments found</option>
              ) : (
                experimentsList.map((exp) => (
                  <option key={exp.experiment_id} value={exp.experiment_id} className="bg-slate-900 text-slate-200">
                    {exp.experiment_id} ({exp.filename?.slice(0, 15) || "scan"})
                  </option>
                ))
              )}
            </select>
            <button
              onClick={refreshExperimentsList}
              title="Refresh experiments"
              className="text-slate-400 hover:text-slate-200 ml-1"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>

          {/* Backend Health Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono border ${
              backendOnline
                ? "bg-emerald-950/40 text-emerald-400 border-emerald-800/50"
                : "bg-rose-950/40 text-rose-400 border-rose-800/50"
            }`}
          >
            <Server className="w-3 h-3" />
            <span>{backendOnline ? "API Online" : "API Offline"}</span>
          </div>

          {/* Master Run Pipeline Button */}
          <button
            onClick={handleRunPipeline}
            disabled={!activeExperimentId || pipelineStatus.status === "running"}
            className="btn-primary text-xs py-1.5 px-3"
          >
            {pipelineStatus.status === "running" ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Running Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run Complete Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
