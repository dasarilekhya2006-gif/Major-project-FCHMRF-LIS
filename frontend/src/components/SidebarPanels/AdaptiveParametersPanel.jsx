import React from "react";
import { Cpu, Check } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function AdaptiveParametersPanel() {
  const { parameters, setParameters } = useExperiment();

  const handleModeChange = (isAdaptive) => {
    if (isAdaptive) {
      // Revert to Bayesian Optimization calibrated parameters
      setParameters((prev) => ({
        ...prev,
        adaptiveMode: true,
        nominalAlpha: 0.05,
        beta: 0.8,
        lambdaDl: 0.5,
        maxIterations: 20,
      }));
    } else {
      setParameters((prev) => ({
        ...prev,
        adaptiveMode: false,
      }));
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-orange-400" />
          5. Parameter Selection
        </h3>
        <span className="text-[10px] font-mono text-slate-400">
          {parameters.adaptiveMode ? "Adaptive (Auto)" : "Manual"}
        </span>
      </div>

      {/* Mode Selector */}
      <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-md border border-slate-800 text-xs">
        <button
          onClick={() => handleModeChange(false)}
          className={`py-1 rounded text-center font-medium transition-all ${
            !parameters.adaptiveMode
              ? "bg-slate-800 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Manual
        </button>
        <button
          onClick={() => handleModeChange(true)}
          className={`py-1 rounded text-center font-medium transition-all ${
            parameters.adaptiveMode
              ? "bg-orange-600 text-white shadow-sm"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Adaptive (BO)
        </button>
      </div>

      {/* Parameter Controls */}
      <div className="space-y-2.5 text-xs font-mono">
        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>FDR Nominal α:</span>
            <span className="text-orange-400 font-bold">{parameters.nominalAlpha}</span>
          </div>
          <input
            type="range"
            min="0.01"
            max="0.10"
            step="0.01"
            value={parameters.nominalAlpha}
            disabled={parameters.adaptiveMode}
            onChange={(e) =>
              setParameters((prev) => ({ ...prev, nominalAlpha: parseFloat(e.target.value) }))
            }
            className="w-full accent-orange-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer disabled:opacity-40"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>Spatial Smoothness β:</span>
            <span className="text-orange-400 font-bold">{parameters.beta}</span>
          </div>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={parameters.beta}
            disabled={parameters.adaptiveMode}
            onChange={(e) =>
              setParameters((prev) => ({ ...prev, beta: parseFloat(e.target.value) }))
            }
            className="w-full accent-orange-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer disabled:opacity-40"
          />
        </div>

        <div>
          <div className="flex justify-between text-slate-400 mb-1">
            <span>DL Prior Weight λ:</span>
            <span className="text-cyan-400 font-bold">{parameters.lambdaDl}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={parameters.lambdaDl}
            disabled={parameters.adaptiveMode}
            onChange={(e) =>
              setParameters((prev) => ({ ...prev, lambdaDl: parseFloat(e.target.value) }))
            }
            className="w-full accent-cyan-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer disabled:opacity-40"
          />
        </div>
      </div>

      {parameters.adaptiveMode && (
        <div className="text-[11px] text-slate-400 bg-slate-950/80 border border-slate-800/80 rounded p-2 leading-relaxed">
          <span className="text-orange-400 font-semibold">Adaptive Mode: </span>
          Parameters are automatically selected via Bayesian optimization on validation data to prevent data leakage.
        </div>
      )}
    </div>
  );
}
