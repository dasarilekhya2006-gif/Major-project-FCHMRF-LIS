import React from "react";
import { ShieldCheck, CheckCircle2 } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function FDRPanel() {
  const { resultsData, stageStatuses, parameters, setParameters } = useExperiment();

  const isCompleted = stageStatuses.fdr === "completed" || stageStatuses.fchmrf_lis === "completed";
  const discoveries = resultsData?.discoveries;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          7. FDR Analysis
        </h3>
        {isCompleted && (
          <span className="badge badge-completed">
            <CheckCircle2 className="w-3 h-3" /> Controlled
          </span>
        )}
      </div>

      <div className="space-y-1.5 font-mono text-xs">
        <div className="flex justify-between text-slate-400">
          <span>Nominal Level α:</span>
          <span className="text-emerald-400 font-bold">{parameters.nominalAlpha}</span>
        </div>

        {discoveries ? (
          <div className="bg-slate-950/70 border border-slate-800/80 rounded p-2 space-y-1 mt-2">
            <div className="flex justify-between text-slate-300">
              <span className="text-slate-500">Proposed Discoveries:</span>
              <span className="text-orange-400 font-bold">
                {discoveries.proposed_discoveries?.toLocaleString() ?? 0}
              </span>
            </div>
            <div className="flex justify-between text-slate-300">
              <span className="text-slate-500">Baseline (BH) Discoveries:</span>
              <span className="text-blue-400 font-bold">
                {discoveries.baseline_discoveries?.toLocaleString() ?? 0}
              </span>
            </div>
            {discoveries.agreement_dice !== undefined && discoveries.agreement_dice !== null && (
              <div className="flex justify-between text-slate-300">
                <span className="text-slate-500">Method Agreement (Dice):</span>
                <span className="text-emerald-400 font-bold">
                  {(discoveries.agreement_dice * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className="text-[11px] text-slate-500 italic">
            Run fcHMRF-LIS to compute FDR-controlled discoveries.
          </div>
        )}
      </div>
    </div>
  );
}
