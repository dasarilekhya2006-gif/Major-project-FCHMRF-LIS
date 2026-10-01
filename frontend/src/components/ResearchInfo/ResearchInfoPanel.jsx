import React from "react";
import { BookOpen, Shield, Sigma, Cpu } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function ResearchInfoPanel() {
  const { datasetMeta, parameters } = useExperiment();

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-3 text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5 text-orange-400" />
          Research Methodology & Formal Specifications
        </h3>
        <span className="text-[11px]  text-slate-400">Adaptive fcHMRF-LIS</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5 bg-slate-950/70 p-3 rounded-lg border border-slate-800">
          <span className="font-bold text-slate-200 flex items-center gap-1">
            <Sigma className="w-3.5 h-3.5 text-orange-400" />
            1. Spatial fcHMRF Model
          </span>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Models dense pairwise spatial dependencies between voxels using bilateral kernel weights:
          </p>
          <div className="bg-slate-900 p-2 rounded text-[10px]  text-orange-300">
            w_ij = exp( - ||p_i - p_j||² / (2 σ_p²) - ||z_i - z_j||² / (2 σ_z²) )
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Mean-field variational updates enable linear-time dense filtering without graph truncation.
          </p>
        </div>

        <div className="space-y-1.5 bg-slate-950/70 p-3 rounded-lg border border-slate-800">
          <span className="font-bold text-slate-200 flex items-center gap-1">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            2. Local Index of Significance (LIS) & FDR
          </span>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            LIS estimates posterior probability of the null hypothesis at each voxel:
          </p>
          <div className="bg-slate-900 p-2 rounded text-[10px]  text-emerald-300">
            LIS_i = P(θ_i = 0 | Z)
          </div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            FDR control selects the largest cutoff k satisfying (1/k) ∑ LIS_(i) ≤ α, maximizing detection power while bounding false discoveries.
          </p>
        </div>
      </div>
    </div>
  );
}
