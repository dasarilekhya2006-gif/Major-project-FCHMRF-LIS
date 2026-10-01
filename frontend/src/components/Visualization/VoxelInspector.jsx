import React from "react";
import { Crosshair, MapPin, Activity } from "lucide-react";
import { voxelToWorld } from "../../utils/colormap";
import { useExperiment } from "../../state/ExperimentContext";

export default function VoxelInspector({ hoveredVoxel }) {
  const { experimentDetails } = useExperiment();
  const affine = experimentDetails?.affine;

  if (!hoveredVoxel) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-400  flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-slate-500">
          <Crosshair className="w-3.5 h-3.5" />
          Hover over brain slice to inspect voxel metrics
        </span>
        <span className="text-[10px] text-slate-600">Preserving raw scientific values</span>
      </div>
    );
  }

  const { x, y, z, mriVal, statVal, lisVal, isSig, inRoi } = hoveredVoxel;
  const [worldX, worldY, worldZ] = voxelToWorld(x, y, z, affine);

  return (
    <div className="bg-slate-950 border border-slate-700/80 rounded-lg p-2.5 text-xs  shadow-lg space-y-1.5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1 text-cyan-400 font-bold">
            <Crosshair className="w-3.5 h-3.5" />
            Grid: ({x}, {y}, {z})
          </span>
          <span className="flex items-center gap-1 text-slate-400 text-[11px]">
            <MapPin className="w-3 h-3 text-slate-500" />
            World: ({worldX}, {worldY}, {worldZ}) mm
          </span>
        </div>
        <div className="flex items-center gap-2">
          {inRoi && (
            <span className="px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-800 text-[10px] font-bold">
              ROI
            </span>
          )}
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
              isSig
                ? "bg-orange-950/80 text-orange-300 border-orange-700"
                : "bg-slate-900 text-slate-500 border-slate-800"
            }`}
          >
            {isSig ? "FDR SIGNIFICANT" : "NON-SIGNIFICANT"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 text-[11px] pt-0.5">
        <div className="bg-slate-900/80 p-1 rounded border border-slate-800">
          <span className="text-slate-500 block text-[9px] uppercase">MRI Intensity</span>
          <span className="text-slate-200 font-bold">{mriVal !== null ? mriVal.toFixed(3) : "—"}</span>
        </div>
        <div className="bg-slate-900/80 p-1 rounded border border-slate-800">
          <span className="text-slate-500 block text-[9px] uppercase">Z-Score</span>
          <span className="text-amber-400 font-bold">{statVal !== null ? statVal.toFixed(3) : "—"}</span>
        </div>
        <div className="bg-slate-900/80 p-1 rounded border border-slate-800">
          <span className="text-slate-500 block text-[9px] uppercase">LIS Value</span>
          <span className="text-orange-400 font-bold">{lisVal !== null ? lisVal.toFixed(4) : "—"}</span>
        </div>
        <div className="bg-slate-900/80 p-1 rounded border border-slate-800">
          <span className="text-slate-500 block text-[9px] uppercase">1 - LIS (Prob)</span>
          <span className="text-yellow-300 font-bold">
            {lisVal !== null ? (1.0 - lisVal).toFixed(4) : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
