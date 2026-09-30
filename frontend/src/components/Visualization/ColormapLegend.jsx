import React from "react";

export default function ColormapLegend() {
  return (
    <div className="bg-slate-950/80 border border-slate-800 rounded-md p-2 flex items-center justify-between text-[11px] font-mono text-slate-400 gap-3">
      <div className="flex items-center gap-1.5 font-medium text-slate-300">
        <span className="w-2 h-2 rounded-full bg-orange-500" />
        <span>Significance Heatmap:</span>
      </div>
      <div className="flex items-center gap-2">
        <span>Low (1 - LIS ≈ 0)</span>
        <div
          className="w-32 h-2.5 rounded border border-slate-700"
          style={{
            background: "linear-gradient(to right, rgb(180, 40, 0), rgb(240, 100, 0), rgb(255, 185, 20), rgb(255, 245, 80))",
          }}
        />
        <span className="text-amber-300 font-bold">High (1 - LIS ≈ 1)</span>
      </div>
      <div className="text-slate-500 text-[10px]">
        Overlaid on Grayscale Anatomical MRI
      </div>
    </div>
  );
}
