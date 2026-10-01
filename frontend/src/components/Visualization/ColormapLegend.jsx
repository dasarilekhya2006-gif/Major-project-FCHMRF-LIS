import React from "react";

export default function ColormapLegend() {
  return (
    <div className="bg-slate-950/80 border border-slate-800 rounded-md p-2 flex items-center justify-between text-[11px]  text-slate-400 gap-3">
      <div className="flex items-center gap-1.5 font-medium text-slate-300">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        <span>Abnormality Heatmap:</span>
      </div>
      <div className="flex items-center gap-2">
        <span>Low</span>
        <div
          className="w-32 h-2.5 rounded border border-slate-700"
          style={{
            background: "linear-gradient(to right, rgb(0, 0, 255), rgb(0, 255, 255), rgb(0, 255, 0), rgb(255, 255, 0), rgb(255, 0, 0))",
          }}
        />
        <span className="text-red-400 font-bold">High</span>
      </div>
      <div className="text-slate-500 text-[10px]">
        Overlaid on Grayscale MRI (Red indicates higher chance of abnormalities)
      </div>
    </div>
  );
}
