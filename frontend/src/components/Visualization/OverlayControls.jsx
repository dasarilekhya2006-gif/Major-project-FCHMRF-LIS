import React from "react";
import { Eye, EyeOff, Layers, ChevronLeft, ChevronRight, Sliders } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function OverlayControls() {
  const {
    orientation,
    changeOrientation,
    sliceIdx,
    changeSlice,
    maxSlice,
    layers,
    setLayers,
    resultsData,
  } = useExperiment();

  const available = resultsData?.available_layers || {};

  const toggleLayer = (key) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const updateOpacity = (key, value) => {
    setLayers((prev) => ({ ...prev, [key]: parseFloat(value) }));
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-3 space-y-3">
      {/* Plane Selection & Slice Slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-orange-400" />
            View Plane & Slice
          </span>
          {/* Orientation Selector */}
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800 text-xs">
            {["axial", "coronal", "sagittal"].map((plane) => (
              <button
                key={plane}
                onClick={() => changeOrientation(plane)}
                className={`px-2.5 py-0.5 rounded capitalize font-medium transition-all ${
                  orientation === plane
                    ? "bg-orange-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {plane}
              </button>
            ))}
          </div>
        </div>

        {/* Slice Index Slider */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => changeSlice(Math.max(0, sliceIdx - 1))}
            disabled={sliceIdx <= 0}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
          <input
            type="range"
            min={0}
            max={maxSlice || 0}
            value={sliceIdx}
            onChange={(e) => changeSlice(parseInt(e.target.value, 10))}
            className="flex-1 accent-orange-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
          />
          <button
            onClick={() => changeSlice(Math.min(maxSlice || 0, sliceIdx + 1))}
            disabled={sliceIdx >= (maxSlice || 0)}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs  text-slate-300 min-w-[4rem] text-right">
            {sliceIdx} / {maxSlice || 0}
          </span>
        </div>
      </div>

      {/* Layer Visibility Toggles & Opacity */}
      <div className="pt-2 border-t border-slate-800/80 space-y-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
          Layer Visibility & Opacity
        </span>

        <div className="grid grid-cols-2 gap-2 text-xs">
          {/* Anatomical MRI */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-slate-200">
              <input
                type="checkbox"
                checked={layers.showMri}
                onChange={() => toggleLayer("showMri")}
                className="rounded bg-slate-800 border-slate-700 text-orange-500 focus:ring-0"
              />
              <span>Grayscale MRI</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={layers.mriOpacity}
              onChange={(e) => updateOpacity("mriOpacity", e.target.value)}
              className="w-16 accent-slate-400 h-1"
              title="MRI Brightness"
            />
          </div>

          {/* 3D U-Net ROI */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-cyan-300">
              <input
                type="checkbox"
                checked={layers.showRoi}
                onChange={() => toggleLayer("showRoi")}
                disabled={!available.roi_mask}
                className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0 disabled:opacity-40"
              />
              <span>3D U-Net ROI</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={layers.roiOpacity}
              onChange={(e) => updateOpacity("roiOpacity", e.target.value)}
              disabled={!available.roi_mask}
              className="w-16 accent-cyan-500 h-1 disabled:opacity-30"
              title="ROI Opacity"
            />
          </div>

          {/* FDR Significant Heatmap */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-orange-400 font-semibold">
              <input
                type="checkbox"
                checked={layers.showFdr}
                onChange={() => toggleLayer("showFdr")}
                disabled={!available.significant_mask}
                className="rounded bg-slate-800 border-slate-700 text-orange-500 focus:ring-0 disabled:opacity-40"
              />
              <span>FDR Clusters</span>
            </label>
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={layers.heatmapOpacity}
              onChange={(e) => updateOpacity("heatmapOpacity", e.target.value)}
              disabled={!available.significant_mask}
              className="w-16 accent-orange-500 h-1 disabled:opacity-30"
              title="Heatmap Opacity"
            />
          </div>

          {/* Statistical Map (|z|) */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-amber-300">
              <input
                type="checkbox"
                checked={layers.showStat}
                onChange={() => toggleLayer("showStat")}
                disabled={!available.statistical_map}
                className="rounded bg-slate-800 border-slate-700 text-amber-500 focus:ring-0 disabled:opacity-40"
              />
              <span>|Z| &gt; {layers.statThreshold}</span>
            </label>
            <input
              type="range"
              min="1.0"
              max="5.0"
              step="0.2"
              value={layers.statThreshold}
              onChange={(e) => setLayers((p) => ({ ...p, statThreshold: parseFloat(e.target.value) }))}
              disabled={!available.statistical_map}
              className="w-16 accent-amber-500 h-1 disabled:opacity-30"
              title="|Z| Threshold"
            />
          </div>

          {/* LIS Map */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-yellow-300">
              <input
                type="checkbox"
                checked={layers.showLis}
                onChange={() => toggleLayer("showLis")}
                disabled={!available.lis_map}
                className="rounded bg-slate-800 border-slate-700 text-yellow-500 focus:ring-0 disabled:opacity-40"
              />
              <span>LIS &le; {layers.lisThreshold}</span>
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.02"
              value={layers.lisThreshold}
              onChange={(e) => setLayers((p) => ({ ...p, lisThreshold: parseFloat(e.target.value) }))}
              disabled={!available.lis_map}
              className="w-16 accent-yellow-500 h-1 disabled:opacity-30"
              title="LIS Cutoff"
            />
          </div>

          {/* Baseline BH-FDR */}
          <div className="flex items-center justify-between p-1.5 rounded bg-slate-950/60 border border-slate-800">
            <label className="flex items-center gap-1.5 cursor-pointer text-blue-400">
              <input
                type="checkbox"
                checked={layers.showBaseline}
                onChange={() => toggleLayer("showBaseline")}
                disabled={!available.baseline_significant_mask}
                className="rounded bg-slate-800 border-slate-700 text-blue-500 focus:ring-0 disabled:opacity-40"
              />
              <span>Baseline BH</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
