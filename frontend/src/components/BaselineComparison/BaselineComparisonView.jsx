import React, { useRef, useEffect, useState } from "react";
import { GitCompare, Layers, ChevronLeft, ChevronRight } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import { renderSliceToCanvas } from "../../utils/canvasRenderer";

export default function BaselineComparisonView() {
  const {
    sliceData,
    orientation,
    changeOrientation,
    sliceIdx,
    changeSlice,
    maxSlice,
    resultsData,
  } = useExperiment();

  const baselineCanvasRef = useRef(null);
  const proposedCanvasRef = useRef(null);
  const diffCanvasRef = useRef(null);

  const [compMode, setCompMode] = useState("side-by-side"); // side-by-side vs difference

  // Render Baseline slice (Grayscale MRI + Baseline BH discoveries in Royal Blue)
  useEffect(() => {
    if (!baselineCanvasRef.current || !sliceData) return;
    renderSliceToCanvas(baselineCanvasRef.current, sliceData, {
      showMri: true,
      mriOpacity: 0.9,
      showRoi: false,
      showFdr: false,
      showBaseline: true,
    });
  }, [sliceData]);

  // Render Proposed slice (Grayscale MRI + fcHMRF-LIS discoveries in Orange-Yellow)
  useEffect(() => {
    if (!proposedCanvasRef.current || !sliceData) return;
    renderSliceToCanvas(proposedCanvasRef.current, sliceData, {
      showMri: true,
      mriOpacity: 0.9,
      showRoi: true,
      roiOpacity: 0.35,
      showFdr: true,
      showBaseline: false,
    });
  }, [sliceData]);

  // Render Difference map (Consensus: Green, Proposed-Only: Orange, Baseline-Only: Blue)
  useEffect(() => {
    if (!diffCanvasRef.current || !sliceData || !sliceData.mri_slice) return;

    const canvas = diffCanvasRef.current;
    const mri = sliceData.mri_slice;
    const height = mri.length;
    const width = mri[0]?.length || 0;
    if (height === 0 || width === 0) return;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    const sig = sliceData.significant_slice || [];
    const baseSig = sliceData.baseline_significant_slice || [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;

        // Base MRI gray
        const gray = Math.round((mri[y][x] / 2.0) * 120);
        let r = gray, g = gray, b = gray, a = 255;

        const isProp = sig[y] && sig[y][x] > 0;
        const isBase = baseSig[y] && baseSig[y][x] > 0;

        if (isProp && isBase) {
          // Consensus: Emerald Green [16, 185, 129]
          r = 16; g = 185; b = 129;
        } else if (isProp && !isBase) {
          // Proposed-Only: Orange [249, 115, 22]
          r = 249; g = 115; b = 22;
        } else if (!isProp && isBase) {
          // Baseline-Only: Royal Blue [59, 130, 246]
          r = 59; g = 130; b = 246;
        }

        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = a;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [sliceData]);

  const discoveries = resultsData?.discoveries || {};

  return (
    <div className="space-y-3">
      {/* Plane Selection & Navigation Toolbar */}
      <div className="flex items-center justify-between text-xs text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
        <div className="flex items-center gap-2 ">
          <GitCompare className="w-4 h-4 text-orange-400" />
          <span className="font-semibold text-slate-200">Baseline (BH-FDR) vs Proposed (Adaptive fcHMRF-LIS)</span>
        </div>

        {/* View Mode & Slices */}
        <div className="flex items-center gap-3">
          <div className="flex bg-slate-950 p-0.5 rounded border border-slate-800 text-xs">
            <button
              onClick={() => setCompMode("side-by-side")}
              className={`px-2.5 py-0.5 rounded font-medium transition-all ${
                compMode === "side-by-side"
                  ? "bg-orange-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setCompMode("difference")}
              className={`px-2.5 py-0.5 rounded font-medium transition-all ${
                compMode === "difference"
                  ? "bg-orange-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Difference Map
            </button>
          </div>

          <div className="flex items-center gap-1.5 ">
            <button
              onClick={() => changeSlice(Math.max(0, sliceIdx - 1))}
              disabled={sliceIdx <= 0}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <span className="text-slate-200">{sliceIdx} / {maxSlice || 0}</span>
            <button
              onClick={() => changeSlice(Math.min(maxSlice || 0, sliceIdx + 1))}
              disabled={sliceIdx >= (maxSlice || 0)}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Viewport Panels */}
      {compMode === "side-by-side" ? (
        <div className="grid grid-cols-2 gap-3">
          {/* Baseline BH-FDR */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs ">
              <span className="font-semibold text-blue-400 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                Standard BH-FDR Baseline
              </span>
              <span className="text-slate-400">
                Discoveries: <strong className="text-blue-300">{discoveries.baseline_discoveries ?? 0}</strong>
              </span>
            </div>
            <div className="bg-black rounded p-2 flex items-center justify-center min-h-[300px]">
              <canvas
                ref={baselineCanvasRef}
                style={{ width: "100%", height: "100%", maxHeight: "280px", objectFit: "contain", imageRendering: "pixelated" }}
                className="border border-slate-900 rounded shadow-md"
              />
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Voxel-wise independent multiple testing without spatial regularization. Vulnerable to isolated false discoveries.
            </p>
          </div>

          {/* Proposed Adaptive fcHMRF-LIS */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs ">
              <span className="font-semibold text-orange-400 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-orange-500" />
                Proposed Adaptive fcHMRF-LIS
              </span>
              <span className="text-slate-400">
                Discoveries: <strong className="text-orange-300">{discoveries.proposed_discoveries ?? 0}</strong>
              </span>
            </div>
            <div className="bg-black rounded p-2 flex items-center justify-center min-h-[300px]">
              <canvas
                ref={proposedCanvasRef}
                style={{ width: "100%", height: "100%", maxHeight: "280px", objectFit: "contain", imageRendering: "pixelated" }}
                className="border border-slate-900 rounded shadow-md"
              />
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Dense bilateral spatial dependency modeling + 3D U-Net structural prior. Recovers spatially contiguous clusters.
            </p>
          </div>
        </div>
      ) : (
        /* Difference Map */
        <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-3">
          <div className="flex items-center justify-between text-xs ">
            <span className="font-semibold text-slate-200">
              Discrepancy Classification Map
            </span>
            <div className="flex items-center gap-4 text-[11px]">
              <span className="flex items-center gap-1 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                Consensus ({discoveries.consensus_discoveries ?? 0})
              </span>
              <span className="flex items-center gap-1 text-orange-400">
                <span className="w-2 h-2 rounded-full bg-orange-500" />
                Proposed Only ({discoveries.proposed_only_discoveries ?? 0})
              </span>
              <span className="flex items-center gap-1 text-blue-400">
                <span className="w-2 h-2 rounded-full bg-blue-500" />
                Baseline Only ({discoveries.baseline_only_discoveries ?? 0})
              </span>
            </div>
          </div>

          <div className="bg-black rounded p-4 flex items-center justify-center min-h-[340px]">
            <canvas
              ref={diffCanvasRef}
              style={{ width: "100%", height: "100%", maxHeight: "320px", objectFit: "contain", imageRendering: "pixelated" }}
              className="border border-slate-900 rounded shadow-md"
            />
          </div>
        </div>
      )}


    </div>
  );
}
