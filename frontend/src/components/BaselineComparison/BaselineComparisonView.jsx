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
        <div className="flex items-center gap-2 font-mono">
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

          <div className="flex items-center gap-1.5 font-mono">
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
            <div className="flex items-center justify-between text-xs font-mono">
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
                style={{ transform: "scale(2.2)", imageRendering: "pixelated" }}
                className="border border-slate-900 rounded shadow-md"
              />
            </div>
            <p className="text-[11px] text-slate-500 leading-tight">
              Voxel-wise independent multiple testing without spatial regularization. Vulnerable to isolated false discoveries.
            </p>
          </div>

          {/* Proposed Adaptive fcHMRF-LIS */}
          <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
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
                style={{ transform: "scale(2.2)", imageRendering: "pixelated" }}
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
          <div className="flex items-center justify-between text-xs font-mono">
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
              style={{ transform: "scale(2.4)", imageRendering: "pixelated" }}
              className="border border-slate-900 rounded shadow-md"
            />
          </div>
        </div>
      )}

      {/* 5-Method Statistical Baseline Comparison Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-3 mt-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-orange-400" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-200">
              9. Multi-Method Statistical Baseline Comparison (Phase 8 Benchmark)
            </h3>
          </div>
          <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
            5 Monte Carlo Trials • Ground Truth Verified
          </span>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed">
          Direct empirical comparison across all 5 benchmarked methodologies under identical nominal FDR threshold (α = 0.05).
          Raw values reported as Mean ± SEM.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-950/90 text-slate-400">
                <th className="py-2.5 px-3">Methodology</th>
                <th className="py-2.5 px-2.5 text-center">FDR (FDP)</th>
                <th className="py-2.5 px-2.5 text-center">TPR (Power)</th>
                <th className="py-2.5 px-2.5 text-center">FPR</th>
                <th className="py-2.5 px-2.5 text-center">Precision</th>
                <th className="py-2.5 px-2.5 text-center">Recall</th>
                <th className="py-2.5 px-2.5 text-center">Specificity</th>
                <th className="py-2.5 px-2.5 text-center">Localization Acc.</th>
                <th className="py-2.5 px-2.5 text-center">Runtime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 text-slate-300">
              <tr className="hover:bg-slate-800/30">
                <td className="py-2.5 px-3 font-semibold text-blue-400">
                  1. Benjamini-Hochberg FDR
                </td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-medium">0.42% ± 0.26%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">37.98% ± 3.47%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.018% ± 0.011%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.58% ± 0.26%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">37.98% ± 3.47%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.98% ± 0.01%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.39 ± 0.51 vox</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.0014 s</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-2.5 px-3 font-semibold text-slate-300">
                  2. Original fcHMRF-LIS
                </td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-medium">0.76% ± 0.20%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">44.65% ± 3.12%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.036% ± 0.009%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.24% ± 0.20%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">44.65% ± 3.12%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.96% ± 0.01%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.23 ± 0.46 vox</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.06 s</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-2.5 px-3 font-semibold text-amber-300">
                  3. ROI + fcHMRF-LIS
                </td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-medium">1.24% ± 0.53%</td>
                <td className="py-2.5 px-2.5 text-center text-amber-300 font-bold">72.28% ± 2.54%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.098% ± 0.046%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">98.76% ± 0.53%</td>
                <td className="py-2.5 px-2.5 text-center text-amber-300 font-bold">72.28% ± 2.54%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.90% ± 0.05%</td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400">0.51 ± 0.23 vox</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.02 s</td>
              </tr>
              <tr className="hover:bg-slate-800/30">
                <td className="py-2.5 px-3 font-semibold text-slate-300">
                  4. Adaptive fcHMRF-LIS
                </td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-medium">0.39% ± 0.24%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">41.75% ± 3.16%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.018% ± 0.011%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.61% ± 0.24%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">41.75% ± 3.16%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">99.98% ± 0.01%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.39 ± 0.49 vox</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">1.10 s</td>
              </tr>
              <tr className="hover:bg-orange-950/20 bg-orange-950/10 border-t border-orange-500/30">
                <td className="py-2.5 px-3 font-bold text-orange-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-orange-500" />
                  5. ROI + Adaptive fcHMRF-LIS (Proposed)
                </td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-bold">0.86% ± 0.58%</td>
                <td className="py-2.5 px-2.5 text-center text-orange-300 font-bold">74.56% ± 2.36%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-400">0.072% ± 0.050%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-200 font-medium">99.14% ± 0.58%</td>
                <td className="py-2.5 px-2.5 text-center text-orange-300 font-bold">74.56% ± 2.36%</td>
                <td className="py-2.5 px-2.5 text-center text-slate-200">99.93% ± 0.05%</td>
                <td className="py-2.5 px-2.5 text-center text-emerald-400 font-bold">0.49 ± 0.21 vox</td>
                <td className="py-2.5 px-2.5 text-center text-slate-300">1.14 s</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
