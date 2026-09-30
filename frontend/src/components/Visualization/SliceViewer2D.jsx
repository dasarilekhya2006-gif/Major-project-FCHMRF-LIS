import React, { useRef, useEffect, useState, useCallback } from "react";
import { Maximize2, RotateCcw, ZoomIn, ZoomOut, Download } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import { renderSliceToCanvas } from "../../utils/canvasRenderer";
import VoxelInspector from "./VoxelInspector";
import ColormapLegend from "./ColormapLegend";

export default function SliceViewer2D({ externalScale }) {
  const { sliceData, orientation, sliceIdx, layers } = useExperiment();
  const canvasRef = useRef(null);

  const [internalScale, setInternalScale] = useState(2.5);
  const scale = externalScale !== undefined ? externalScale : internalScale;
  const [hoveredVoxel, setHoveredVoxel] = useState(null);
  const [crosshairPos, setCrosshairPos] = useState(null);

  // Redraw canvas whenever sliceData or layers change
  useEffect(() => {
    if (!canvasRef.current || !sliceData) return;
    renderSliceToCanvas(canvasRef.current, sliceData, {
      showMri: layers.showMri,
      mriOpacity: layers.mriOpacity,
      showRoi: layers.showRoi,
      roiOpacity: layers.roiOpacity,
      showStat: layers.showStat,
      statThreshold: layers.statThreshold,
      showLis: layers.showLis,
      lisThreshold: layers.lisThreshold,
      showFdr: layers.showFdr,
      heatmapOpacity: layers.heatmapOpacity,
      showGroundTruth: layers.showGroundTruth,
      showBaseline: layers.showBaseline,
      crosshair: crosshairPos,
    });
  }, [sliceData, layers, crosshairPos]);

  // Handle mouse move to extract voxel metrics under cursor
  const handleMouseMove = useCallback(
    (e) => {
      const canvas = canvasRef.current;
      if (!canvas || !sliceData || !sliceData.mri_slice) return;

      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      const voxelX = Math.floor((clientX / rect.width) * canvas.width);
      const voxelY = Math.floor((clientY / rect.height) * canvas.height);

      if (
        voxelX >= 0 &&
        voxelX < canvas.width &&
        voxelY >= 0 &&
        voxelY < canvas.height
      ) {
        setCrosshairPos({ x: voxelX, y: voxelY });

        // Map slice coordinates back to full 3D coordinates (x, y, z)
        let x = voxelX,
          y = voxelY,
          z = sliceIdx;
        if (orientation === "coronal") {
          x = voxelX;
          y = sliceIdx;
          z = voxelY;
        } else if (orientation === "sagittal") {
          x = sliceIdx;
          y = voxelX;
          z = voxelY;
        }

        const mriVal = sliceData.mri_slice?.[voxelY]?.[voxelX] ?? null;
        const statVal = sliceData.statistical_slice?.[voxelY]?.[voxelX] ?? null;
        const lisVal = sliceData.lis_slice?.[voxelY]?.[voxelX] ?? null;
        const isSig = sliceData.significant_slice?.[voxelY]?.[voxelX] > 0;
        const inRoi = sliceData.roi_slice?.[voxelY]?.[voxelX] > 0;

        setHoveredVoxel({
          x,
          y,
          z,
          mriVal,
          statVal,
          lisVal,
          isSig,
          inRoi,
        });
      }
    },
    [sliceData, orientation, sliceIdx]
  );

  const handleMouseLeave = () => {
    setHoveredVoxel(null);
    setCrosshairPos(null);
  };

  const handleDownloadSnapshot = () => {
    if (!canvasRef.current) return;
    const link = document.createElement("a");
    link.download = `slice_${orientation}_${sliceIdx}.png`;
    link.href = canvasRef.current.toDataURL("image/png");
    link.click();
  };

  return (
    <div className="space-y-3">
      {/* Top Slice Info Bar & Zoom Controls */}
      <div className="flex items-center justify-between text-xs text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
        <div className="flex items-center gap-2 font-mono">
          <span className="font-semibold text-slate-200 capitalize">{orientation} Slice</span>
          <span>•</span>
          <span>Index: <strong className="text-orange-400">{sliceIdx}</strong></span>
          {sliceData && (
            <>
              <span>•</span>
              <span>Resolution: {sliceData.width} × {sliceData.height} px</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setInternalScale((s) => Math.max(1.0, s - 0.5))}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-[11px] font-mono w-10 text-center text-slate-300">
            {scale.toFixed(1)}x
          </span>
          <button
            onClick={() => setInternalScale((s) => Math.min(5.0, s + 0.5))}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setInternalScale(2.5)}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 ml-1"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleDownloadSnapshot}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-orange-400 ml-1"
            title="Save High-Res Snapshot"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Slice Canvas Viewport */}
      <div className="bg-black border border-slate-800 rounded-lg p-4 flex items-center justify-center min-h-[380px] overflow-hidden relative shadow-inner">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{
            transform: `scale(${scale})`,
            imageRendering: "pixelated",
            cursor: "crosshair",
            transition: "transform 0.1s ease-out",
          }}
          className="border border-slate-900 rounded shadow-2xl"
        />
        {!sliceData && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
            No volume loaded. Upload a NIfTI scan or select an experiment to begin.
          </div>
        )}
      </div>

      {/* Temperature Colormap Bar */}
      <ColormapLegend />

      {/* Interactive Voxel Inspector */}
      <VoxelInspector hoveredVoxel={hoveredVoxel} />
    </div>
  );
}
