import React, { useRef, useEffect, useState, useCallback } from "react";
import { Rotate3d, RefreshCw, ZoomIn, ZoomOut, Layers, Eye, EyeOff } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

export default function Brain3DViewer() {
  const { activeExperimentId, meshData, load3DMesh } = useExperiment();
  const canvasRef = useRef(null);

  const [rotX, setRotX] = useState(25);
  const [rotY, setRotY] = useState(45);
  const [zoom, setZoom] = useState(1.2);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState(false);

  const [visibleLayers, setVisibleLayers] = useState({
    shell: true,
    roi: true,
    significant: true,
  });

  // Request 3D mesh when mounted or activeExperimentId changes
  useEffect(() => {
    if (activeExperimentId && !meshData) {
      setIsLoading(true);
      load3DMesh(activeExperimentId, 1).finally(() => setIsLoading(false));
    }
  }, [activeExperimentId, meshData, load3DMesh]);

  // Handle mouse drag for 360-degree rotation
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setRotY((prev) => (prev + dx * 0.5) % 360);
    setRotX((prev) => Math.max(-85, Math.min(85, prev - dy * 0.5)));
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY * -0.001;
    setZoom((prev) => Math.max(0.4, Math.min(3.0, prev + delta)));
  };

  // Render 3D isosurface meshes to Canvas
  const renderMesh = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Deep dark background
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, width, height);

    if (!meshData) return;

    // Convert angles to radians
    const radX = (rotX * Math.PI) / 180;
    const radY = (rotY * Math.PI) / 180;
    const cosX = Math.cos(radX), sinX = Math.sin(radX);
    const cosY = Math.cos(radY), sinY = Math.sin(radY);

    const centerX = width / 2;
    const centerY = height / 2;
    const baseScale = Math.min(width, height) * 0.02 * zoom;

    // Helper to rotate and project a vertex (x, y, z)
    const project = (x, y, z, offset = [16, 16, 16]) => {
      // Center coordinates around volume midpoint
      const cx = x - offset[0];
      const cy = y - offset[1];
      const cz = z - offset[2];

      // Rotate around Y (yaw)
      const x1 = cx * cosY + cz * sinY;
      const y1 = cy;
      const z1 = -cx * sinY + cz * cosY;

      // Rotate around X (pitch)
      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      const screenX = centerX + x2 * baseScale;
      const screenY = centerY - y2 * baseScale;
      return { sx: screenX, sy: screenY, depth: z2 };
    };

    const triangles = [];

    // Helper to add mesh triangles with depth
    const addMeshTriangles = (mesh, color, alpha) => {
      if (!mesh || !mesh.x || !mesh.i) return;
      const { x, y, z, i, j, k } = mesh;
      const count = i.length;

      for (let idx = 0; idx < count; idx++) {
        const v1 = project(x[i[idx]], y[i[idx]], z[i[idx]]);
        const v2 = project(x[j[idx]], y[j[idx]], z[j[idx]]);
        const v3 = project(x[k[idx]], y[k[idx]], z[k[idx]]);

        const avgDepth = (v1.depth + v2.depth + v3.depth) / 3;

        // Simple normal for shading
        const ax = v2.sx - v1.sx, ay = v2.sy - v1.sy;
        const bx = v3.sx - v1.sx, by = v3.sy - v1.sy;
        const cross = ax * by - ay * bx;

        triangles.push({
          v1,
          v2,
          v3,
          depth: avgDepth,
          color,
          alpha,
          isFacing: cross > 0,
        });
      }
    };

    if (visibleLayers.shell && meshData.brain_shell) {
      addMeshTriangles(meshData.brain_shell, [100, 116, 139], 0.25);
    }
    if (visibleLayers.roi && meshData.roi_mesh) {
      addMeshTriangles(meshData.roi_mesh, [6, 182, 212], 0.65);
    }
    if (visibleLayers.significant && meshData.significant_mesh) {
      addMeshTriangles(meshData.significant_mesh, [249, 115, 22], 0.85);
    }

    // Sort triangles Painter's algorithm (back to front)
    triangles.sort((a, b) => a.depth - b.depth);

    // Draw triangles
    for (const tri of triangles) {
      ctx.beginPath();
      ctx.moveTo(tri.v1.sx, tri.v1.sy);
      ctx.lineTo(tri.v2.sx, tri.v2.sy);
      ctx.lineTo(tri.v3.sx, tri.v3.sy);
      ctx.closePath();

      const [r, g, b] = tri.color;
      // Ambient + subtle directional lighting based on depth
      const lightFactor = Math.max(0.4, Math.min(1.2, 0.8 + tri.depth * 0.02));
      const lr = Math.min(255, Math.round(r * lightFactor));
      const lg = Math.min(255, Math.round(g * lightFactor));
      const lb = Math.min(255, Math.round(b * lightFactor));

      ctx.fillStyle = `rgba(${lr}, ${lg}, ${lb}, ${tri.alpha})`;
      ctx.fill();

      // Subtle wireframe for significant clusters & ROI
      if (tri.alpha > 0.5) {
        ctx.strokeStyle = `rgba(${lr}, ${lg}, ${lb}, 0.3)`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }, [meshData, rotX, rotY, zoom, visibleLayers]);

  useEffect(() => {
    renderMesh();
  }, [renderMesh]);

  const resetView = () => {
    setRotX(25);
    setRotY(45);
    setZoom(1.2);
  };

  return (
    <div className="space-y-3">
      {/* 3D Toolbar */}
      <div className="flex items-center justify-between text-xs text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
        <div className="flex items-center gap-2 font-mono">
          <Rotate3d className="w-4 h-4 text-orange-400" />
          <span className="font-semibold text-slate-200">3D Isosurface (Marching Cubes)</span>
          <span>•</span>
          <span>Drag to rotate 360° • Scroll to zoom</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Layer Toggles */}
          <button
            onClick={() => setVisibleLayers((p) => ({ ...p, shell: !p.shell }))}
            className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
              visibleLayers.shell
                ? "bg-slate-800 text-slate-200 border-slate-600"
                : "bg-slate-950 text-slate-500 border-slate-800"
            }`}
          >
            Brain Shell
          </button>
          <button
            onClick={() => setVisibleLayers((p) => ({ ...p, roi: !p.roi }))}
            className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
              visibleLayers.roi
                ? "bg-cyan-950 text-cyan-300 border-cyan-800"
                : "bg-slate-950 text-slate-500 border-slate-800"
            }`}
          >
            3D U-Net ROI
          </button>
          <button
            onClick={() => setVisibleLayers((p) => ({ ...p, significant: !p.significant }))}
            className={`px-2 py-0.5 rounded text-[11px] font-medium border transition-colors ${
              visibleLayers.significant
                ? "bg-orange-950 text-orange-300 border-orange-800"
                : "bg-slate-950 text-slate-500 border-slate-800"
            }`}
          >
            FDR Clusters
          </button>

          <button
            onClick={resetView}
            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 ml-1"
            title="Reset Orientation"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 3D Viewport */}
      <div
        className="bg-black border border-slate-800 rounded-lg p-2 flex items-center justify-center min-h-[440px] relative overflow-hidden cursor-grab active:cursor-grabbing shadow-inner"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onWheel={handleWheel}
      >
        <canvas
          ref={canvasRef}
          width={640}
          height={440}
          className="rounded shadow-2xl"
        />

        {isLoading && (
          <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center text-xs text-orange-400 gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Generating 3D Marching Cubes Surface Mesh...</span>
          </div>
        )}

        {!meshData && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
            Run segmentation and fcHMRF-LIS to generate 3D brain isosurfaces.
          </div>
        )}
      </div>
    </div>
  );
}
