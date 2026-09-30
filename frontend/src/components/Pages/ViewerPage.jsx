import React, { useState } from "react";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import SliceViewer2D from "../Visualization/SliceViewer2D";

function LayerToggle({ label, checked, onChange, disabled, color }) {
  return (
    <label style={{
      display: "flex", alignItems: "center", gap: 7, cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.4 : 1,
      padding: "7px 10px",
      background: "var(--bg-input)", borderRadius: 6,
      border: `1px solid ${checked && !disabled ? color || "var(--accent-border)" : "var(--border-dim)"}`,
      fontSize: 12, fontWeight: checked ? 600 : 400,
      color: checked && !disabled ? color || "var(--accent)" : "var(--text-muted)",
      transition: "all 0.15s", marginBottom: 6,
    }}>
      <input
        type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        disabled={disabled} style={{ accentColor: color || "var(--accent)" }}
      />
      {label}
    </label>
  );
}

function OpacitySlider({ label, value, onChange, disabled, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4,
        fontSize: 11, color: "var(--text-muted)" }}>
        <span>{label}</span>
        <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{value.toFixed(2)}</span>
      </div>
      <input
        type="range" min="0.1" max="1.0" step="0.05" value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        style={{ width: "100%", accentColor: color || "var(--accent)",
          opacity: disabled ? 0.4 : 1 }}
      />
    </div>
  );
}

export default function ViewerPage() {
  const {
    orientation, changeOrientation,
    sliceIdx, changeSlice, maxSlice,
    layers, setLayers,
    sliceData, resultsData,
  } = useExperiment();

  const [scale, setScale] = useState(2.5);

  const available = resultsData?.available_layers || {};

  const toggle = (key) => setLayers((p) => ({ ...p, [key]: !p[key] }));
  const setOp  = (key, v) => setLayers((p) => ({ ...p, [key]: v }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <div className="section-title">Neuroimaging Viewer</div>
        <div className="section-sub">Interactive 2D slice viewer with multi-layer overlay</div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 16, alignItems: "start" }}>
        {/* Left: viewer */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Plane + slice toolbar */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "8px 12px",
            background: "var(--bg-card)", border: "1px solid var(--border-dim)",
            borderRadius: 8, gap: 12,
          }}>
            {/* Orientation */}
            <div style={{ display: "flex", gap: 3 }}>
              {["axial", "coronal", "sagittal"].map((plane) => (
                <button
                  key={plane}
                  onClick={() => changeOrientation(plane)}
                  style={{
                    padding: "4px 10px", borderRadius: 5, border: "1px solid",
                    fontSize: 11, fontWeight: 600, cursor: "pointer",
                    background: orientation === plane ? "var(--accent)" : "var(--bg-input)",
                    borderColor: orientation === plane ? "var(--accent-hover)" : "var(--border-dim)",
                    color: orientation === plane ? "#fff" : "var(--text-muted)",
                    textTransform: "capitalize",
                  }}
                >
                  {plane.charAt(0).toUpperCase() + plane.slice(1)}
                </button>
              ))}
            </div>

            {/* Slice slider */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1, maxWidth: 280 }}>
              <button
                onClick={() => changeSlice(Math.max(0, sliceIdx - 1))}
                disabled={sliceIdx <= 0}
                style={{ background: "var(--bg-input)", border: "1px solid var(--border-dim)",
                  borderRadius: 5, padding: "3px 6px", cursor: "pointer",
                  color: "var(--text-muted)", display: "flex", opacity: sliceIdx <= 0 ? 0.4 : 1 }}
              >
                <ChevronLeft size={14} />
              </button>
              <input
                type="range" min={0} max={maxSlice || 0} value={sliceIdx}
                onChange={(e) => changeSlice(parseInt(e.target.value, 10))}
                style={{ flex: 1, accentColor: "var(--accent)" }}
              />
              <button
                onClick={() => changeSlice(Math.min(maxSlice || 0, sliceIdx + 1))}
                disabled={sliceIdx >= (maxSlice || 0)}
                style={{ background: "var(--bg-input)", border: "1px solid var(--border-dim)",
                  borderRadius: 5, padding: "3px 6px", cursor: "pointer",
                  color: "var(--text-muted)", display: "flex",
                  opacity: sliceIdx >= (maxSlice || 0) ? 0.4 : 1 }}
              >
                <ChevronRight size={14} />
              </button>
              <span style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
                color: "var(--text-secondary)", minWidth: 52, textAlign: "right" }}>
                {sliceIdx} / {maxSlice || 0}
              </span>
            </div>

            {/* Zoom controls */}
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <button onClick={() => setScale((s) => Math.max(1.0, s - 0.5))}
                style={{ background: "var(--bg-input)", border: "1px solid var(--border-dim)",
                  borderRadius: 5, padding: "4px 6px", cursor: "pointer",
                  color: "var(--text-muted)", display: "flex" }}>
                <ZoomOut size={13} />
              </button>
              <span style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
                color: "var(--text-secondary)", minWidth: 34, textAlign: "center" }}>
                {scale.toFixed(1)}x
              </span>
              <button onClick={() => setScale((s) => Math.min(5.0, s + 0.5))}
                style={{ background: "var(--bg-input)", border: "1px solid var(--border-dim)",
                  borderRadius: 5, padding: "4px 6px", cursor: "pointer",
                  color: "var(--text-muted)", display: "flex" }}>
                <ZoomIn size={13} />
              </button>
              <button onClick={() => setScale(2.5)}
                style={{ background: "var(--bg-input)", border: "1px solid var(--border-dim)",
                  borderRadius: 5, padding: "4px 6px", cursor: "pointer",
                  color: "var(--text-muted)", display: "flex" }}>
                <RotateCcw size={13} />
              </button>
            </div>
          </div>

          {/* Canvas viewer — includes colormap legend and voxel inspector internally */}
          <SliceViewer2D externalScale={scale} />
        </div>

        {/* Right: controls panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Layer visibility */}
          <div className="card-sm">
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase",
              letterSpacing: "0.07em", color: "var(--text-muted)", marginBottom: 10 }}>
              Layer Visibility
            </div>
            <LayerToggle label="Anatomical MRI" checked={layers.showMri}
              onChange={(v) => setLayers((p) => ({ ...p, showMri: v }))}
              color="var(--text-secondary)" />
            <LayerToggle label="3D U-Net ROI" checked={layers.showRoi}
              onChange={(v) => setLayers((p) => ({ ...p, showRoi: v }))}
              disabled={!available.roi_mask}
              color="var(--teal)" />
            <LayerToggle label="FDR Clusters" checked={layers.showFdr}
              onChange={(v) => setLayers((p) => ({ ...p, showFdr: v }))}
              disabled={!available.significant_mask}
              color="var(--accent)" />
            <LayerToggle label="LIS Map" checked={layers.showLis}
              onChange={(v) => setLayers((p) => ({ ...p, showLis: v }))}
              disabled={!available.lis_map}
              color="var(--amber)" />
            <LayerToggle label="Baseline BH-FDR" checked={layers.showBaseline}
              onChange={(v) => setLayers((p) => ({ ...p, showBaseline: v }))}
              disabled={!available.baseline_significant_mask}
              color="var(--teal)" />
          </div>

          {/* Opacity controls */}
          <div className="card-sm">
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase",
              letterSpacing: "0.07em", color: "var(--text-muted)", marginBottom: 10 }}>
              Opacity
            </div>
            <OpacitySlider label="MRI Brightness" value={layers.mriOpacity}
              onChange={(v) => setOp("mriOpacity", v)} />
            <OpacitySlider label="ROI Overlay" value={layers.roiOpacity}
              onChange={(v) => setOp("roiOpacity", v)}
              disabled={!available.roi_mask} color="var(--teal)" />
            <OpacitySlider label="FDR Heatmap" value={layers.heatmapOpacity}
              onChange={(v) => setOp("heatmapOpacity", v)}
              disabled={!available.significant_mask} color="var(--accent)" />
          </div>

          {/* Thresholds */}
          <div className="card-sm">
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase",
              letterSpacing: "0.07em", color: "var(--text-muted)", marginBottom: 10 }}>
              Thresholds
            </div>
            <div style={{ marginBottom: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4,
                fontSize: 11, color: "var(--text-muted)" }}>
                <span>|Z| threshold</span>
                <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{layers.statThreshold.toFixed(1)}</span>
              </div>
              <input type="range" min="1.0" max="5.0" step="0.2"
                value={layers.statThreshold}
                onChange={(e) => setLayers((p) => ({ ...p, statThreshold: parseFloat(e.target.value) }))}
                style={{ width: "100%", accentColor: "var(--amber)" }} />
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4,
                fontSize: 11, color: "var(--text-muted)" }}>
                <span>LIS threshold</span>
                <span style={{ fontFamily: "JetBrains Mono, monospace" }}>{layers.lisThreshold.toFixed(2)}</span>
              </div>
              <input type="range" min="0.01" max="0.5" step="0.02"
                value={layers.lisThreshold}
                onChange={(e) => setLayers((p) => ({ ...p, lisThreshold: parseFloat(e.target.value) }))}
                style={{ width: "100%", accentColor: "var(--teal)" }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
