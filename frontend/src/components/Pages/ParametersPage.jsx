import React from "react";
import { Cpu, Info } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

function Slider({ label, min, max, step, value, onChange, color, disabled }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, fontFamily: "JetBrains Mono, monospace",
          color: disabled ? "var(--text-muted)" : color || "var(--accent)" }}>
          {value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        style={{ width: "100%", accentColor: color || "var(--accent)",
          opacity: disabled ? 0.4 : 1 }}
      />
    </div>
  );
}

export default function ParametersPage() {
  const { parameters, setParameters } = useExperiment();

  const handleModeChange = (mode) => {
    if (mode === "adaptive") {
      setParameters((p) => ({
        ...p, adaptiveMode: true,
        nominalAlpha: 0.05, beta: 0.8, lambdaDl: 0.5, maxIterations: 20,
      }));
    } else {
      setParameters((p) => ({ ...p, adaptiveMode: false }));
    }
  };

  const modes = [
    { id: "manual",   label: "Manual" },
    { id: "adaptive", label: "Adaptive Bayesian Optimization" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 660 }}>
      <div>
        <div className="section-title">Parameter Selection</div>
        <div className="section-sub">Configure fcHMRF-LIS hyperparameters for spatial inference</div>
      </div>

      {/* Mode selector */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Cpu size={13} /> Selection Mode</span>
          <span style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
            color: "var(--text-muted)" }}>
            {parameters.adaptiveMode ? "Adaptive (BO)" : "Manual"}
          </span>
        </div>

        <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
          {modes.map(({ id, label }) => {
            const isActive = (id === "adaptive") === parameters.adaptiveMode;
            return (
              <button
                key={id}
                className={`btn ${isActive ? "btn-primary" : "btn-secondary"}`}
                onClick={() => handleModeChange(id)}
              >
                {label}
              </button>
            );
          })}
        </div>

        {parameters.adaptiveMode && (
          <div style={{
            display: "flex", gap: 8, padding: "10px 12px", borderRadius: 6,
            background: "var(--accent-dim)", border: "1px solid var(--accent-border)",
            fontSize: 12, color: "var(--accent)", marginBottom: 14,
          }}>
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>
              Adaptive parameters are selected using validation data via Bayesian Optimization (Gaussian
              Process + Expected Improvement) to reduce parameter-selection leakage.
              Parameters are not tuned on test data.
            </span>
          </div>
        )}
      </div>

      {/* Parameter controls */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">fcHMRF-LIS Parameters</span>
          {parameters.adaptiveMode && (
            <span className="badge badge-running">Auto</span>
          )}
        </div>

        <Slider
          label="FDR Nominal α"
          min={0.01} max={0.10} step={0.01}
          value={parameters.nominalAlpha}
          onChange={(v) => setParameters((p) => ({ ...p, nominalAlpha: v }))}
          color="var(--emerald)"
          disabled={parameters.adaptiveMode}
        />
        <Slider
          label="Spatial Coupling Strength β"
          min={0.1} max={2.0} step={0.1}
          value={parameters.beta}
          onChange={(v) => setParameters((p) => ({ ...p, beta: v }))}
          color="var(--accent)"
          disabled={parameters.adaptiveMode}
        />
        <Slider
          label="DL Prior Weight λ"
          min={0.0} max={1.0} step={0.05}
          value={parameters.lambdaDl}
          onChange={(v) => setParameters((p) => ({ ...p, lambdaDl: v }))}
          color="var(--teal)"
          disabled={parameters.adaptiveMode}
        />

        <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "10px 0 0",
          borderTop: "1px solid var(--border-dim)", lineHeight: 1.6 }}>
          <strong style={{ color: "var(--text-secondary)" }}>λ = 0:</strong> Recovers original published fcHMRF-LIS
          (no DL modification). &nbsp;
          <strong style={{ color: "var(--text-secondary)" }}>λ = 1:</strong> Full DL prior influence. &nbsp;
          <strong style={{ color: "var(--text-secondary)" }}>β:</strong> Controls strength of spatial smoothness.
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{
        padding: "12px 14px", background: "var(--bg-card)",
        border: "1px solid var(--border-dim)", borderRadius: 8,
        fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7,
      }}>
        <strong style={{ color: "var(--amber)" }}>Scientific neutrality: </strong>
        No assumption is made that adaptive parameter selection is necessarily superior to
        pre-specified defaults. Empirical experiments determine whether adaptive optimization
        provides measurable benefit over manual configuration.
      </div>
    </div>
  );
}
