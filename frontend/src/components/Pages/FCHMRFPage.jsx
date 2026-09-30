import React, { useState } from "react";
import { Network, Play, Loader2, CheckCircle2, Info, AlertTriangle, Activity } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as pipelineApi from "../../api/pipelineApi";

function StepBlock({ number, title, formula, description }) {
  return (
    <div style={{
      display: "flex", gap: 12, padding: "10px 14px",
      background: "var(--bg-input)", borderRadius: 7,
      border: "1px solid var(--border-dim)", marginBottom: 8,
    }}>
      <div style={{
        width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
        background: "var(--accent-dim)", border: "1px solid var(--accent-border)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, fontWeight: 700, color: "var(--accent)", fontFamily: "JetBrains Mono, monospace",
      }}>{number}</div>
      <div>
        <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)", marginBottom: 2 }}>
          {title}
        </div>
        {formula && (
          <div style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
            color: "var(--teal)", padding: "4px 8px", background: "var(--bg-card)",
            borderRadius: 4, marginBottom: 4, display: "inline-block" }}>
            {formula}
          </div>
        )}
        <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.6 }}>
          {description}
        </div>
      </div>
    </div>
  );
}

function MonoRow({ label, value, highlight }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0",
      borderBottom: "1px solid var(--border-dim)" }}>
      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{label}</span>
      <span style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
        color: highlight ? "var(--teal)" : "var(--text-primary)", fontWeight: highlight ? 600 : 400 }}>
        {value}
      </span>
    </div>
  );
}

export default function FCHMRFPage() {
  const {
    activeExperimentId, stageStatuses, parameters,
    setLayers, refreshResults,
  } = useExperiment();

  const [isRunning, setIsRunning] = useState(false);
  const [execTime, setExecTime] = useState(null);
  const [error, setError] = useState(null);
  const [lastResult, setLastResult] = useState(null);

  const isCompleted = stageStatuses.fchmrf_lis === "completed";

  const handleRun = async () => {
    if (!activeExperimentId) return;
    setError(null);
    setLastResult(null);
    setIsRunning(true);
    const start = performance.now();
    try {
      const result = await pipelineApi.runFchmrfLis(activeExperimentId, {
        nominalAlpha: parameters.nominalAlpha,
        beta: parameters.beta,
        lambdaDl: parameters.lambdaDl,
        maxIterations: parameters.maxIterations,
      });
      setExecTime(((performance.now() - start) / 1000).toFixed(2));
      setLastResult(result);
      await refreshResults(activeExperimentId);
      setLayers((prev) => ({ ...prev, showFdr: true, showLis: true }));
    } catch (err) {
      setError(err);
    } finally {
      setIsRunning(false);
    }
  };

  const renderError = () => {
    if (!error) return null;

    if (error.code === "FC_HMRF_MEMORY_LIMIT") {
      const details = error.detail?.details || {};
      const connectivity = details.neighbor_connectivity;
      return (
        <div style={{ fontSize: 12, color: "var(--rose)", marginTop: 8, padding: "12px",
          background: "var(--rose-dim)", border: "1px solid rgba(244,63,94,0.3)", borderRadius: 6 }}>
          <div style={{ fontWeight: 600, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
            <AlertTriangle size={14} />
            fcHMRF-LIS spatial graph exceeds the configured memory-safe limit.
          </div>
          <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11,
            background: "rgba(0,0,0,0.1)", padding: "8px", borderRadius: 4, marginBottom: 8 }}>
            <MonoRow label="Valid voxels" value={details.valid_voxels?.toLocaleString()} />
            <MonoRow label="Neighborhood" value={connectivity ? `${connectivity}-connectivity lattice` : "N/A"} />
            <MonoRow label="Estimated edges" value={details.estimated_edges?.toLocaleString()} />
            <MonoRow label="Edge limit" value={details.max_graph_edges?.toLocaleString()} />
            <MonoRow label="Voxel limit" value={details.max_valid_voxels?.toLocaleString()} />
            {details.note && (
              <div style={{ marginTop: 6, color: "var(--text-muted)", fontSize: 10 }}>
                Note: {details.note}
              </div>
            )}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-primary)" }}>
            <strong>Actionable options:</strong>
            <ul style={{ marginLeft: 16, marginTop: 4, listStyleType: "circle" }}>
              <li>Reduce neighbor_connectivity: 26 → 18 → 6 (less spatial coupling, lower memory)</li>
              <li>Reduce the active mask (increase ROI threshold / reduce brain mask)</li>
              <li>Use a smaller test volume</li>
            </ul>
          </div>
        </div>
      );
    }

    return (
      <div style={{ fontSize: 12, color: "var(--rose)", marginTop: 8, padding: "8px 10px",
        background: "var(--rose-dim)", border: "1px solid rgba(244,63,94,0.3)", borderRadius: 6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontWeight: 600, marginBottom: 4 }}>
          <AlertTriangle size={13} />Error
        </div>
        {error.message || "fcHMRF-LIS execution failed."}
      </div>
    );
  };

  const renderGraphStats = () => {
    if (!lastResult?.details) return null;
    const details = lastResult.details;
    const gs = details.graph_stats || {};
    const params = details.model_parameters || {};

    return (
      <div style={{ marginTop: 12, padding: "10px 12px",
        background: "var(--bg-input)", border: "1px solid var(--border-dim)", borderRadius: 6 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--teal)", marginBottom: 8,
          display: "flex", alignItems: "center", gap: 5 }}>
          <Activity size={12} /> Graph & Inference Statistics
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
          <em>
            Sparse local lattice neighborhood — no N×N dense matrix allocated.
            {gs.connectivity ? ` ${gs.connectivity}-connectivity 3D lattice.` : ""}
          </em>
        </div>
        <MonoRow label="Valid voxels analyzed" value={details.num_analyzed_voxels?.toLocaleString()} highlight />
        <MonoRow label="Neighborhood" value={gs.connectivity ? `${gs.connectivity}-connected (3D lattice)` : "—"} />
        <MonoRow label="Total graph edges (nnz)" value={gs.num_edges?.toLocaleString() ?? "—"} />
        <MonoRow label="Mean degree" value={gs.mean_degree != null ? gs.mean_degree.toFixed(2) : "—"} />
        <MonoRow label="Graph memory" value={gs.estimated_memory_mb != null ? `${gs.estimated_memory_mb} MB` : "—"} />
        <MonoRow label="MF iterations" value={params.mean_field_iterations ?? "—"} />
        <MonoRow label="MF final δ" value={params.mean_field_final_delta != null ? params.mean_field_final_delta.toExponential(3) : "—"} />
        <MonoRow label="Discoveries (fcHMRF-LIS)" value={details.num_discoveries?.toLocaleString() ?? "—"} highlight />
        <MonoRow label="LIS threshold" value={details.fdr_results?.lis_threshold != null ? details.fdr_results.lis_threshold.toFixed(4) : "—"} />
        <MonoRow label="Estimated FDR" value={details.fdr_results?.estimated_fdr != null ? details.fdr_results.estimated_fdr.toFixed(4) : "—"} />
      </div>
    );
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 720 }}>
      <div>
        <div className="section-title">fcHMRF-LIS Spatial Inference</div>
        <div className="section-sub">
          Fully Connected Hidden Markov Random Field with Local Index of Significance
        </div>
      </div>

      {/* Algorithm overview */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><Network size={13} /> Algorithm</span>
          {isCompleted && <span className="badge badge-completed"><CheckCircle2 size={9} /> Completed</span>}
        </div>

        <StepBlock
          number="1"
          title="Local Lattice Bilateral Spatial Filtering (Memory-Safe)"
          formula="K_ij = w₁·exp(-‖p_i-p_j‖²/2σ_s² - (z_i-z_j)²/2σ_r²) + w₂·exp(-‖p_i-p_j‖²/2σ_p²)"
          description="Spatial interactions are modeled over a local 3D lattice neighborhood (6/18/26-connectivity). Affinity weights use the same bilateral Gaussian kernel as the original fcHMRF formulation. No N×N dense matrix is allocated — graph storage is O(26·N) ≈ linear in voxel count."
        />
        <StepBlock
          number="2"
          title="Mean-Field Variational Inference (Sparse Matvec)"
          formula="m_i = [K·s]_i = Σ_{j∈N(i)} K_ij·(2q_j−1)   →   sparse CSR @ vector"
          description="Damped coordinate-ascent updates converge the variational posterior Q(θ) toward the true joint posterior P(θ|Z). Message passing is a sparse matrix-vector product — O(|E|) time, O(|E|) memory, where |E| ≤ 26·N."
        />
        <StepBlock
          number="3"
          title="Local Index of Significance (LIS)"
          formula="LIS_i = P(θ_i = 0 | Z) = 1 − q_i(1)"
          description="The LIS is the optimal test statistic under spatial dependence (Sun & Cai, 2009). Lower LIS indicates stronger evidence of activation at voxel i."
        />
        <StepBlock
          number="4"
          title="Spatial FDR Control (Step-Up)"
          formula="k = max { j : (1/j) Σ LIS_(i) ≤ α }"
          description="Rejects the null hypothesis for all voxels with LIS ≤ LIS_(k), guaranteeing FDR ≤ α while maximizing detection power."
        />
      </div>

      {/* Run control */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Execution</span>
          <div style={{ display: "flex", gap: 8, fontSize: 11,
            fontFamily: "JetBrains Mono, monospace", color: "var(--text-muted)" }}>
            <span>α = {parameters.nominalAlpha}</span>
            <span>β = {parameters.beta}</span>
            <span>λ = {parameters.lambdaDl}</span>
            <span>26-conn lattice</span>
          </div>
        </div>

        {!activeExperimentId && (
          <div style={{
            display: "flex", gap: 8, padding: "9px 12px", borderRadius: 6,
            background: "var(--amber-dim)", border: "1px solid rgba(245,158,11,0.25)",
            fontSize: 12, color: "var(--amber)", marginBottom: 12,
          }}>
            <Info size={14} style={{ flexShrink: 0 }} />
            <span>Statistical analysis must be completed before running fcHMRF-LIS.</span>
          </div>
        )}

        {isRunning && (
          <div style={{ marginBottom: 10, padding: "9px 12px", borderRadius: 6,
            background: "var(--accent-dim)", border: "1px solid var(--accent-border)",
            fontSize: 11, color: "var(--accent)", fontFamily: "JetBrains Mono, monospace" }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Processing…</div>
            <div>Building sparse 26-connectivity lattice graph → running mean-field iterations → computing LIS → applying FDR step-up</div>
            <div style={{ marginTop: 4, color: "var(--text-muted)", fontSize: 10 }}>
              FDR stage will remain NOT STARTED until fcHMRF-LIS completes successfully.
            </div>
          </div>
        )}

        <button
          className="btn btn-primary btn-full"
          onClick={handleRun}
          disabled={!activeExperimentId || isRunning}
        >
          {isRunning ? (
            <><Loader2 size={13} className="animate-spin" /> Computing Sparse Lattice LIS…</>
          ) : (
            <><Play size={13} style={{ fill: "currentColor" }} /> {isCompleted ? "Re-run fcHMRF-LIS" : "Run fcHMRF-LIS"}</>
          )}
        </button>

        {renderError()}
        {renderGraphStats()}

        {execTime && !error && (
          <div style={{ display: "flex", justifyContent: "space-between",
            marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--border-dim)",
            fontSize: 11, fontFamily: "JetBrains Mono, monospace", color: "var(--text-muted)" }}>
            <span>Execution time</span>
            <span style={{ color: "var(--text-primary)" }}>{execTime} s</span>
          </div>
        )}
      </div>
    </div>
  );
}
