import React from "react";
import { ShieldCheck, CheckCircle2 } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";

function ResultRow({ label, value, highlight }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "10px 14px",
      background: highlight ? "var(--accent-dim)" : "transparent",
      borderBottom: "1px solid var(--border-dim)",
      fontSize: 13,
    }}>
      <span style={{ color: "var(--text-secondary)", fontWeight: highlight ? 600 : 400 }}>{label}</span>
      <span style={{ fontFamily: "JetBrains Mono, monospace", fontWeight: 700,
        color: highlight ? "var(--accent)" : "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}

export default function FDRPage() {
  const { resultsData, stageStatuses, parameters } = useExperiment();

  const isCompleted = stageStatuses.fdr === "completed" || stageStatuses.fchmrf_lis === "completed";
  const discoveries = resultsData?.discoveries;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 640 }}>
      <div>
        <div className="section-title">FDR Results</div>
        <div className="section-sub">Spatial False Discovery Rate control via LIS step-up procedure</div>
      </div>

      {!isCompleted ? (
        <div className="card">
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", gap: 12, padding: "40px 24px", textAlign: "center",
          }}>
            <div style={{ width: 48, height: 48, borderRadius: "50%",
              background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)",
              display: "flex", alignItems: "center", justifyContent: "center",
              color: "var(--text-dim)" }}>
              <ShieldCheck size={22} />
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-secondary)" }}>
              No FDR Results Available
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", maxWidth: 320, lineHeight: 1.7 }}>
              Run fcHMRF-LIS and FDR analysis first to see spatial FDR control results.
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Status card */}
          <div style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "12px 16px", borderRadius: 8,
            background: "var(--emerald-dim)",
            border: "1px solid rgba(16,185,129,0.25)",
            color: "var(--emerald)", fontSize: 13, fontWeight: 600,
          }}>
            <CheckCircle2 size={16} />
            FDR-controlled analysis complete — α = {parameters.nominalAlpha}
          </div>

          {/* Results */}
          <div className="card">
            <div className="card-header">
              <span className="card-title"><ShieldCheck size={13} /> Spatial FDR Results</span>
            </div>
            <div style={{ background: "var(--bg-input)", borderRadius: 7,
              border: "1px solid var(--border-dim)", overflow: "hidden" }}>
              <ResultRow
                label="Nominal FDR Level (α)"
                value={parameters.nominalAlpha}
                highlight
              />
              {discoveries && (
                <>
                  <ResultRow
                    label="Proposed Discoveries (fcHMRF-LIS)"
                    value={(discoveries.proposed_discoveries ?? "—").toLocaleString?.() ?? discoveries.proposed_discoveries ?? "—"}
                    highlight
                  />
                  <ResultRow
                    label="Baseline Discoveries (BH-FDR)"
                    value={(discoveries.baseline_discoveries ?? "—").toLocaleString?.() ?? discoveries.baseline_discoveries ?? "—"}
                  />
                  {discoveries.agreement_dice !== undefined && discoveries.agreement_dice !== null && (
                    <ResultRow
                      label="Method Agreement (Dice)"
                      value={(discoveries.agreement_dice * 100).toFixed(1) + "%"}
                    />
                  )}
                  {discoveries.lis_threshold !== undefined && (
                    <ResultRow
                      label="LIS Cutoff Threshold"
                      value={discoveries.lis_threshold?.toFixed(4) ?? "—"}
                    />
                  )}
                </>
              )}
            </div>
          </div>

          {/* Explanation */}
          <div style={{
            padding: "12px 14px", background: "var(--bg-card)",
            border: "1px solid var(--border-dim)", borderRadius: 8,
            fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7,
          }}>
            <strong style={{ color: "var(--text-secondary)" }}>LIS Step-Up Procedure (Sun & Cai 2009): </strong>
            All voxels with LIS ≤ LIS<sub>(k)</sub> are rejected, where k is the largest index satisfying
            (1/k) Σ LIS<sub>(i)</sub> ≤ α. This procedure guarantees FDR ≤ α while minimizing the
            False Non-Discovery Rate.
          </div>
        </>
      )}
    </div>
  );
}
