import React from "react";
import { GitCompare, Info } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import BaselineComparisonView from "../BaselineComparison/BaselineComparisonView";

export default function ComparisonPage() {
  const { resultsData } = useExperiment();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <div className="section-title">Baseline Comparison</div>
        <div className="section-sub">BH-FDR vs Adaptive fcHMRF-LIS side-by-side analysis</div>
      </div>

      {/* Phase 8 multi-method comparison table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><GitCompare size={13} /> Multi-Method Benchmark (Phase 8)</span>
          <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "JetBrains Mono, monospace" }}>
            5 methods · α = 0.05
          </span>
        </div>

        <div style={{
          display: "flex", gap: 8, padding: "9px 12px", borderRadius: 6, marginBottom: 14,
          background: "var(--amber-dim)", border: "1px solid rgba(245,158,11,0.25)",
          fontSize: 11, color: "var(--text-muted)",
        }}>
          <Info size={13} style={{ flexShrink: 0, color: "var(--amber)", marginTop: 1 }} />
          <span>
            Results below are from Phase 8 controlled synthetic benchmarks (5 Monte Carlo trials,
            ground truth verified). Values report Mean ± SEM.
            These are benchmark reference values, not live experiment values.
          </span>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table className="research-table">
            <thead>
              <tr>
                <th>Method</th>
                <th style={{ textAlign: "center" }}>FDR (FDP)</th>
                <th style={{ textAlign: "center", color: "var(--accent)" }}>TPR (Power)</th>
                <th style={{ textAlign: "center" }}>FPR</th>
                <th style={{ textAlign: "center" }}>Precision</th>
                <th style={{ textAlign: "center" }}>Loc. Accuracy</th>
                <th style={{ textAlign: "center" }}>Runtime</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ color: "var(--teal)", fontWeight: 700 }}>1. Benjamini-Hochberg FDR</td>
                <td style={{ textAlign: "center", color: "var(--emerald)" }}>0.42% ± 0.26%</td>
                <td style={{ textAlign: "center" }}>37.98% ± 3.47%</td>
                <td style={{ textAlign: "center" }}>0.018% ± 0.011%</td>
                <td style={{ textAlign: "center" }}>99.58% ± 0.26%</td>
                <td style={{ textAlign: "center" }}>1.39 ± 0.51 vox</td>
                <td style={{ textAlign: "center" }}>0.0014 s</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>2. Original fcHMRF-LIS</td>
                <td style={{ textAlign: "center", color: "var(--emerald)" }}>0.76% ± 0.20%</td>
                <td style={{ textAlign: "center" }}>44.65% ± 3.12%</td>
                <td style={{ textAlign: "center" }}>0.036% ± 0.009%</td>
                <td style={{ textAlign: "center" }}>99.24% ± 0.20%</td>
                <td style={{ textAlign: "center" }}>1.23 ± 0.46 vox</td>
                <td style={{ textAlign: "center" }}>1.06 s</td>
              </tr>
              <tr>
                <td style={{ color: "var(--amber)", fontWeight: 700 }}>3. ROI + fcHMRF-LIS</td>
                <td style={{ textAlign: "center", color: "var(--emerald)" }}>1.24% ± 0.53%</td>
                <td style={{ textAlign: "center", color: "var(--amber)", fontWeight: 700 }}>72.28% ± 2.54%</td>
                <td style={{ textAlign: "center" }}>0.098% ± 0.046%</td>
                <td style={{ textAlign: "center" }}>98.76% ± 0.53%</td>
                <td style={{ textAlign: "center", color: "var(--emerald)" }}>0.51 ± 0.23 vox</td>
                <td style={{ textAlign: "center" }}>1.02 s</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>4. Adaptive fcHMRF-LIS</td>
                <td style={{ textAlign: "center", color: "var(--emerald)" }}>0.39% ± 0.24%</td>
                <td style={{ textAlign: "center" }}>41.75% ± 3.16%</td>
                <td style={{ textAlign: "center" }}>0.018% ± 0.011%</td>
                <td style={{ textAlign: "center" }}>99.61% ± 0.24%</td>
                <td style={{ textAlign: "center" }}>1.39 ± 0.49 vox</td>
                <td style={{ textAlign: "center" }}>1.10 s</td>
              </tr>
              <tr style={{ background: "rgba(59,130,212,0.07)", borderTop: "1px solid var(--accent-border)" }}>
                <td style={{ color: "var(--accent)", fontWeight: 700 }}>
                  5. ROI + Adaptive fcHMRF-LIS <span style={{ fontSize: 10, color: "var(--text-muted)" }}>(Proposed)</span>
                </td>
                <td style={{ textAlign: "center", color: "var(--emerald)", fontWeight: 700 }}>0.86% ± 0.58%</td>
                <td style={{ textAlign: "center", color: "var(--accent)", fontWeight: 700 }}>74.56% ± 2.36%</td>
                <td style={{ textAlign: "center" }}>0.072% ± 0.050%</td>
                <td style={{ textAlign: "center" }}>99.14% ± 0.58%</td>
                <td style={{ textAlign: "center", color: "var(--emerald)", fontWeight: 700 }}>0.49 ± 0.21 vox</td>
                <td style={{ textAlign: "center" }}>1.14 s</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual comparison */}
      <BaselineComparisonView />
    </div>
  );
}
