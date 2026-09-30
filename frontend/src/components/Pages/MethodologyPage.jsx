import React from "react";
import { BookOpen, ArrowRight } from "lucide-react";

function Stage({ number, title, formula, description, accent }) {
  return (
    <div style={{
      display: "flex", gap: 14,
      padding: "14px 16px",
      background: "var(--bg-card)",
      border: "1px solid var(--border-dim)",
      borderRadius: 8, marginBottom: 8,
      borderLeft: `3px solid ${accent || "var(--accent)"}`,
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: 7, flexShrink: 0,
        background: "var(--bg-elevated)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 12, fontWeight: 800,
        color: accent || "var(--accent)",
        fontFamily: "JetBrains Mono, monospace",
      }}>{number}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", marginBottom: 3 }}>
          {title}
        </div>
        {formula && (
          <div style={{
            fontSize: 11, fontFamily: "JetBrains Mono, monospace",
            color: "var(--teal)", background: "var(--bg-input)",
            padding: "4px 8px", borderRadius: 4, display: "inline-block",
            marginBottom: 6, border: "1px solid var(--border-dim)",
          }}>
            {formula}
          </div>
        )}
        <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.65 }}>
          {description}
        </div>
      </div>
    </div>
  );
}

export default function MethodologyPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 820 }}>
      <div>
        <div className="section-title">Methodology</div>
        <div className="section-sub">
          fcHMRF-LIS pipeline — mathematical foundations and stage descriptions
        </div>
      </div>

      {/* Pipeline flow description */}
      <div className="card">
        <div className="card-header">
          <span className="card-title"><BookOpen size={13} /> Research Pipeline</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 6,
          fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
          flexWrap: "wrap", marginBottom: 16, padding: "8px 0",
          borderBottom: "1px solid var(--border-dim)" }}>
          {[
            "NIfTI Input", "Preprocessing", "3D U-Net ROI",
            "Statistical Testing", "Adaptive Parameters",
            "fcHMRF-LIS", "FDR Control", "Visualization",
          ].map((s, i, arr) => (
            <React.Fragment key={s}>
              <span style={{ padding: "3px 8px", borderRadius: 4, background: "var(--bg-input)",
                border: "1px solid var(--border-dim)", color: "var(--text-secondary)" }}>
                {s}
              </span>
              {i < arr.length - 1 && (
                <ArrowRight size={12} style={{ color: "var(--text-dim)", flexShrink: 0 }} />
              )}
            </React.Fragment>
          ))}
        </div>

        <Stage
          number="1"
          title="NIfTI Ingestion & Validation"
          accent="var(--teal)"
          description="Raw 3D or 4D neuroimaging volumes are validated using NiBabel. Structural integrity checks include affine invertibility, positive voxel spacing, and non-finite value detection. Data is canonicalized to RAS+ orientation in-memory without modifying source files."
        />
        <Stage
          number="2"
          title="Modular Preprocessing"
          accent="var(--accent)"
          description="Applies intensity normalization (Z-score), Otsu brain mask extraction, and optional Gaussian spatial smoothing. An audit trail JSON and QC figure are generated automatically."
        />
        <Stage
          number="3"
          title="3D U-Net ROI Prior Detection"
          formula="π_i^(DL) = P(θ_i = 1 | context)"
          accent="var(--accent)"
          description="A lightweight configurable 3D U-Net produces a continuous probability map encoding structural prior activation likelihood. Setting λ_DL = 0 disables this component and recovers the original published fcHMRF-LIS behavior."
        />
        <Stage
          number="4"
          title="Voxel-Level Statistical Testing"
          formula="z_i = Φ⁻¹(1 − p_i)"
          accent="var(--accent)"
          description="Two-sided voxel-wise hypothesis testing within the brain mask generates z-scores and raw p-values. Supports one-sample t-test, two-sample t-test, paired t-test, and GLM with contrast vectors."
        />
        <Stage
          number="5"
          title="Adaptive Parameter Selection (Optional)"
          formula="Score = Power − γ · max(0, FDP − α)"
          accent="var(--amber)"
          description="A Bayesian Optimization loop (Gaussian Process + Expected Improvement) selects fcHMRF-LIS hyperparameters (β, σ_s, σ_r, λ_DL) using a validation partition. Parameters are NEVER tuned on test data. Whether adaptive selection improves over defaults must be empirically verified."
        />
        <Stage
          number="6"
          title="fcHMRF-LIS Spatial Inference"
          formula="LIS_i = P(θ_i = 0 | Z) = 1 − q_i(1)"
          accent="var(--accent)"
          description="Dense bilateral Gaussian filters capture all-voxel-pair spatial dependencies. Mean-field variational inference iteratively refines posterior activation probabilities q_i(1). The Local Index of Significance provides the optimal test statistic under spatial dependency (Sun & Cai, 2009)."
        />
        <Stage
          number="7"
          title="Spatial FDR Control — LIS Step-Up Procedure"
          formula="k = max { j : (1/j) Σ LIS_(i) ≤ α }"
          accent="var(--emerald)"
          description="All voxels with LIS ≤ LIS_(k) are declared significant. This guarantees FDR ≤ α while minimizing the False Non-Discovery Rate — maximizing statistical power under spatial dependence."
        />
      </div>

      {/* Attribution */}
      <div style={{
        padding: "12px 16px", background: "var(--bg-card)",
        border: "1px solid var(--border-dim)", borderRadius: 8,
        fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7,
      }}>
        <strong style={{ color: "var(--text-secondary)" }}>References: </strong>
        Sun, W. &amp; Cai, T.T. (2009). Large-scale multiple testing under dependence.
        <em> Journal of the Royal Statistical Society: Series B</em>, 71(2), 393–424. ·
        Eklund, A. et al. (2016). Cluster failure: Why fMRI inferences for spatial extent have
        inflated false-positive rates. <em>PNAS</em>, 113(28), 7900–7905.
        <br /><br />
        <strong style={{ color: "var(--amber)" }}>Research Integrity: </strong>
        This platform makes no claims of scientific superiority over existing methods until controlled
        empirical experiments and synthetic benchmarks establish statistically validated evidence.
      </div>
    </div>
  );
}
