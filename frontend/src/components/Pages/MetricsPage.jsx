import React from "react";
import MetricsTable from "../MetricsTable/MetricsTable";
import ComputationalTimePanel from "../ComputationalTime/ComputationalTimePanel";

export default function MetricsPage() {

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <div className="section-title">Metrics</div>
        <div className="section-sub">Quantitative evaluation and computational performance</div>
      </div>

      <MetricsTable />
      <ComputationalTimePanel />
    </div>
  );
}
