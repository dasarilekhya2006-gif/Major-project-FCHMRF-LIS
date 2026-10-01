import React from "react";
import { Download, FileText, Database, Image, CheckCircle, ExternalLink } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import { getDownloadUrl } from "../../api/resultsApi";

export default function DownloadResultsModal() {
  const { activeExperimentId, resultsData, experimentDetails } = useExperiment();

  const available = resultsData?.available_layers || {};

  const downloadCsvMetrics = () => {
    if (!resultsData) return;
    const discoveries = resultsData.discoveries || {};
    const metrics = resultsData.metrics || {};

    const csvContent = [
      ["Metric", "Value"],
      ["Experiment ID", activeExperimentId],
      ["Proposed Discoveries", discoveries.proposed_discoveries ?? 0],
      ["Baseline Discoveries", discoveries.baseline_discoveries ?? 0],
      ["Consensus Discoveries", discoveries.consensus_discoveries ?? 0],
      ["Proposed Only Discoveries", discoveries.proposed_only_discoveries ?? 0],
      ["Baseline Only Discoveries", discoveries.baseline_only_discoveries ?? 0],
      ["Method Agreement (Dice)", discoveries.agreement_dice ?? "N/A"],
      ["Empirical FDR", metrics.proposed_fdr ?? "N/A"],
      ["Sensitivity", metrics.proposed_sensitivity ?? "N/A"],
      ["Specificity", metrics.proposed_specificity ?? "N/A"],
    ]
      .map((row) => row.join(","))
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `experiment_${activeExperimentId}_metrics.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadJsonMetadata = () => {
    if (!experimentDetails) return;
    const blob = new Blob([JSON.stringify(experimentDetails, null, 2)], {
      type: "application/json;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `experiment_${activeExperimentId}_metadata.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const artifacts = [
    {
      name: "Preprocessed Volume (.nii.gz)",
      stage: "preprocess",
      file: "preprocessed.nii.gz",
      isReady: available.preprocessed,
    },
    {
      name: "3D U-Net ROI Mask (.nii.gz)",
      stage: "segmentation",
      file: "roi_mask.nii.gz",
      isReady: available.roi_mask,
    },
    {
      name: "Voxel Statistical Map (.nii.gz)",
      stage: "statistics",
      file: "statistical_map.nii.gz",
      isReady: available.statistical_map,
    },
    {
      name: "Local Index of Significance (LIS) Map (.nii.gz)",
      stage: "fchmrf",
      file: "lis_map.nii.gz",
      isReady: available.lis_map,
    },
    {
      name: "FDR-Significant Voxels Mask (.nii.gz)",
      stage: "fdr",
      file: "significant_mask.nii.gz",
      isReady: available.significant_mask,
    },
    {
      name: "Baseline BH Significant Mask (.nii.gz)",
      stage: "fdr",
      file: "baseline_significant_mask.nii.gz",
      isReady: available.baseline_significant_mask,
    },
  ];

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-4 space-y-4  text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Download className="w-3.5 h-3.5 text-orange-400" />
          Export Research Artifacts & Data
        </h3>
        <span className="text-slate-400">Exp: {activeExperimentId}</span>
      </div>

      {/* Structured Reports */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="font-bold text-slate-200 block">Quantitative Metrics Summary</span>
            <span className="text-[11px] text-slate-500">CSV spreadsheet with statistical comparison</span>
          </div>
          <button onClick={downloadCsvMetrics} className="btn-secondary text-xs py-1 px-3">
            <FileText className="w-3.5 h-3.5 text-orange-400" />
            <span>Export CSV</span>
          </button>
        </div>

        <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="font-bold text-slate-200 block">Provenance & Parameters</span>
            <span className="text-[11px] text-slate-500">JSON metadata with affine and runtime parameters</span>
          </div>
          <button onClick={downloadJsonMetadata} className="btn-secondary text-xs py-1 px-3">
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* Volumetric NIfTI Downloads */}
      <div className="space-y-2 pt-2">
        <span className="text-slate-400 font-semibold block text-[11px] uppercase tracking-wider">
          Download Sliced & Spatially Aligned NIfTI Volumes
        </span>

        <div className="grid grid-cols-2 gap-2">
          {artifacts.map((art) => (
            <div
              key={art.name}
              className="bg-slate-950/70 border border-slate-800/80 rounded p-2 flex items-center justify-between"
            >
              <div className="truncate pr-2">
                <span className="text-slate-300 block text-xs truncate">{art.name}</span>
                <span className="text-[10px] text-slate-500 ">{art.file}</span>
              </div>

              {art.isReady && activeExperimentId ? (
                <a
                  href={getDownloadUrl(activeExperimentId, art.stage, art.file)}
                  download={art.file}
                  className="btn-secondary text-[11px] py-1 px-2 text-orange-400 hover:text-orange-300 flex-shrink-0"
                >
                  <Download className="w-3 h-3" />
                  <span>Download</span>
                </a>
              ) : (
                <span className="text-[10px] text-slate-600 italic px-2">Pending stage</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
