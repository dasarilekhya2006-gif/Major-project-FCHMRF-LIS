import React, { useState } from "react";
import { Upload, FileCheck, AlertCircle, Loader2, Database } from "lucide-react";
import { useExperiment } from "../../state/ExperimentContext";
import * as uploadApi from "../../api/uploadApi";

export default function DatasetUploadPanel() {
  const {
    activeExperimentId,
    setActiveExperimentId,
    datasetMeta,
    setDatasetMeta,
    refreshResults,
    refreshExperimentsList,
    changeOrientation,
  } = useExperiment();

  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".nii") && !file.name.endsWith(".nii.gz")) {
      setErrorMessage("Only .nii or .nii.gz NIfTI files are supported.");
      return;
    }

    setErrorMessage(null);
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const res = await uploadApi.uploadNiftiVolume(file, (p) => setUploadProgress(p));
      setActiveExperimentId(res.experiment_id);
      setDatasetMeta({
        filename: res.filename,
        shape: res.shape,
        voxel_spacing: res.voxel_spacing,
        orientation: res.orientation,
      });

      await refreshExperimentsList();
      await refreshResults(res.experiment_id);
      changeOrientation("axial");
    } catch (err) {
      setErrorMessage(err.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <Database className="w-3.5 h-3.5 text-orange-400" />
          1. Dataset Upload
        </h3>
        <span className="text-[11px]  text-slate-400">.nii, .nii.gz</span>
      </div>

      {/* Upload Dropzone */}
      <label className="border border-dashed border-slate-700 hover:border-orange-500/80 rounded-md p-3.5 flex flex-col items-center justify-center gap-1.5 cursor-pointer bg-slate-950/40 hover:bg-slate-900 transition-all text-center">
        <input
          type="file"
          accept=".nii,.nii.gz"
          className="hidden"
          onChange={handleFileChange}
          disabled={isUploading}
        />
        {isUploading ? (
          <div className="space-y-1.5 w-full">
            <Loader2 className="w-5 h-5 text-orange-400 animate-spin mx-auto" />
            <div className="text-xs text-slate-300">Validating NIfTI headers... {uploadProgress}%</div>
            <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
              <div
                className="bg-orange-500 h-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        ) : (
          <>
            <Upload className="w-5 h-5 text-slate-400 hover:text-orange-400" />
            <span className="text-xs font-medium text-slate-300">Choose MRI / fMRI Volume</span>
            <span className="text-[11px] text-slate-500">Validated with NiBabel headers</span>
          </>
        )}
      </label>

      {errorMessage && (
        <div className="bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs p-2 rounded flex items-center gap-1.5">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Dataset Metadata Display */}
      {datasetMeta && (
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2.5 space-y-1.5 text-xs ">
          <div className="flex items-center gap-1.5 text-emerald-400 font-semibold text-[11px]">
            <FileCheck className="w-3.5 h-3.5" />
            <span>✓ Dataset Validated</span>
          </div>
          <div className="text-slate-300 truncate" title={datasetMeta.filename}>
            <span className="text-slate-500">File: </span>
            {datasetMeta.filename}
          </div>
          <div className="text-slate-300">
            <span className="text-slate-500">Dimensions: </span>
            {datasetMeta.shape?.join(" × ")}
          </div>
          <div className="text-slate-300">
            <span className="text-slate-500">Voxel spacing: </span>
            {datasetMeta.voxel_spacing?.map((v) => v.toFixed(1)).join(" × ")} mm
          </div>
          <div className="text-slate-300">
            <span className="text-slate-500">Orientation: </span>
            {datasetMeta.orientation || "RAS"}
          </div>
          <div className="text-orange-400 font-bold truncate">
            <span className="text-slate-500">Exp ID: </span>
            {activeExperimentId}
          </div>
        </div>
      )}
    </div>
  );
}
