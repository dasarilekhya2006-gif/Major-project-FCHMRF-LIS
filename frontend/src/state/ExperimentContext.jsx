import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import * as pipelineApi from "../api/pipelineApi";
import * as resultsApi from "../api/resultsApi";
import * as experimentApi from "../api/experimentApi";

const ExperimentContext = createContext(null);

export function ExperimentProvider({ children }) {
  const [activeExperimentId, setActiveExperimentId] = useState(null);
  const [datasetMeta, setDatasetMeta] = useState(null);

  // Pipeline execution tracking
  const [pipelineStatus, setPipelineStatus] = useState({
    status: "idle", // idle, queued, running, completed, failed
    stage: "idle",
    progress: 0,
    error: null,
  });

  const [stageStatuses, setStageStatuses] = useState({
    upload: "pending",
    preprocess: "pending",
    segmentation: "pending",
    statistics: "pending",
    fchmrf_lis: "pending",
    fdr: "pending",
  });

  // Parameters (Manual vs Adaptive)
  const [parameters, setParameters] = useState({
    adaptiveMode: true,
    nominalAlpha: 0.05,
    beta: 0.8,
    lambdaDl: 0.5,
    maxIterations: 20,
    probabilityThreshold: 0.5,
  });

  // Visualization settings
  const [activeTab, setActiveTab] = useState("2d"); // 2d, 3d, comparison, metrics, history, export
  const [orientation, setOrientation] = useState("axial");
  const [sliceIdx, setSliceIdx] = useState(0);
  const [maxSlice, setMaxSlice] = useState(0);

  const [layers, setLayers] = useState({
    showMri: true,
    mriOpacity: 1.0,
    showRoi: true,
    roiOpacity: 0.45,
    showStat: false,
    statThreshold: 2.0,
    showLis: false,
    lisThreshold: 0.1,
    showFdr: true,
    heatmapOpacity: 0.75,
    showGroundTruth: false,
    showBaseline: false,
  });

  // Raw data payloads
  const [sliceData, setSliceData] = useState(null);
  const [meshData, setMeshData] = useState(null);
  const [resultsData, setResultsData] = useState(null);
  const [experimentDetails, setExperimentDetails] = useState(null);
  const [experimentsList, setExperimentsList] = useState([]);
  const [backendOnline, setBackendOnline] = useState(true);

  // Probe backend health periodically
  useEffect(() => {
    async function checkHealth() {
      try {
        await experimentApi.checkBackendHealth();
        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Fetch experiments list
  const refreshExperimentsList = useCallback(async () => {
    try {
      const list = await experimentApi.listExperiments();
      setExperimentsList(list || []);
      // If no active experiment but experiments exist, select first
      if (!activeExperimentId && list && list.length > 0) {
        selectExperiment(list[0].experiment_id);
      }
    } catch (err) {
      console.warn("Failed to list experiments:", err);
    }
  }, [activeExperimentId]);

  useEffect(() => {
    refreshExperimentsList();
  }, [refreshExperimentsList]);

  // Load slice data for active experiment, orientation, and slice index
  const loadSlice = useCallback(
    async (expId, plane, idx) => {
      if (!expId) return;
      try {
        const data = await resultsApi.getSliceData(expId, plane, idx);
        setSliceData(data);
        setMaxSlice(data.max_slice);
        setSliceIdx(data.slice_idx);
      } catch (err) {
        console.warn("Failed to load slice data:", err);
      }
    },
    []
  );

  // Load 3D mesh data
  const load3DMesh = useCallback(async (expId, stepSize = 1) => {
    if (!expId) return;
    try {
      const data = await resultsApi.getMesh3D(expId, stepSize);
      setMeshData(data);
    } catch (err) {
      console.warn("Failed to load 3D mesh:", err);
    }
  }, []);

  // Refresh results and metadata
  const refreshResults = useCallback(
    async (expId) => {
      if (!expId) return;
      try {
        const [res, details] = await Promise.all([
          resultsApi.getResults(expId),
          experimentApi.getExperimentDetails(expId),
        ]);
        setResultsData(res);
        setExperimentDetails(details);
        if (details.stages) {
          setStageStatuses((prev) => ({ ...prev, ...details.stages }));
        }
        if (details.shape) {
          setDatasetMeta({
            filename: details.filename,
            shape: details.shape,
            voxel_spacing: details.voxel_spacing,
            orientation: details.orientation,
          });
        }
      } catch (err) {
        console.warn("Failed to refresh results:", err);
      }
    },
    []
  );

  // Select active experiment
  const selectExperiment = useCallback(
    async (expId) => {
      setActiveExperimentId(expId);
      await refreshResults(expId);
      await loadSlice(expId, orientation, null);
    },
    [refreshResults, loadSlice, orientation]
  );

  // When orientation or slice index changes, load slice
  const changeOrientation = useCallback(
    (newOrientation) => {
      setOrientation(newOrientation);
      if (activeExperimentId) {
        loadSlice(activeExperimentId, newOrientation, null);
      }
    },
    [activeExperimentId, loadSlice]
  );

  const changeSlice = useCallback(
    (newSlice) => {
      setSliceIdx(newSlice);
      if (activeExperimentId) {
        loadSlice(activeExperimentId, orientation, newSlice);
      }
    },
    [activeExperimentId, orientation, loadSlice]
  );

  // Pipeline status polling
  useEffect(() => {
    if (!activeExperimentId || pipelineStatus.status !== "running") return;

    const interval = setInterval(async () => {
      try {
        const status = await pipelineApi.getPipelineStatus(activeExperimentId);
        setPipelineStatus({
          status: status.status,
          stage: status.stage,
          progress: status.progress,
          error: status.error,
        });

        if (status.status === "completed" || status.status === "failed") {
          clearInterval(interval);
          refreshResults(activeExperimentId);
          loadSlice(activeExperimentId, orientation, sliceIdx);
          refreshExperimentsList();
        }
      } catch (err) {
        console.warn("Polling error:", err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeExperimentId, pipelineStatus.status, orientation, sliceIdx, refreshResults, loadSlice, refreshExperimentsList]);

  const value = {
    activeExperimentId,
    setActiveExperimentId,
    selectExperiment,
    datasetMeta,
    setDatasetMeta,
    pipelineStatus,
    setPipelineStatus,
    stageStatuses,
    setStageStatuses,
    parameters,
    setParameters,
    activeTab,
    setActiveTab,
    orientation,
    changeOrientation,
    sliceIdx,
    changeSlice,
    maxSlice,
    layers,
    setLayers,
    sliceData,
    meshData,
    load3DMesh,
    resultsData,
    experimentDetails,
    experimentsList,
    refreshExperimentsList,
    refreshResults,
    backendOnline,
  };

  return <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>;
}

export function useExperiment() {
  const context = useContext(ExperimentContext);
  if (!context) {
    throw new Error("useExperiment must be used within an ExperimentProvider");
  }
  return context;
}
