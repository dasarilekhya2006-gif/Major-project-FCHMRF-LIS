import React, { useState } from "react";
import { useExperiment } from "./state/ExperimentContext";
import WorkstationHeader from "./components/Workstation/WorkstationHeader";
import WorkstationSidebar from "./components/Workstation/WorkstationSidebar";

// Pages
import OverviewPage     from "./components/Pages/OverviewPage";
import DatasetPage      from "./components/Pages/DatasetPage";
import PreprocessingPage from "./components/Pages/PreprocessingPage";
import ROIPage          from "./components/Pages/ROIPage";
import StatisticsPage   from "./components/Pages/StatisticsPage";
import ParametersPage   from "./components/Pages/ParametersPage";
import FCHMRFPage       from "./components/Pages/FCHMRFPage";
import FDRPage          from "./components/Pages/FDRPage";
import ViewerPage       from "./components/Pages/ViewerPage";
import ComparisonPage   from "./components/Pages/ComparisonPage";
import MetricsPage      from "./components/Pages/MetricsPage";
import HistoryPage      from "./components/Pages/HistoryPage";
import MethodologyPage  from "./components/Pages/MethodologyPage";

// Disclaimer banner
import DisclaimerBanner from "./components/Header/DisclaimerBanner";

export default function App() {
  const [activePage, setActivePage] = useState("overview");
  useExperiment(); // keep context warm

  function renderPage() {
    switch (activePage) {
      case "overview":      return <OverviewPage onNavigate={setActivePage} />;
      case "dataset":       return <DatasetPage />;
      case "preprocessing": return <PreprocessingPage />;
      case "roi":           return <ROIPage />;
      case "statistics":    return <StatisticsPage />;
      case "parameters":    return <ParametersPage />;
      case "fchmrf":        return <FCHMRFPage />;
      case "fdr":           return <FDRPage />;
      case "viewer":        return <ViewerPage />;
      case "comparison":    return <ComparisonPage />;
      case "metrics":       return <MetricsPage />;
      case "history":       return <HistoryPage onNavigate={setActivePage} />;
      case "methodology":   return <MethodologyPage />;
      default:              return <OverviewPage onNavigate={setActivePage} />;
    }
  }

  return (
    <div className="ws-layout">
      {/* Disclaimer */}
      <DisclaimerBanner />

      {/* Header */}
      <WorkstationHeader />

      {/* Body: sidebar + main content */}
      <div className="ws-body">
        <WorkstationSidebar activePage={activePage} onNavigate={setActivePage} />

        <main className="ws-main">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}
