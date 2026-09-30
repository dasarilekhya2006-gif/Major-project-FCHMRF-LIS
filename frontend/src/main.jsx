import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ExperimentProvider } from "./state/ExperimentContext";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ExperimentProvider>
      <App />
    </ExperimentProvider>
  </React.StrictMode>
);
