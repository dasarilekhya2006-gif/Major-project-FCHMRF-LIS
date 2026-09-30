import React, { useState } from "react";
import { AlertTriangle, Info, X } from "lucide-react";

export default function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return (
      <div className="bg-amber-950/40 border-b border-amber-900/50 px-4 py-1 text-xs text-amber-300 flex items-center justify-between">
        <span className="flex items-center gap-1.5 font-medium">
          <Info className="w-3.5 h-3.5 text-amber-400" />
          Research Prototype — Not for Clinical Diagnosis
        </span>
        <button
          onClick={() => setDismissed(false)}
          className="text-amber-400 hover:text-amber-200 underline text-[11px]"
        >
          View Full Disclaimer
        </button>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-amber-950/80 via-orange-950/80 to-amber-950/80 border-b border-amber-800/60 px-4 py-2 text-xs text-amber-200">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1 rounded bg-amber-500/20 text-amber-400 flex-shrink-0">
            <AlertTriangle className="w-4 h-4" />
          </div>
          <div>
            <span className="font-bold tracking-wide uppercase text-amber-300 mr-2 text-[11px] px-1.5 py-0.5 rounded bg-amber-900/60 border border-amber-700/50">
              Research Prototype — Not for Clinical Diagnosis
            </span>
            <span className="text-amber-200/90 leading-tight">
              This system is developed for research and demonstration purposes only. It has not been validated,
              approved, or intended for clinical diagnosis, treatment decisions, or medical use.
            </span>
          </div>
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="text-amber-400 hover:text-amber-100 p-1 rounded hover:bg-amber-900/40 transition-colors"
          title="Minimize disclaimer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
