// Frontend application logic for Adaptive DL-fcHMRF-LIS Research Prototype

let currentSampleId = "default";
let maxSliceIndex = 23;

// DOM Elements
const apiStatusEl = document.getElementById("api-status");
const snrInput = document.getElementById("snr-input");
const snrVal = document.getElementById("snr-val");
const clustersInput = document.getElementById("clusters-input");
const clustersVal = document.getElementById("clusters-val");
const alphaInput = document.getElementById("alpha-input");
const alphaVal = document.getElementById("alpha-val");
const betaInput = document.getElementById("beta-input");
const betaVal = document.getElementById("beta-val");
const lambdaInput = document.getElementById("lambda-input");
const lambdaVal = document.getElementById("lambda-val");
const sliceSlider = document.getElementById("slice-slider");
const sliceVal = document.getElementById("slice-val");
const toggleOverlay = document.getElementById("toggle-overlay");
const sliceImg = document.getElementById("slice-img");
const sampleMetaEl = document.getElementById("sample-meta");

// Metric cards
const statDiscoveries = document.getElementById("stat-discoveries");
const statPct = document.getElementById("stat-pct");
const statFdr = document.getElementById("stat-fdr");
const statLis = document.getElementById("stat-lis");
const statTime = document.getElementById("stat-time");
const statMethod = document.getElementById("stat-method");
const benchmarkBody = document.getElementById("benchmark-body");

// Event Listeners for Sliders
snrInput.addEventListener("input", (e) => snrVal.textContent = e.target.value);
clustersInput.addEventListener("input", (e) => clustersVal.textContent = e.target.value);
alphaInput.addEventListener("input", (e) => alphaVal.textContent = e.target.value);
betaInput.addEventListener("input", (e) => betaVal.textContent = e.target.value);
lambdaInput.addEventListener("input", (e) => lambdaVal.textContent = e.target.value);

sliceSlider.addEventListener("input", (e) => {
    sliceVal.textContent = e.target.value;
    updateSliceImage();
});

toggleOverlay.addEventListener("change", () => updateSliceImage());

function updateSliceImage() {
    const sliceIdx = sliceSlider.value;
    const showDisc = toggleOverlay.checked;
    sliceImg.src = `/api/v1/slice?sample_id=${currentSampleId}&slice_idx=${sliceIdx}&show_discoveries=${showDisc}&t=${Date.now()}`;
}

// 1. Health Check
async function checkHealth() {
    try {
        const res = await fetch("/api/v1/health");
        if (res.ok) {
            const data = await res.json();
            apiStatusEl.textContent = `API: Online (v${data.version})`;
        } else {
            apiStatusEl.textContent = "API: Offline / Error";
        }
    } catch (e) {
        apiStatusEl.textContent = "API: Standalone Mode (Dev)";
    }
}

// 2. Generate Synthetic Volume
document.getElementById("btn-generate-data").addEventListener("click", async () => {
    sampleMetaEl.textContent = "Generating 3D spatial random field...";
    try {
        const res = await fetch("/api/v1/synthetic/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                grid_size: [48, 48, 24],
                snr: parseFloat(snrInput.value),
                num_clusters: parseInt(clustersInput.value),
                seed: 42
            })
        });
        const data = await res.json();
        currentSampleId = data.sample_id;
        maxSliceIndex = data.shape[2] - 1;
        sliceSlider.max = maxSliceIndex;
        sliceSlider.value = Math.floor(maxSliceIndex / 2);
        sliceVal.textContent = sliceSlider.value;

        sampleMetaEl.textContent = `Sample ID: ${data.sample_id} | Shape: [${data.shape.join(", ")}] | Active: ${data.true_active_voxels} voxels (${(100 * data.true_active_voxels / data.total_voxels).toFixed(1)}%)`;
        updateSliceImage();
    } catch (e) {
        sampleMetaEl.textContent = "Error generating volume: " + e;
    }
});

// 3. Run Inference
document.getElementById("btn-run-inference").addEventListener("click", async () => {
    statDiscoveries.textContent = "...";
    try {
        const res = await fetch("/api/v1/inference", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sample_id: currentSampleId,
                nominal_alpha: parseFloat(alphaInput.value),
                use_dl_prior: document.getElementById("toggle-dl-prior").checked,
                adaptive_tuning: document.getElementById("toggle-adaptive").checked,
                beta: parseFloat(betaInput.value),
                lambda_dl: parseFloat(lambdaInput.value)
            })
        });
        const data = await res.json();

        statDiscoveries.textContent = data.num_discoveries;
        statPct.textContent = `${data.discovery_percentage.toFixed(2)}% of total voxels`;
        statFdr.textContent = data.estimated_fdr.toFixed(4);
        statLis.textContent = data.lis_cutoff_threshold.toFixed(4);
        statTime.textContent = `${data.execution_time_seconds.toFixed(2)}s`;
        statMethod.textContent = data.method;

        // Update slider values to reflect adapted parameters
        betaInput.value = data.beta_used.toFixed(2);
        betaVal.textContent = data.beta_used.toFixed(2);
        lambdaInput.value = data.lambda_dl_used.toFixed(2);
        lambdaVal.textContent = data.lambda_dl_used.toFixed(2);

        updateSliceImage();
    } catch (e) {
        statDiscoveries.textContent = "Error";
        console.error(e);
    }
});

// 4. Run Benchmark
document.getElementById("btn-run-benchmark").addEventListener("click", async () => {
    benchmarkBody.innerHTML = `<tr><td colspan="8" class="text-center">Computing benchmark across 5 candidate methods... please wait...</td></tr>`;
    try {
        const res = await fetch(`/api/v1/benchmark?snr=${snrInput.value}&nominal_alpha=${alphaInput.value}`, {
            method: "POST"
        });
        const data = await res.json();

        let rows = "";
        for (const item of data.results) {
            rows += `
                <tr>
                    <td><strong>${item.method}</strong></td>
                    <td>${item.empirical_fdr.toFixed(4)}</td>
                    <td>${item.power_tpr.toFixed(4)}</td>
                    <td>${item.fndr.toFixed(4)}</td>
                    <td>${item.f1_score.toFixed(4)}</td>
                    <td>${item.dice.toFixed(4)}</td>
                    <td>${item.mcc.toFixed(4)}</td>
                    <td>${item.num_discoveries}</td>
                </tr>
            `;
        }
        benchmarkBody.innerHTML = rows;
    } catch (e) {
        benchmarkBody.innerHTML = `<tr><td colspan="8" class="text-center">Benchmark failed: ${e}</td></tr>`;
    }
});

// Initialize on page load
checkHealth();
