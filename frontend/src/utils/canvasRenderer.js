import { temperatureColormap } from "./colormap";

/**
 * Composite multi-layer neuroimaging scientific slice onto an HTML5 Canvas.
 * Strictly preserves anatomical visibility and raw scientific fidelity.
 */
export function renderSliceToCanvas(canvas, sliceData, options = {}) {
  if (!canvas || !sliceData || !sliceData.mri_slice) return;

  const {
    showMri = true,
    mriOpacity = 1.0,
    showRoi = true,
    roiOpacity = 0.45,
    showStat = false,
    statThreshold = 2.0,
    showLis = false,
    lisThreshold = 0.1,
    showFdr = true,
    heatmapOpacity = 0.75,
    showGroundTruth = false,
    showBaseline = false,
    crosshair = null, // { x, y }
  } = options;

  const mri = sliceData.mri_slice;
  const height = mri.length;
  const width = mri[0]?.length || 0;
  if (height === 0 || width === 0) return;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  // Determine MRI dynamic range
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = mri[y][x];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const range = maxVal - minVal > 1e-6 ? maxVal - minVal : 1.0;

  const stat = sliceData.statistical_slice;
  const lis = sliceData.lis_slice;
  const roi = sliceData.roi_slice;
  const sig = sliceData.significant_slice;
  const gt = sliceData.ground_truth_slice;
  const baseSig = sliceData.baseline_significant_slice;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;

      // 1. Base Grayscale Anatomical MRI
      let gray = 0;
      if (showMri) {
        const norm = (mri[y][x] - minVal) / range;
        gray = Math.round(norm * 255 * mriOpacity);
      }

      let r = gray;
      let g = gray;
      let b = gray;
      let a = 255;

      // 2. 3D U-Net ROI Layer (Translucent Cyan: [6, 182, 212])
      if (showRoi && roi && roi[y] && roi[y][x] > 0) {
        const roia = roiOpacity;
        r = Math.round(r * (1 - roia) + 6 * roia);
        g = Math.round(g * (1 - roia) + 182 * roia);
        b = Math.round(b * (1 - roia) + 212 * roia);
      }

      // 3. Ground Truth Layer (Translucent Magenta: [236, 72, 153])
      if (showGroundTruth && gt && gt[y] && gt[y][x] > 0) {
        const gta = 0.45;
        r = Math.round(r * (1 - gta) + 236 * gta);
        g = Math.round(g * (1 - gta) + 72 * gta);
        b = Math.round(b * (1 - gta) + 153 * gta);
      }

      // 4. Baseline Significant Layer (Translucent Royal Blue: [59, 130, 246])
      if (showBaseline && baseSig && baseSig[y] && baseSig[y][x] > 0) {
        const baseA = 0.5;
        r = Math.round(r * (1 - baseA) + 59 * baseA);
        g = Math.round(g * (1 - baseA) + 130 * baseA);
        b = Math.round(b * (1 - baseA) + 246 * baseA);
      }

      // 5. Statistical / LIS / FDR Heatmap (Temperature Orange-Yellow)
      let heatVal = null;
      if (showFdr && sig && sig[y] && sig[y][x] > 0) {
        // Highlighting FDR significant voxels
        const z = stat ? Math.abs(stat[y][x]) : 3.0;
        heatVal = Math.min(1.0, (z - 1.96) / 4.0); // normalized significance strength
      } else if (showLis && lis && lis[y]) {
        // Highlighting low LIS (1 - LIS)
        const lVal = lis[y][x];
        if (lVal <= lisThreshold) {
          heatVal = 1.0 - lVal / Math.max(1e-4, lisThreshold);
        }
      } else if (showStat && stat && stat[y]) {
        // Raw z-score threshold
        const z = Math.abs(stat[y][x]);
        if (z >= statThreshold) {
          heatVal = Math.min(1.0, (z - statThreshold) / 4.0);
        }
      }

      if (heatVal !== null) {
        const [hr, hg, hb] = temperatureColormap(heatVal);
        const ha = heatmapOpacity;
        r = Math.round(r * (1 - ha) + hr * ha);
        g = Math.round(g * (1 - ha) + hg * ha);
        b = Math.round(b * (1 - ha) + hb * ha);
      }

      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = a;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Optional Crosshair
  if (crosshair && crosshair.x >= 0 && crosshair.x < width && crosshair.y >= 0 && crosshair.y < height) {
    ctx.strokeStyle = "rgba(6, 182, 212, 0.75)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(0, crosshair.y + 0.5);
    ctx.lineTo(width, crosshair.y + 0.5);
    ctx.stroke();

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(crosshair.x + 0.5, 0);
    ctx.lineTo(crosshair.x + 0.5, height);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
