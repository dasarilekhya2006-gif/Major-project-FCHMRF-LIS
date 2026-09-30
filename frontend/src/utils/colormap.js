/**
 * Scientific colormaps and coordinate transformations for neuroimaging visualization.
 * Preserves raw numerical values without artificial enhancement.
 */

/**
 * Temperature Colormap (Dark Orange -> Orange -> Vivid Yellow)
 * @param {number} t - Normalized value between 0 and 1
 * @returns {[number, number, number]} [R, G, B]
 */
export function temperatureColormap(t) {
  const clamped = Math.max(0, Math.min(1, t));

  let r, g, b;
  if (clamped < 0.33) {
    // Deep reddish-orange to standard orange
    const f = clamped / 0.33;
    r = Math.round(180 + (240 - 180) * f);
    g = Math.round(40 + (100 - 40) * f);
    b = 0;
  } else if (clamped < 0.75) {
    // Standard orange to bright amber
    const f = (clamped - 0.33) / (0.75 - 0.33);
    r = Math.round(240 + (255 - 240) * f);
    g = Math.round(100 + (185 - 100) * f);
    b = Math.round(0 + (20 - 0) * f);
  } else {
    // Bright amber to vivid luminous yellow
    const f = (clamped - 0.75) / (1.0 - 0.75);
    r = 255;
    g = Math.round(185 + (245 - 185) * f);
    b = Math.round(20 + (80 - 20) * f);
  }

  return [r, g, b];
}

/**
 * Apply affine transformation to map voxel indices (x, y, z) to physical world coordinates (X, Y, Z) in mm.
 * @param {number} x
 * @param {number} y
 * @param {number} z
 * @param {number[][]} affine 4x4 matrix
 * @returns {[number, number, number]}
 */
export function voxelToWorld(x, y, z, affine) {
  if (!affine || affine.length !== 4) {
    return [x, y, z];
  }
  const X = affine[0][0] * x + affine[0][1] * y + affine[0][2] * z + affine[0][3];
  const Y = affine[1][0] * x + affine[1][1] * y + affine[1][2] * z + affine[1][3];
  const Z = affine[2][0] * x + affine[2][1] * y + affine[2][2] * z + affine[2][3];
  return [Number(X.toFixed(2)), Number(Y.toFixed(2)), Number(Z.toFixed(2))];
}
