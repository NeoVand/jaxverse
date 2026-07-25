/**
 * Heatmap colorization — the loss grid as a quiet monochrome wash. Loss maps
 * in log space (same normalization as the contour thresholds), and each pixel
 * interpolates between the page's --paper and a low-alpha ramp of --ink:
 * basins collect the most ink, the far field melts into the page. Both token
 * colors are resolved by the caller at draw time, so the buffer follows theme
 * switches for free.
 *
 * Pure byte-pushing — no DOM. The component wraps the RGBA buffer into an
 * ImageData/canvas inside an effect, so this stays SSR-safe and testable.
 */

import { LOG_EPS, type LossGrid } from './landscape';

export type RGB = [number, number, number];

/** Ink share at the basin floor — the wash never gets louder than this. */
const INK_MAX = 0.16;
/** Shading gamma: keeps mid-heights airy so only true lows collect ink. */
const INK_POW = 1.6;

/** Parse a resolved CSS color — `#rgb`, `#rrggbb`, or `rgb(a)(r, g, b)`. */
export function parseColor(s: string): RGB {
	const c = s.trim();
	if (c.startsWith('#')) {
		if (c.length === 4) {
			return [parseInt(c[1] + c[1], 16), parseInt(c[2] + c[2], 16), parseInt(c[3] + c[3], 16)];
		}
		return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
	}
	const m = c.match(/[\d.]+/g);
	return m ? [Math.round(+m[0]), Math.round(+m[1]), Math.round(+m[2])] : [128, 128, 128];
}

/**
 * Colorize the grid into an opaque RGBA buffer, one pixel per cell, row 0 at
 * the TOP of the plot (canvas y points down, parameter y points up). Pixels
 * are paper with up to INK_MAX of ink mixed in at the basin floor.
 */
export function heatmapRGBA(grid: LossGrid, paper: RGB, ink: RGB) {
	const { res, values, logMin, logMax } = grid;
	const out = new Uint8ClampedArray(res * res * 4);
	const logSpan = logMax - logMin || 1;

	for (let j = 0; j < res; j++) {
		const row = res - 1 - j;
		for (let i = 0; i < res; i++) {
			let t = (Math.log(Math.max(values[j * res + i] + LOG_EPS, 1e-9)) - logMin) / logSpan;
			t = t < 0 ? 0 : t > 1 ? 1 : t;
			const w = INK_MAX * Math.pow(1 - t, INK_POW);
			const o = (row * res + i) * 4;
			out[o] = paper[0] + (ink[0] - paper[0]) * w;
			out[o + 1] = paper[1] + (ink[1] - paper[1]) * w;
			out[o + 2] = paper[2] + (ink[2] - paper[2]) * w;
			out[o + 3] = 255;
		}
	}
	return out;
}
