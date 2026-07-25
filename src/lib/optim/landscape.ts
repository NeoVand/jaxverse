/**
 * Loss landscapes for the descent race — pure math, no DOM.
 *
 * Conventions: one grid per preset covering the visible window extended 20%
 * each side (contours run past the frame instead of closing on it), samples
 * at cell centers (the d3-contour convention), color/threshold normalization
 * in LOG space over the VISIBLE subregion only.
 */

export interface Pt {
	x: number;
	y: number;
}

export type PresetId = 'basins' | 'ravine' | 'saddle';

export interface Preset {
	id: PresetId;
	label: string;
	/** Visible parameter window. Span ratio ≈ φ (1.62) so angles survive
	 *  projection onto the golden-ratio canvas. */
	xMin: number;
	xMax: number;
	yMin: number;
	yMax: number;
	/** Default θ₀ — hand-tuned so the race tells its story at the default γ. */
	start: Pt;
	f(x: number, y: number): number;
}

export const PRESETS: Preset[] = [
	{
		// A curved trough (y ≈ 0.45x² − 0.9) carrying a double well along its
		// floor, tilted so the far basin is worth reaching, with a mild ripple
		// for texture.
		id: 'basins',
		label: 'Two basins',
		xMin: -4.37,
		xMax: 4.37,
		yMin: -2.7,
		yMax: 2.7,
		start: { x: 2.7, y: 2.3 },
		f(x, y) {
			const bend = y - 0.45 * x * x + 0.9;
			const well = x * x - 2.4;
			return (
				1.7 * bend * bend +
				0.035 * well * well +
				0.18 * x +
				0.25 * Math.sin(2.1 * x) * Math.cos(1.7 * y) +
				0.35
			);
		}
	},
	{
		// Rosenbrock-style banana ravine: steep walls across y = x², a nearly
		// flat floor along it, minimum at (1, 1). SGD zigzags across the
		// trench; momentum surfs it; adaptive methods walk the floor.
		id: 'ravine',
		label: 'Ravine',
		xMin: -2.187,
		xMax: 2.187,
		yMin: -0.85,
		yMax: 1.85,
		start: { x: -1.45, y: 1.6 },
		f(x, y) {
			const d = y - x * x;
			return 0.03 * (1 - x) * (1 - x) + 1.2 * d * d;
		}
	},
	{
		// A saddle at the origin (curves up along x, down along y) with a
		// quartic to bottom out into two basins at y ≈ ±2.24. Everything
		// stalls on the ridge; whoever builds y-velocity first escapes.
		id: 'saddle',
		label: 'Saddle',
		xMin: -4.37,
		xMax: 4.37,
		yMin: -2.7,
		yMax: 2.7,
		start: { x: -2.6, y: 0.12 },
		f(x, y) {
			return 0.28 * x * x - 0.5 * y * y + 0.05 * y * y * y * y + 1.4;
		}
	}
];

export function presetById(id: PresetId): Preset {
	return PRESETS.find((p) => p.id === id) ?? PRESETS[0];
}

/** Numeric gradient — central differences, h = 1e-4. */
const H_DIFF = 1e-4;
export function gradAt(preset: Preset, x: number, y: number): [number, number] {
	return [
		(preset.f(x + H_DIFF, y) - preset.f(x - H_DIFF, y)) / (2 * H_DIFF),
		(preset.f(x, y + H_DIFF) - preset.f(x, y - H_DIFF)) / (2 * H_DIFF)
	];
}

/** log(loss + 0.001) is the mapping the heatmap and contour levels share. */
export const LOG_EPS = 0.001;
export const GRID_RES = 110;
export const GRID_EXTEND = 0.2;
export const CONTOUR_LEVELS = 16;

export interface LossGrid {
	res: number;
	extXMin: number;
	extXMax: number;
	extYMin: number;
	extYMax: number;
	/** values[j * res + i]; i → x ascending, j → y ascending; cell centers. */
	values: Float64Array;
	/** Loss extrema over the visible subregion (drives colors + contours). */
	visMin: number;
	visMax: number;
	logMin: number;
	logMax: number;
}

export function computeGrid(
	preset: Preset,
	res: number = GRID_RES,
	extend: number = GRID_EXTEND
): LossGrid {
	const spanX = preset.xMax - preset.xMin;
	const spanY = preset.yMax - preset.yMin;
	const extXMin = preset.xMin - spanX * extend;
	const extXMax = preset.xMax + spanX * extend;
	const extYMin = preset.yMin - spanY * extend;
	const extYMax = preset.yMax + spanY * extend;
	const extSpanX = extXMax - extXMin;
	const extSpanY = extYMax - extYMin;

	const values = new Float64Array(res * res);
	let visMin = Infinity;
	let visMax = -Infinity;

	for (let j = 0; j < res; j++) {
		const y = extYMin + ((j + 0.5) / res) * extSpanY;
		const yVisible = y >= preset.yMin && y <= preset.yMax;
		for (let i = 0; i < res; i++) {
			const x = extXMin + ((i + 0.5) / res) * extSpanX;
			let loss = preset.f(x, y);
			if (!Number.isFinite(loss)) loss = 1e300;
			values[j * res + i] = loss;
			if (yVisible && x >= preset.xMin && x <= preset.xMax) {
				if (loss < visMin) visMin = loss;
				if (loss > visMax) visMax = loss;
			}
		}
	}

	if (!Number.isFinite(visMin) || visMin === Infinity) {
		visMin = 0;
		visMax = 1;
	}

	return {
		res,
		extXMin,
		extXMax,
		extYMin,
		extYMax,
		values,
		visMin,
		visMax,
		logMin: safeLog(visMin),
		logMax: safeLog(visMax)
	};
}

function safeLog(v: number): number {
	return Math.log(Math.max(v + LOG_EPS, 1e-9));
}

/** Normalized log-height t ∈ [0, 1] — the exact mapping the heatmap uses. */
export function normalizedLogLoss(grid: LossGrid, loss: number): number {
	const span = grid.logMax - grid.logMin || 1;
	const t = (safeLog(loss) - grid.logMin) / span;
	return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Log-spaced contour levels between the grid's visible min/max. */
export function contourThresholds(grid: LossGrid, count: number = CONTOUR_LEVELS): number[] {
	const out: number[] = [];
	for (let k = 1; k < count; k++) {
		out.push(Math.exp(grid.logMin + (k / count) * (grid.logMax - grid.logMin)) - LOG_EPS);
	}
	return out;
}
