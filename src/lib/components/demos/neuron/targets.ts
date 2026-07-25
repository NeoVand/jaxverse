// The curve workshop's fixed sample grid and its cast of target curves.
// Data x's are evenly spaced but *stored* in a shuffled order, so the engine's
// held-out tail (valFraction) is an unbiased sample of the interval.

export const DATA_N = 256;
export const GRID_N = 160;

export type PresetId = 'sine' | 'bumps' | 'step' | 'abs';
export type TargetId = PresetId | 'draw';

export const PRESETS: ReadonlyArray<{ id: PresetId; label: string; fn: (x: number) => number }> = [
	{ id: 'sine', label: 'sine', fn: (x) => 0.7 * Math.sin(3.1 * x) },
	{
		id: 'bumps',
		label: 'bumps',
		fn: (x) =>
			0.8 * Math.exp(-(((x + 0.45) / 0.22) ** 2)) - 0.8 * Math.exp(-(((x - 0.45) / 0.22) ** 2))
	},
	{ id: 'step', label: 'step', fn: (x) => 0.6 * Math.tanh(12 * x) },
	{ id: 'abs', label: '|x|', fn: (x) => 0.9 * Math.abs(x) - 0.45 }
];

/** n points evenly spaced over [-1, 1], in sorted order. */
export function evenXs(n: number): Float32Array {
	const xs = new Float32Array(n);
	for (let i = 0; i < n; i++) xs[i] = -1 + (2 * i) / (n - 1);
	return xs;
}

function mulberry32(seed: number) {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** A fixed shuffle: storage index → sorted index. */
export function makePerm(n: number, seed: number): Uint16Array {
	const rand = mulberry32(seed);
	const p = new Uint16Array(n);
	for (let i = 0; i < n; i++) p[i] = i;
	for (let i = n - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		const t = p[i];
		p[i] = p[j];
		p[j] = t;
	}
	return p;
}
