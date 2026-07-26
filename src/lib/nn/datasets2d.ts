// Two-dimensional toy datasets for the classification playground — the cast
// of characters in “Bending Space”. All coordinates live in [-1, 1]².

export type Dataset2dId = 'blobs' | 'waves' | 'moons' | 'circles' | 'xor' | 'yinyang' | 'spirals';

export interface Dataset2d {
	x: Float32Array; // n × 2
	labels: Int32Array; // n, values 0 | 1
	n: number;
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

/** Box–Muller gaussian from a uniform rng. */
function gauss(rand: () => number): number {
	let u = 0;
	let v = 0;
	while (u === 0) u = rand();
	while (v === 0) v = rand();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Interleave the two classes and shuffle, so train/val tails are unbiased. */
function assemble(pts: Array<[number, number, number]>, rand: () => number): Dataset2d {
	for (let i = pts.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[pts[i], pts[j]] = [pts[j], pts[i]];
	}
	const n = pts.length;
	const x = new Float32Array(n * 2);
	const labels = new Int32Array(n);
	pts.forEach(([px, py, l], i) => {
		x[2 * i] = px;
		x[2 * i + 1] = py;
		labels[i] = l;
	});
	return { x, labels, n };
}

export function makeDataset2d(id: Dataset2dId, n = 400, noise = 0.06, seed = 7): Dataset2d {
	const rand = mulberry32(seed);
	const half = Math.floor(n / 2);
	const pts: Array<[number, number, number]> = [];

	if (id === 'blobs') {
		for (let i = 0; i < half; i++)
			pts.push([-0.45 + gauss(rand) * (0.18 + noise), -0.4 + gauss(rand) * (0.18 + noise), 0]);
		for (let i = 0; i < half; i++)
			pts.push([0.45 + gauss(rand) * (0.18 + noise), 0.4 + gauss(rand) * (0.18 + noise), 1]);
	} else if (id === 'circles') {
		for (let i = 0; i < half; i++) {
			const a = rand() * Math.PI * 2;
			const r = 0.28 * Math.sqrt(rand()) + gauss(rand) * noise * 0.5;
			pts.push([Math.cos(a) * r, Math.sin(a) * r, 0]);
		}
		for (let i = 0; i < half; i++) {
			const a = rand() * Math.PI * 2;
			const r = 0.75 + gauss(rand) * noise;
			pts.push([Math.cos(a) * r, Math.sin(a) * r, 1]);
		}
	} else if (id === 'moons') {
		for (let i = 0; i < half; i++) {
			const a = rand() * Math.PI;
			pts.push([
				Math.cos(a) * 0.6 - 0.25 + gauss(rand) * noise,
				Math.sin(a) * 0.6 - 0.22 + gauss(rand) * noise,
				0
			]);
		}
		for (let i = 0; i < half; i++) {
			const a = rand() * Math.PI;
			pts.push([
				0.25 - Math.cos(a) * 0.6 + gauss(rand) * noise,
				0.22 - Math.sin(a) * 0.6 + gauss(rand) * noise,
				1
			]);
		}
	} else if (id === 'waves') {
		// two bands separated by a sine ridge — one cut is enough, if it can bend
		let placed = 0;
		while (placed < n) {
			const px = (rand() * 2 - 1) * 0.95;
			const py = (rand() * 2 - 1) * 0.95;
			const s = py - 0.42 * Math.sin(3.1 * px);
			if (Math.abs(s) < 0.08 + noise) continue; // carve a margin along the ridge
			pts.push([px, py, s > 0 ? 1 : 0]);
			placed++;
		}
	} else if (id === 'xor') {
		// four quadrants, opposite corners allied — the classic xor
		for (let i = 0; i < n; i++) {
			let px = (rand() * 2 - 1) * 0.88;
			let py = (rand() * 2 - 1) * 0.88;
			px += Math.sign(px) * 0.07;
			py += Math.sign(py) * 0.07;
			pts.push([
				px + gauss(rand) * noise * 0.3,
				py + gauss(rand) * noise * 0.3,
				px * py > 0 ? 0 : 1
			]);
		}
	} else if (id === 'yinyang') {
		// the taijitu: two lobes chasing each other, each carrying the other's eye
		const R = 0.92;
		let placed = 0;
		while (placed < n) {
			const a = rand() * Math.PI * 2;
			const r = R * Math.sqrt(rand());
			const px = Math.cos(a) * r;
			const py = Math.sin(a) * r;
			const dTop = Math.hypot(px, py - R / 2);
			const dBot = Math.hypot(px, py + R / 2);
			let l: number;
			if (dTop < R * 0.17) l = 1;
			else if (dBot < R * 0.17) l = 0;
			else if (dTop < R / 2) l = 0;
			else if (dBot < R / 2) l = 1;
			else l = px > 0 ? 0 : 1;
			pts.push([px + gauss(rand) * noise * 0.4, py + gauss(rand) * noise * 0.4, l]);
			placed++;
		}
	} else {
		// two interlocking spirals — the chapter’s final boss
		for (let c = 0; c < 2; c++) {
			for (let i = 0; i < half; i++) {
				const t = (i / half) * 3.2 * Math.PI + 0.4;
				const r = 0.045 + (0.82 * t) / (3.2 * Math.PI + 0.4);
				const a = t + c * Math.PI;
				pts.push([
					Math.cos(a) * r + gauss(rand) * noise * 0.5,
					Math.sin(a) * r + gauss(rand) * noise * 0.5,
					c
				]);
			}
		}
	}
	return assemble(pts, rand);
}

/** Ordered easiest → hardest; the order sets the thumbnail rail. */
export const DATASET_LABELS: Record<Dataset2dId, string> = {
	blobs: 'Two clouds',
	waves: 'Waves',
	moons: 'Two moons',
	circles: 'Rings',
	xor: 'XOR',
	yinyang: 'Yin-yang',
	spirals: 'Spirals'
};
