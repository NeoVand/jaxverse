/** Ink primitives for the 28×28 pads. Two chapters draw digits by hand — the
 * classifier's pad and the map's query — and both must lay ink the same way,
 * because a stroke drawn one way and read another is a bug the reader would
 * blame on the model. */

export const SIDE = 28;
export const DIM = SIDE * SIDE;

/** Nib width, as the gaussian's σ in 28-grid pixels. The default sits at
 * MNIST's own stroke weight; the top of the range is a fat marker, not a
 * roller — past σ ≈ 1.5 a digit drawn at this size stops having a shape. */
export const DEFAULT_SIGMA = 0.5;
export const SIGMA_MIN = 0.3;
export const SIGMA_MAX = 1.5;

/** Add a soft dab of ink at (x, y), in 28-pixel coordinates. */
export function stampDab(
	px: Float32Array,
	x: number,
	y: number,
	sigma = DEFAULT_SIGMA,
	strength = 0.55
): void {
	const r = Math.ceil(sigma * 2.6);
	const cx = Math.round(x);
	const cy = Math.round(y);
	for (let j = cy - r; j <= cy + r; j++) {
		if (j < 0 || j >= SIDE) continue;
		for (let i = cx - r; i <= cx + r; i++) {
			if (i < 0 || i >= SIDE) continue;
			const d2 = (i - x) * (i - x) + (j - y) * (j - y);
			const k = j * SIDE + i;
			px[k] = Math.min(1, px[k] + strength * Math.exp(-d2 / (2 * sigma * sigma)));
		}
	}
}

/** Dabs interpolated along the segment so fast strokes stay solid. */
export function stampLine(
	px: Float32Array,
	x0: number,
	y0: number,
	x1: number,
	y1: number,
	sigma = DEFAULT_SIGMA
): void {
	const d = Math.hypot(x1 - x0, y1 - y0);
	const steps = Math.max(1, Math.ceil(d / 0.35));
	for (let s = 1; s <= steps; s++) {
		const t = s / steps;
		stampDab(px, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, sigma);
	}
}

/** Integer shift that moves the ink's center of mass to the image center.
 * MNIST digits are centered this way — without the same courtesy, a digit
 * drawn in a corner sits far from anything the model has ever seen. */
export function centerShift(px: Float32Array): { dx: number; dy: number } {
	let sum = 0;
	let sx = 0;
	let sy = 0;
	for (let j = 0; j < SIDE; j++) {
		for (let i = 0; i < SIDE; i++) {
			const v = px[j * SIDE + i];
			sum += v;
			sx += v * i;
			sy += v * j;
		}
	}
	if (sum < 1e-6) return { dx: 0, dy: 0 };
	const mid = (SIDE - 1) / 2;
	return { dx: Math.round(mid - sx / sum), dy: Math.round(mid - sy / sum) };
}

/** Copy shifted by (dx, dy); pixels shifted in from outside are zero. */
export function shiftImage(px: Float32Array, dx: number, dy: number): Float32Array {
	const out = new Float32Array(DIM);
	if (dx === 0 && dy === 0) {
		out.set(px);
		return out;
	}
	for (let j = 0; j < SIDE; j++) {
		const sj = j - dy;
		if (sj < 0 || sj >= SIDE) continue;
		for (let i = 0; i < SIDE; i++) {
			const si = i - dx;
			if (si < 0 || si >= SIDE) continue;
			out[j * SIDE + i] = px[sj * SIDE + si];
		}
	}
	return out;
}

/** Ink as the pad hands it to a model: centred by mass, like every digit in
 * the training set. Returns the shift too, for painting evidence back over
 * the stroke as drawn. */
export function centered(px: Float32Array): { x: Float32Array; dx: number; dy: number } {
	const { dx, dy } = centerShift(px);
	return { x: shiftImage(px, dx, dy), dx, dy };
}
