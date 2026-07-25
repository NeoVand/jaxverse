// Pure helpers for the digit plates: crisp 28×28 blitting, tiny math, and the
// drawing pad's brush physics. Reactive state lives in digits-context.svelte.ts;
// the only thing kept here is one reused scratch canvas.

export const SIDE = 28;
export const DIM = SIDE * SIDE;

export type Rgb = [number, number, number];

/** Design tokens resolved at draw time so canvases follow theme flips. */
export function readTokens(el: Element) {
	const s = getComputedStyle(el);
	const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback;
	return {
		paper: v('--paper', '#faf9f5'),
		ink: v('--ink', '#1d1c18'),
		ink2: v('--ink-2', '#605d54'),
		ink3: v('--ink-3', '#a3a094'),
		line: v('--line', '#e5e2d8'),
		accent: v('--accent', '#2b45d8'),
		warm: v('--warm', '#d3541f'),
		good: v('--good', '#22774d'),
		bad: v('--bad', '#bb3a2b'),
		cats: Array.from({ length: 10 }, (_, d) => v(`--cat-${d}`, '#888888'))
	};
}

/** Parse #rrggbb → [r, g, b]; tolerates whitespace. */
export function hexRgb(hex: string): Rgb {
	const h = hex.replace('#', '').trim();
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Numerically stable softmax over `len` values starting at `off`. */
export function softmax(z: ArrayLike<number>, off = 0, len = 10): number[] {
	let mx = -Infinity;
	for (let j = 0; j < len; j++) mx = Math.max(mx, z[off + j]);
	const e: number[] = [];
	let sum = 0;
	for (let j = 0; j < len; j++) {
		const v = Math.exp(z[off + j] - mx);
		e.push(v);
		sum += v;
	}
	return e.map((v) => v / sum);
}

export function argmax(z: ArrayLike<number>, off = 0, len = 10): number {
	let best = 0;
	let bv = -Infinity;
	for (let j = 0; j < len; j++) {
		if (z[off + j] > bv) {
			bv = z[off + j];
			best = j;
		}
	}
	return best;
}

// ── crisp 28×28 blitting ─────────────────────────────────────────────────────

let scratch: CanvasRenderingContext2D | null = null;
function scratchCtx(): CanvasRenderingContext2D | null {
	if (!scratch) {
		const el = document.createElement('canvas');
		el.width = SIDE;
		el.height = SIDE;
		scratch = el.getContext('2d');
	}
	return scratch;
}

/** Compose SIDE×SIDE ImageData layers onto `canvas`: DPR-aware (capped at 2),
 * smoothing off so pixels stay square. `css` is the layout-size fallback. */
export function blit(canvas: HTMLCanvasElement, layers: ImageData[], css = 84): void {
	const size = canvas.clientWidth || css;
	const dpr = Math.min(typeof devicePixelRatio === 'number' ? devicePixelRatio : 1, 2);
	const w = Math.max(SIDE, Math.round(size * dpr));
	if (canvas.width !== w || canvas.height !== w) {
		canvas.width = w;
		canvas.height = w;
	}
	const ctx = canvas.getContext('2d');
	const s = scratchCtx();
	if (!ctx || !s) return;
	ctx.imageSmoothingEnabled = false;
	ctx.clearRect(0, 0, w, w);
	for (const img of layers) {
		s.putImageData(img, 0, 0);
		ctx.drawImage(s.canvas, 0, 0, w, w);
	}
}

/** Intensities 0..1 → ink-colored pixels; the value rides in the alpha. */
export function inkImage(px: ArrayLike<number>, off: number, ink: Rgb, alpha = 1): ImageData {
	const img = new ImageData(SIDE, SIDE);
	for (let i = 0; i < DIM; i++) {
		const k = 4 * i;
		img.data[k] = ink[0];
		img.data[k + 1] = ink[1];
		img.data[k + 2] = ink[2];
		img.data[k + 3] = Math.round(Math.min(1, Math.max(0, px[off + i])) * alpha * 255);
	}
	return img;
}

/** Signed values → diverging pixels: `pos` above zero, `neg` below, alpha
 * proportional to |v| normalized per tile (each tile sets its own contrast). */
export function divergingImage(
	vals: ArrayLike<number>,
	off: number,
	pos: Rgb,
	neg: Rgb,
	alphaMax = 0.9
): ImageData {
	let m = 1e-12;
	for (let i = 0; i < DIM; i++) m = Math.max(m, Math.abs(vals[off + i]));
	const img = new ImageData(SIDE, SIDE);
	for (let i = 0; i < DIM; i++) {
		const v = vals[off + i];
		const c = v >= 0 ? pos : neg;
		const k = 4 * i;
		img.data[k] = c[0];
		img.data[k + 1] = c[1];
		img.data[k + 2] = c[2];
		img.data[k + 3] = Math.round((Math.abs(v) / m) * alphaMax * 255);
	}
	return img;
}

// ── the drawing pad's brush ──────────────────────────────────────────────────

/** σ ≈ 0.75 gives a ≈1.5 px soft gaussian — MNIST's own stroke weight. */
export const DEFAULT_SIGMA = 0.75;

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

// ── sparklines ───────────────────────────────────────────────────────────────

/** SVG path for a tiny sparkline; log scale when `logY`. */
export function sparkPath(vals: number[], w: number, h: number, logY: boolean): string {
	if (vals.length < 2) return '';
	let lo = Infinity;
	let hi = -Infinity;
	for (const v of vals) {
		const y = logY ? Math.log(Math.max(v, 1e-4)) : v;
		lo = Math.min(lo, y);
		hi = Math.max(hi, y);
	}
	if (hi - lo < 1e-9) hi = lo + 1e-9;
	return vals
		.map((v, i) => {
			const yv = logY ? Math.log(Math.max(v, 1e-4)) : v;
			const x = (i / (vals.length - 1)) * w;
			const y = h - ((yv - lo) / (hi - lo)) * (h - 4) - 2;
			return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
		})
		.join(' ');
}
