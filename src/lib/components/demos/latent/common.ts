// Shared drawing seams for the Hidden Map plates: theme tokens resolved at
// draw time (so canvases follow theme flips), DPR-aware sizing, the
// surface→ink ramp that turns a 784-float row into a printed digit tile,
// and log-scale sparkline paths for the loss telemetry.

export const SIDE = 28;
export const DIM = SIDE * SIDE;

export interface Tokens {
	paper: string;
	surface: string;
	ink: string;
	ink2: string;
	ink3: string;
	line: string;
	lineSoft: string;
	accent: string;
	warm: string;
	good: string;
	bad: string;
	/** --cat-0 … --cat-9, the ten digit-class categoricals. */
	cats: string[];
}

/** Design tokens read from the live cascade — call inside the draw, not once. */
export function readTokens(el: Element): Tokens {
	const s = getComputedStyle(el);
	const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback;
	const cats: string[] = [];
	for (let d = 0; d < 10; d++) cats.push(v(`--cat-${d}`, '#888888'));
	return {
		paper: v('--paper', '#faf9f5'),
		surface: v('--surface', '#ffffff'),
		ink: v('--ink', '#1d1c18'),
		ink2: v('--ink-2', '#605d54'),
		ink3: v('--ink-3', '#a3a094'),
		line: v('--line', '#e5e2d8'),
		lineSoft: v('--line-soft', '#efede4'),
		accent: v('--accent', '#2b45d8'),
		warm: v('--warm', '#d3541f'),
		good: v('--good', '#22774d'),
		bad: v('--bad', '#bb3a2b'),
		cats
	};
}

/** Parse #rrggbb → [r, g, b]; tolerates whitespace. */
export function hexRgb(hex: string): [number, number, number] {
	const h = hex.replace('#', '').trim();
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** DPR-aware canvas sizing (capped at 2) with an identity-at-dpr transform. */
export function setupCanvas(canvas: HTMLCanvasElement): {
	ctx: CanvasRenderingContext2D;
	W: number;
	H: number;
} {
	const dpr = Math.min(devicePixelRatio || 1, 2);
	const W = canvas.clientWidth;
	const H = canvas.clientHeight;
	if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
		canvas.width = W * dpr;
		canvas.height = H * dpr;
	}
	const ctx = canvas.getContext('2d')!;
	ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	return { ctx, W, H };
}

/** Paint one 28×28 digit (0..1 floats starting at `offset`) into `img` as a
 * bg→fg ramp — ink strokes on paper, inverting cleanly in dark mode. */
export function rasterizeDigit(
	img: ImageData,
	row: Float32Array,
	offset: number,
	bg: [number, number, number],
	fg: [number, number, number]
): void {
	for (let i = 0; i < DIM; i++) {
		let v = row[offset + i];
		v = v <= 0 ? 0 : v >= 1 ? 1 : v;
		const k = 4 * i;
		img.data[k] = (bg[0] + (fg[0] - bg[0]) * v) | 0;
		img.data[k + 1] = (bg[1] + (fg[1] - bg[1]) * v) | 0;
		img.data[k + 2] = (bg[2] + (fg[2] - bg[2]) * v) | 0;
		img.data[k + 3] = 255;
	}
}

/** A cached row of 28×28 offscreen tiles, re-rasterized only when the data
 * version or the theme colors change — so painters can call ensure() every
 * frame and pay for pixels only when something is actually new. */
export class TileStrip {
	private tiles: HTMLCanvasElement[] = [];
	private key = '';

	ensure(
		data: Float32Array | null,
		count: number,
		version: number,
		bg: string,
		fg: string
	): HTMLCanvasElement[] | null {
		if (!data) return null;
		const key = `${version}|${count}|${bg}|${fg}`;
		if (key !== this.key) {
			const b = hexRgb(bg);
			const f = hexRgb(fg);
			while (this.tiles.length < count) {
				const c = document.createElement('canvas');
				c.width = SIDE;
				c.height = SIDE;
				this.tiles.push(c);
			}
			for (let t = 0; t < count; t++) {
				const ctx = this.tiles[t].getContext('2d')!;
				const img = ctx.createImageData(SIDE, SIDE);
				rasterizeDigit(img, data, t * DIM, b, f);
				ctx.putImageData(img, 0, 0);
			}
			this.key = key;
		}
		return this.tiles.slice(0, count);
	}
}

/** Paint one 28×28 digit at (x0, y0) inside a larger ImageData sheet. */
export function rasterizeDigitAt(
	img: ImageData,
	x0: number,
	y0: number,
	row: Float32Array,
	offset: number,
	bg: [number, number, number],
	fg: [number, number, number]
): void {
	for (let yy = 0; yy < SIDE; yy++) {
		for (let xx = 0; xx < SIDE; xx++) {
			let v = row[offset + yy * SIDE + xx];
			v = v <= 0 ? 0 : v >= 1 ? 1 : v;
			const k = 4 * ((y0 + yy) * img.width + x0 + xx);
			img.data[k] = (bg[0] + (fg[0] - bg[0]) * v) | 0;
			img.data[k + 1] = (bg[1] + (fg[1] - bg[1]) * v) | 0;
			img.data[k + 2] = (bg[2] + (fg[2] - bg[2]) * v) | 0;
			img.data[k + 3] = 255;
		}
	}
}

/** One spritesheet holding every digit as a 28×28 tile — thumbnails are
 * drawImage'd from it, so the 2000-tile rasterization happens once per theme
 * (or tint scheme), not per frame. */
export class TileAtlas {
	canvas: HTMLCanvasElement | null = null;
	cols = 0;
	private key = '';

	/** fg per tile comes from `fgs[labels[i]]`, or `fgs[0]` when labels is null. */
	ensure(
		data: Float32Array,
		n: number,
		labels: Int32Array | null,
		bg: string,
		fgs: string[]
	): HTMLCanvasElement {
		const key = `${n}|${bg}|${fgs.join('|')}|${labels ? 'lab' : 'flat'}`;
		if (key !== this.key || !this.canvas) {
			const cols = Math.ceil(Math.sqrt(n));
			const rows = Math.ceil(n / cols);
			const c = document.createElement('canvas');
			c.width = cols * SIDE;
			c.height = rows * SIDE;
			const ctx = c.getContext('2d')!;
			const img = ctx.createImageData(c.width, c.height);
			const b = hexRgb(bg);
			const fgRgb = fgs.map(hexRgb);
			for (let i = 0; i < n; i++) {
				const fg = labels ? fgRgb[labels[i]] : fgRgb[0];
				rasterizeDigitAt(img, (i % cols) * SIDE, Math.floor(i / cols) * SIDE, data, i * DIM, b, fg);
			}
			ctx.putImageData(img, 0, 0);
			this.canvas = c;
			this.cols = cols;
			this.key = key;
		}
		return this.canvas;
	}
}

/** An orthonormal frame fitted to a cloud: the mean it is centred on, and `k`
 * unit directions stored row-major. Enough to project a latent point down for
 * display and — because the directions are orthonormal — to lift a display
 * coordinate back up to a real latent address. */
export interface Basis {
	d: number;
	k: number;
	mean: Float32Array;
	comps: Float32Array; // k × d
}

/** Top-k principal directions of an n×d cloud by power iteration with
 * deflation. The seeds are deterministic on purpose: this is refitted after
 * every trained chunk, and a basis that flipped sign between fits would make
 * the whole map jump rather than drift. */
export function pcaBasis(data: Float32Array, n: number, d: number, k: number): Basis {
	const mean = new Float64Array(d);
	for (let i = 0; i < n; i++) for (let q = 0; q < d; q++) mean[q] += data[i * d + q];
	for (let q = 0; q < d; q++) mean[q] /= Math.max(1, n);

	const matVec = (v: Float64Array, out: Float64Array) => {
		out.fill(0);
		for (let i = 0; i < n; i++) {
			let dot = 0;
			for (let q = 0; q < d; q++) dot += (data[i * d + q] - mean[q]) * v[q];
			for (let q = 0; q < d; q++) out[q] += dot * (data[i * d + q] - mean[q]);
		}
	};
	const normalize = (v: Float64Array) => {
		let s = 0;
		for (let q = 0; q < d; q++) s += v[q] * v[q];
		s = Math.sqrt(s) || 1;
		for (let q = 0; q < d; q++) v[q] /= s;
	};

	const comps = Math.min(k, d);
	const basis: Float64Array[] = [];
	for (let c = 0; c < comps; c++) {
		let v = new Float64Array(d);
		for (let q = 0; q < d; q++) v[q] = Math.sin(q * 2.3 + c * 1.7) + 0.5;
		const tmp = new Float64Array(d);
		for (let it = 0; it < 24; it++) {
			matVec(v, tmp);
			for (const b of basis) {
				let dot = 0;
				for (let q = 0; q < d; q++) dot += tmp[q] * b[q];
				for (let q = 0; q < d; q++) tmp[q] -= dot * b[q];
			}
			normalize(tmp);
			v = tmp.slice();
		}
		basis.push(v.slice());
	}

	const out: Basis = {
		d,
		k: comps,
		mean: Float32Array.from(mean),
		comps: new Float32Array(comps * d)
	};
	for (let c = 0; c < comps; c++) for (let q = 0; q < d; q++) out.comps[c * d + q] = basis[c][q];
	return out;
}

/** Project every row of an n×d cloud onto the basis — n×k out. */
export function projectAll(data: Float32Array, n: number, b: Basis): Float32Array {
	const out = new Float32Array(n * b.k);
	for (let i = 0; i < n; i++)
		for (let c = 0; c < b.k; c++) {
			let dot = 0;
			for (let q = 0; q < b.d; q++) dot += (data[i * b.d + q] - b.mean[q]) * b.comps[c * b.d + q];
			out[i * b.k + c] = dot;
		}
	return out;
}

/** Lift display coordinates back to a full latent address: mean + Σ uₖ compₖ.
 * The lifted point is the one nearest the origin among all latents that
 * project to `u`, which is the honest answer to "what lives at this spot". */
export function liftFrom(u: ArrayLike<number>, b: Basis): Float32Array {
	const z = b.mean.slice();
	for (let c = 0; c < b.k; c++) {
		const s = u[c] ?? 0;
		if (s === 0) continue;
		for (let q = 0; q < b.d; q++) z[q] += s * b.comps[c * b.d + q];
	}
	return z;
}

/** Rotate-project a 3-D latent point to view coordinates (y up);
 * the returned z is depth, used for sorting and shading. */
export function project3(
	x: number,
	y: number,
	z: number,
	yaw: number,
	pitch: number
): [number, number, number] {
	const cy = Math.cos(yaw);
	const sy = Math.sin(yaw);
	const x1 = cy * x + sy * z;
	const z1 = -sy * x + cy * z;
	const cp = Math.cos(pitch);
	const sp = Math.sin(pitch);
	const y2 = cp * y - sp * z1;
	const z2 = sp * y + cp * z1;
	return [x1, y2, z2];
}

/** Grid-occupancy subsample over the frame `center ± span`: one representative
 * per occupied cell (the point nearest the cell centre), so thumbnails never
 * pile up. `cells` is the 2-D resolution — how many digits the reader asked to
 * see; a third axis spends the same budget over a cube, so it gets fewer. */
export function binSubsample(
	z: Float32Array,
	n: number,
	d: number,
	center: number[],
	span: number,
	cells: number
): Int32Array {
	const G = d === 2 ? cells : Math.max(4, Math.round(cells * 0.38));
	const best = new Map<number, { i: number; dd: number }>();
	for (let i = 0; i < n; i++) {
		let key = 0;
		let dd = 0;
		for (let k = 0; k < d; k++) {
			const v = (z[i * d + k] - (center[k] ?? 0)) / span;
			let c = Math.floor(((v + 1) / 2) * G);
			c = c < 0 ? 0 : c >= G ? G - 1 : c;
			const centre = ((c + 0.5) / G) * 2 - 1;
			dd += (v - centre) * (v - centre);
			key = key * G + c;
		}
		const cur = best.get(key);
		if (!cur || dd < cur.dd) best.set(key, { i, dd });
	}
	const out = new Int32Array(best.size);
	let j = 0;
	for (const { i } of best.values()) out[j++] = i;
	return out;
}
