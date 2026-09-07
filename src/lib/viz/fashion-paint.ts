// Drawing model output onto the page.
//
// Fashion-MNIST is one grayscale channel: bright where the garment is, black
// where the backdrop is. Painting that literally would put a black square on
// every plate and invert itself between the book's two themes, so the value is
// read as ink coverage instead — nothing at -1, full ink at +1 — and composited
// over whatever colour the page happens to be. A sneaker is then dark on the
// day theme and pale on the night one, with no second copy of the data and no
// branch in the loop.

export interface PaintOptions {
	/** Tiles across. Rows follow from the count. */
	columns: number;
	/** Page colour to composite onto, as `#rrggbb`. */
	background: string;
	/** Colour of the garment itself, as `#rrggbb`. */
	ink: string;
	/** Blank space between tiles, in source pixels. */
	gap?: number;
	/** Draw only this many tiles, for progressive reveals. */
	limit?: number;
}

function parseHex(hex: string): [number, number, number] {
	const h = hex.trim().replace('#', '');
	if (h.length === 3) {
		return [parseInt(h[0] + h[0], 16), parseInt(h[1] + h[1], 16), parseInt(h[2] + h[2], 16)];
	}
	if (h.length >= 6) {
		return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
	}
	return [128, 128, 128];
}

/** Model output in [-1, 1] to ink coverage in [0, 1]. */
const coverage = (v: number) => Math.max(0, Math.min(1, (v + 1) / 2));

/**
 * Compose `count` tiles into one ImageData laid out in a grid.
 *
 * `pixels` is one plane per tile, which for a single channel means the tiles
 * simply follow one another.
 */
export function tilesToImageData(
	pixels: Float32Array,
	count: number,
	res: number,
	opts: PaintOptions
): ImageData {
	const { columns, gap = 0 } = opts;
	const [br, bg, bb] = parseHex(opts.background);
	const [ir, ig, ib] = parseHex(opts.ink);
	const rows = Math.ceil(count / columns);
	const w = columns * res + (columns - 1) * gap;
	const h = rows * res + (rows - 1) * gap;
	const img = new ImageData(w, h);
	const plane = res * res;

	// start as flat background so the gaps are not black
	for (let i = 0; i < w * h; i++) {
		img.data[i * 4] = br;
		img.data[i * 4 + 1] = bg;
		img.data[i * 4 + 2] = bb;
		img.data[i * 4 + 3] = 255;
	}

	const limit = Math.min(count, opts.limit ?? count);
	for (let k = 0; k < limit; k++) {
		const ox = (k % columns) * (res + gap);
		const oy = Math.floor(k / columns) * (res + gap);
		const base = k * plane;
		for (let y = 0; y < res; y++) {
			for (let x = 0; x < res; x++) {
				const a = coverage(pixels[base + y * res + x]);
				const d = ((oy + y) * w + ox + x) * 4;
				img.data[d] = Math.round(ir * a + br * (1 - a));
				img.data[d + 1] = Math.round(ig * a + bg * (1 - a));
				img.data[d + 2] = Math.round(ib * a + bb * (1 - a));
			}
		}
	}
	return img;
}

/**
 * Blit an ImageData to a canvas at whatever size CSS gave it, without
 * smoothing — these are 28-pixel pictures and they should look like it.
 */
export function blitCrisp(canvas: HTMLCanvasElement, img: ImageData): void {
	const dpr = Math.min(typeof devicePixelRatio === 'number' ? devicePixelRatio : 1, 2);
	const cssW = canvas.clientWidth || img.width;
	const cssH = canvas.clientHeight || img.height;
	const w = Math.round(cssW * dpr);
	const h = Math.round(cssH * dpr);
	if (canvas.width !== w || canvas.height !== h) {
		canvas.width = w;
		canvas.height = h;
	}
	const ctx = canvas.getContext('2d');
	if (!ctx) return;
	const scratch = document.createElement('canvas');
	scratch.width = img.width;
	scratch.height = img.height;
	scratch.getContext('2d')!.putImageData(img, 0, 0);
	ctx.setTransform(1, 0, 0, 1, 0, 0);
	ctx.clearRect(0, 0, w, h);
	ctx.imageSmoothingEnabled = false;
	// preserve the grid's aspect ratio inside whatever box CSS handed us
	const scale = Math.min(w / img.width, h / img.height);
	const dw = img.width * scale;
	const dh = img.height * scale;
	ctx.drawImage(scratch, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

/** One-shot convenience for the many plates that just want a row of tiles. */
export function paintTiles(
	canvas: HTMLCanvasElement,
	pixels: Float32Array,
	count: number,
	res: number,
	opts: PaintOptions
): void {
	blitCrisp(canvas, tilesToImageData(pixels, count, res, opts));
}

/**
 * Add Gaussian noise to a tile the way the forward process does, for the
 * plates that destroy a picture without going anywhere near the GPU.
 */
export function corrupt(
	src: Float32Array,
	out: Float32Array,
	signal: number,
	rand: () => number
): void {
	const noise = Math.sqrt(Math.max(0, 1 - signal * signal));
	for (let i = 0; i < src.length; i += 2) {
		const u = Math.max(rand(), 1e-7);
		const r = Math.sqrt(-2 * Math.log(u));
		const th = 2 * Math.PI * rand();
		out[i] = signal * src[i] + noise * r * Math.cos(th);
		if (i + 1 < src.length) out[i + 1] = signal * src[i + 1] + noise * r * Math.sin(th);
	}
}
