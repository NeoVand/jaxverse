// MNIST, decoded from the spritesheets that scripts/build-mnist.mjs packs
// into static/data/. The browser's PNG decoder does the decompression;
// here we only rasterize the sheets and slice tiles into training rows.

import { base } from '$app/paths';

export interface MnistData {
	/** train×784 row-major pixels, 0..1. */
	trainX: Float32Array;
	trainY: Int32Array;
	/** test×784 row-major pixels, 0..1. */
	testX: Float32Array;
	testY: Int32Array;
	/** Tile edge in pixels (28). */
	side: number;
}

interface MnistMeta {
	side: number;
	cols: number;
	train: number;
	test: number;
}

async function fetchOk(path: string): Promise<Response> {
	const res = await fetch(`${base}${path}`);
	if (!res.ok) throw new Error(`mnist: fetch ${path} failed (${res.status})`);
	return res;
}

// Rasterize a decoded PNG to RGBA bytes; OffscreenCanvas keeps this off the
// DOM, the <canvas> branch covers engines that lack it.
function rasterize(bitmap: ImageBitmap): Uint8ClampedArray {
	const { width, height } = bitmap;
	if (typeof OffscreenCanvas !== 'undefined') {
		const ctx = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
		if (!ctx) throw new Error('mnist: OffscreenCanvas 2d context unavailable');
		ctx.drawImage(bitmap, 0, 0);
		return ctx.getImageData(0, 0, width, height).data;
	}
	const el = document.createElement('canvas');
	el.width = width;
	el.height = height;
	const ctx = el.getContext('2d', { willReadFrequently: true });
	if (!ctx) throw new Error('mnist: canvas 2d context unavailable');
	ctx.drawImage(bitmap, 0, 0);
	return ctx.getImageData(0, 0, width, height).data;
}

// Tile t sits at column t%cols, row ⌊t/cols⌋ of the sheet; the PNG is
// grayscale so the red channel carries the pixel.
function sliceTiles(rgba: Uint8ClampedArray, meta: MnistMeta, count: number): Float32Array {
	const { side, cols } = meta;
	const sheetW = cols * side;
	const dim = side * side;
	const out = new Float32Array(count * dim);
	for (let t = 0; t < count; t++) {
		const ox = (t % cols) * side;
		const oy = Math.floor(t / cols) * side;
		for (let y = 0; y < side; y++) {
			const src = ((oy + y) * sheetW + ox) * 4;
			const dst = t * dim + y * side;
			for (let x = 0; x < side; x++) out[dst + x] = rgba[src + x * 4] / 255;
		}
	}
	return out;
}

async function loadSheet(path: string, meta: MnistMeta, count: number): Promise<Float32Array> {
	const blob = await (await fetchOk(path)).blob();
	const bitmap = await createImageBitmap(blob);
	const rows = sliceTiles(rasterize(bitmap), meta, count);
	bitmap.close();
	return rows;
}

let cache: Promise<MnistData> | null = null;

/** Fetch + decode once; every demo shares the same in-flight promise. */
export async function loadMnist(): Promise<MnistData> {
	cache ??= (async () => {
		const meta = (await (await fetchOk('/data/mnist-meta.json')).json()) as MnistMeta;
		const [trainX, testX, labelsBuf] = await Promise.all([
			loadSheet('/data/mnist-train.png', meta, meta.train),
			loadSheet('/data/mnist-test.png', meta, meta.test),
			fetchOk('/data/mnist-labels.bin').then((r) => r.arrayBuffer())
		]);
		const labels = new Uint8Array(labelsBuf);
		if (labels.length !== meta.train + meta.test)
			throw new Error(`mnist: expected ${meta.train + meta.test} labels, got ${labels.length}`);
		return {
			trainX,
			trainY: Int32Array.from(labels.subarray(0, meta.train)),
			testX,
			testY: Int32Array.from(labels.subarray(meta.train)),
			side: meta.side
		};
	})();
	return cache;
}
