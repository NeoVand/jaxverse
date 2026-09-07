// Load the Fashion-MNIST spritesheet the build script wrote and hand back the
// bytes the trainer and the plates both want.
//
// The pictures arrive as one grayscale PNG of 28-pixel tiles because a PNG is
// the only decompressor a browser already ships. Decoding is done once and
// cached: several plates on the same page share a single download.

import type { Corpus } from './runtime';

export interface FashionMeta {
	side: number;
	cols: number;
	train: number;
	test: number;
	/** Zalando's ordering; the label byte indexes straight into this. */
	classes: string[];
}

export interface FashionCorpus extends Corpus {
	meta: FashionMeta;
	/** Held-out pictures, same layout, for the memorization check. */
	test: Uint8Array;
	testLabels: Uint8Array;
}

async function rasterize(url: string, w: number, h: number): Promise<Uint8ClampedArray> {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${res.status} loading ${url}`);
	const bitmap = await createImageBitmap(await res.blob());
	const canvas =
		typeof OffscreenCanvas !== 'undefined'
			? new OffscreenCanvas(w, h)
			: Object.assign(document.createElement('canvas'), { width: w, height: h });
	const ctx = canvas.getContext('2d', {
		willReadFrequently: true
	}) as CanvasRenderingContext2D | null;
	if (!ctx) throw new Error('no 2d context for the garment sheets');
	ctx.clearRect(0, 0, w, h);
	ctx.drawImage(bitmap, 0, 0);
	bitmap.close();
	return ctx.getImageData(0, 0, w, h).data;
}

/** Cut a sheet of tiles into one grayscale plane per picture. */
function untile(px: Uint8ClampedArray, cols: number, side: number, count: number): Uint8Array {
	const plane = side * side;
	const out = new Uint8Array(count * plane);
	const width = cols * side;
	for (let i = 0; i < count; i++) {
		const tx = (i % cols) * side;
		const ty = Math.floor(i / cols) * side;
		for (let y = 0; y < side; y++) {
			for (let x = 0; x < side; x++) {
				// the sheet is written single-channel, so red carries the ink
				out[i * plane + y * side + x] = px[((ty + y) * width + tx + x) * 4];
			}
		}
	}
	return out;
}

let pending: Promise<FashionCorpus> | null = null;

/** Fetch and decode both sheets. Cached — several plates share one download. */
export function loadFashion(base = ''): Promise<FashionCorpus> {
	pending ??= (async () => {
		const meta: FashionMeta = await fetch(`${base}/data/fashion-meta.json`).then((r) => r.json());
		const { side, cols, train, test } = meta;

		const [trainPx, testPx, labelBuf] = await Promise.all([
			rasterize(`${base}/data/fashion-train.png`, cols * side, (train / cols) * side),
			rasterize(`${base}/data/fashion-test.png`, cols * side, (test / cols) * side),
			fetch(`${base}/data/fashion-labels.bin`).then((r) => r.arrayBuffer())
		]);

		const labels = new Uint8Array(labelBuf);
		return {
			meta,
			images: untile(trainPx, cols, side, train),
			labels: labels.slice(0, train),
			count: train,
			test: untile(testPx, cols, side, test),
			testLabels: labels.slice(train, train + test)
		};
	})();
	return pending;
}
