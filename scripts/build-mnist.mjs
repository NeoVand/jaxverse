// Pack MNIST into browser-friendly spritesheets. The PNG decoder is the one
// u8 decompressor every client already ships, so pixels go into two grayscale
// sheets (100 tiles per row) plus a flat label stream; the loader in
// src/lib/data/mnist.ts turns them back into Float32Array rows via
// createImageBitmap — no idx parsing on the client.
//
// Written assets (static/data/):
//   mnist-train.png   first 8000 train digits, 2800×2240 (100×80 tiles of 28²)
//   mnist-test.png    first 2000 test digits, 2800×560 (100×20 tiles)
//   mnist-labels.bin  Uint8, 8000 train labels then 2000 test labels
//   mnist-meta.json   {side, cols, train, test}
//
// Usage: node scripts/build-mnist.mjs [rawDir]
//   rawDir holds the gzipped idx files: train-images.gz train-labels.gz
//   test-images.gz test-labels.gz (from yann.lecun.com/exdb/mnist).

import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import sharp from 'sharp';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RAW = process.argv[2] ?? join(ROOT, 'scripts', '.cache', 'mnist');
const SIDE = 28;
const COLS = 100;
const N_TRAIN = 8000;
const N_TEST = 2000;

function readIdx(file, magic) {
	const buf = gunzipSync(readFileSync(join(RAW, file)));
	if (buf.readUInt32BE(0) !== magic)
		throw new Error(`${file}: magic ${buf.readUInt32BE(0)}, expected ${magic}`);
	if (magic === 2051 && (buf.readUInt32BE(8) !== SIDE || buf.readUInt32BE(12) !== SIDE))
		throw new Error(`${file}: not ${SIDE}×${SIDE}`);
	return { n: buf.readUInt32BE(4), data: buf.subarray(magic === 2051 ? 16 : 8) };
}

// Tile row-major: tile t sits at column t%COLS, row ⌊t/COLS⌋ of the sheet.
function packSheet(pixels, count) {
	const w = COLS * SIDE;
	const h = (count / COLS) * SIDE;
	const out = Buffer.alloc(w * h);
	for (let t = 0; t < count; t++) {
		const ox = (t % COLS) * SIDE;
		const oy = Math.floor(t / COLS) * SIDE;
		for (let y = 0; y < SIDE; y++)
			pixels.copy(out, (oy + y) * w + ox, (t * SIDE + y) * SIDE, (t * SIDE + y + 1) * SIDE);
	}
	return { data: out, w, h };
}

const trainImg = readIdx('train-images.gz', 2051);
const trainLab = readIdx('train-labels.gz', 2049);
const testImg = readIdx('test-images.gz', 2051);
const testLab = readIdx('test-labels.gz', 2049);
if (trainImg.n < N_TRAIN || testImg.n < N_TEST) throw new Error('idx files too small');

const outDir = join(ROOT, 'static', 'data');
mkdirSync(outDir, { recursive: true });

async function writeSheet(name, img, count) {
	const sheet = packSheet(img.data, count);
	await sharp(sheet.data, { raw: { width: sheet.w, height: sheet.h, channels: 1 } })
		.toColourspace('b-w') // keep the PNG single-channel; sharp otherwise upconverts to sRGB
		.png({ compressionLevel: 9, palette: false })
		.toFile(join(outDir, name));
	console.log(
		`${name}: ${sheet.w}×${sheet.h}, ${(statSync(join(outDir, name)).size / 1e6).toFixed(2)} MB`
	);
	return sheet;
}

const trainSheet = await writeSheet('mnist-train.png', trainImg, N_TRAIN);
await writeSheet('mnist-test.png', testImg, N_TEST);

writeFileSync(
	join(outDir, 'mnist-labels.bin'),
	Buffer.concat([trainLab.data.subarray(0, N_TRAIN), testLab.data.subarray(0, N_TEST)])
);
writeFileSync(
	join(outDir, 'mnist-meta.json'),
	JSON.stringify({ side: SIDE, cols: COLS, train: N_TRAIN, test: N_TEST })
);

// ── roundtrip check: decode the written PNG, compare tiles to the raw idx ──
const { data: dec, info } = await sharp(join(outDir, 'mnist-train.png'))
	.toColourspace('b-w')
	.raw()
	.toBuffer({ resolveWithObject: true });
if (info.width !== trainSheet.w || info.height !== trainSheet.h || info.channels !== 1)
	throw new Error(`decode shape ${info.width}×${info.height}×${info.channels}`);
for (const t of [0, 1, 137, 4321, N_TRAIN - 1]) {
	const ox = (t % COLS) * SIDE;
	const oy = Math.floor(t / COLS) * SIDE;
	for (let y = 0; y < SIDE; y++)
		for (let x = 0; x < SIDE; x++) {
			const got = dec[(oy + y) * info.width + ox + x];
			const want = trainImg.data[(t * SIDE + y) * SIDE + x];
			if (got !== want) throw new Error(`tile ${t} (${x},${y}): got ${got}, want ${want}`);
		}
}
console.log('roundtrip: 5 tiles verified against raw idx bytes');
console.log(
	`first 10 train labels: ${[...trainLab.data.subarray(0, 10)].join(',')} (expect 5,0,4,1,9,2,1,3,1,4)`
);
