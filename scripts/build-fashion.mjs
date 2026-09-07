// Pack Fashion-MNIST into browser-friendly spritesheets, the same way
// build-mnist.mjs packs digits: one grayscale PNG of tiles plus a flat label
// stream, so the client needs no idx parser and no second decompressor.
//
// Ten classes of 28×28 clothing instead of a thousand emoji across eight art
// styles. That is the whole point of the change: a generator has to hold the
// entire corpus in its weights, and this one is small enough to hold.
//
// Written assets (static/data/):
//   fashion-train.png   N_TRAIN garments, 2800×(N_TRAIN/100·28)
//   fashion-test.png    N_TEST garments
//   fashion-labels.bin  Uint8, train labels then test labels
//   fashion-meta.json   {side, cols, train, test, classes}
//
// Usage: node scripts/build-fashion.mjs [rawDir]
//   rawDir holds the gzipped idx files from
//   github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion

import { readFileSync, writeFileSync, mkdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import sharp from 'sharp';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RAW = process.argv[2] ?? join(ROOT, 'scripts', '.cache', 'fashion');
const SIDE = 28;
const COLS = 100;
const N_TRAIN = 12000;
const N_TEST = 2000;

/** Zalando's own ordering; the loader and the prose both depend on it. */
const CLASSES = [
	'T-shirt',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot'
];

// Zalando ships the long LeCun-style names; accept the short ones too so a
// hand-assembled raw directory works.
function pick(...names) {
	for (const n of names) if (existsSync(join(RAW, n))) return n;
	throw new Error(`none of ${names.join(', ')} found in ${RAW}`);
}

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

const trainImg = readIdx(pick('train-images-idx3-ubyte.gz', 'train-images.gz'), 2051);
const trainLab = readIdx(pick('train-labels-idx1-ubyte.gz', 'train-labels.gz'), 2049);
const testImg = readIdx(pick('t10k-images-idx3-ubyte.gz', 'test-images.gz'), 2051);
const testLab = readIdx(pick('t10k-labels-idx1-ubyte.gz', 'test-labels.gz'), 2049);
if (trainImg.n < N_TRAIN || testImg.n < N_TEST) throw new Error('idx files too small');
if (N_TRAIN % COLS || N_TEST % COLS) throw new Error('counts must be whole rows of tiles');

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

const trainSheet = await writeSheet('fashion-train.png', trainImg, N_TRAIN);
await writeSheet('fashion-test.png', testImg, N_TEST);

writeFileSync(
	join(outDir, 'fashion-labels.bin'),
	Buffer.concat([trainLab.data.subarray(0, N_TRAIN), testLab.data.subarray(0, N_TEST)])
);
writeFileSync(
	join(outDir, 'fashion-meta.json'),
	JSON.stringify({ side: SIDE, cols: COLS, train: N_TRAIN, test: N_TEST, classes: CLASSES })
);

// ── roundtrip check: decode the written PNG, compare tiles to the raw idx ──
const { data: dec, info } = await sharp(join(outDir, 'fashion-train.png'))
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

const counts = new Array(CLASSES.length).fill(0);
for (const l of trainLab.data.subarray(0, N_TRAIN)) counts[l]++;
console.log(`class balance: ${counts.map((c, i) => `${CLASSES[i]} ${c}`).join(', ')}`);
