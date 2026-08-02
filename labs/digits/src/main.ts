// Lab 3 — a real MNIST classifier, honestly graded.
//
// The dataset ships in this zip as two PNG spritesheets (the browser's image
// decoder is the decompressor) plus a small label file. Training is minibatch
// softmax cross-entropy; the score that matters is accuracy on 2,000 test
// digits the gradients never see.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, numpy as np, nn, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const out = document.getElementById('out') as HTMLPreElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.join('\n');
};

const DIM = 784;
const CLASSES = 10;
const LAYERS = [DIM, 128, 128, CLASSES];
const BATCH = 128;

// ── decode the spritesheets ──────────────────────────────────────────────────
interface Meta {
	side: number;
	cols: number;
	train: number;
	test: number;
}

async function loadSheet(path: string, meta: Meta, count: number): Promise<Float32Array> {
	const blob = await (await fetch(path)).blob();
	const bmp = await createImageBitmap(blob);
	const cv = new OffscreenCanvas(bmp.width, bmp.height);
	const cx = cv.getContext('2d', { willReadFrequently: true })!;
	cx.drawImage(bmp, 0, 0);
	const rgba = cx.getImageData(0, 0, bmp.width, bmp.height).data;
	bmp.close();
	const { side, cols } = meta;
	const outBuf = new Float32Array(count * side * side);
	for (let t = 0; t < count; t++) {
		const ox = (t % cols) * side;
		const oy = Math.floor(t / cols) * side;
		for (let y = 0; y < side; y++) {
			const src = ((oy + y) * cols * side + ox) * 4;
			const dst = t * side * side + y * side;
			for (let x = 0; x < side; x++) outBuf[dst + x] = rgba[src + x * 4] / 255;
		}
	}
	return outBuf;
}

// ── the model (the same recipe as labs 1 and 2, scaled up) ───────────────────
function initParams(seed: number) {
	let s = seed >>> 0;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < LAYERS.length - 1; i++) {
		const [fin, fout] = [LAYERS[i], LAYERS[i + 1]];
		const limit = Math.sqrt(6 / (fin + fout));
		const buf = new Float32Array(fin * fout);
		for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}

function forward(p: any, x: any) {
	let h = x;
	for (let k = 0; k < p.w.length; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < p.w.length - 1) h = nn.relu(h);
	}
	return h;
}

function lossFn(p: any, x: any, y: any) {
	const logp = nn.logSoftmax(forward(p, x), -1);
	return np.mean(np.sum(logp.mul(y), -1).neg()); // −log p_true, averaged
}

async function main() {
	const devices = await init();
	const device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as any);
	log(`device: ${device} · network ${LAYERS.join(' → ')}`);
	log('decoding spritesheets…');

	const meta = (await (await fetch('/data/mnist-meta.json')).json()) as Meta;
	const [trainX, testX, labelsBuf] = await Promise.all([
		loadSheet('/data/mnist-train.png', meta, meta.train),
		loadSheet('/data/mnist-test.png', meta, meta.test),
		fetch('/data/mnist-labels.bin').then((r) => r.arrayBuffer())
	]);
	const labels = new Uint8Array(labelsBuf);
	log(`train ${meta.train} rows · test ${meta.test} rows, locked away from the gradients`);

	let params: any = initParams(7);
	const solver = adam(3e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));
	const jitStep = jit((p: any, xx: any, yy: any) =>
		valueAndGrad((pp: any) => lossFn(pp, xx, yy))(p)
	);
	const jitPredict = jit((p: any, xx: any) => forward(p, xx));

	// test rows go up once; accuracy reads them back in chunks of 500
	const TEST = meta.test;
	const testChunks: any[] = [];
	for (let o = 0; o < TEST; o += 500)
		testChunks.push(np.array(testX.slice(o * DIM, (o + 500) * DIM)).reshape([500, DIM]));

	function testAccuracy(): number {
		let hit = 0;
		for (let ci = 0; ci < testChunks.length; ci++) {
			const logits = jitPredict(tree.ref(params), testChunks[ci].ref).dataSync() as Float32Array;
			for (let i = 0; i < 500; i++) {
				let best = 0;
				for (let k = 1; k < CLASSES; k++)
					if (logits[i * CLASSES + k] > logits[i * CLASSES + best]) best = k;
				if (best === labels[meta.train + ci * 500 + i]) hit++;
			}
		}
		return hit / TEST;
	}

	// ── minibatch loop: 128 random training rows per step ──────────────────────
	let s = 99;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	const xBuf = new Float32Array(BATCH * DIM);
	const yBuf = new Float32Array(BATCH * CLASSES);
	const t0 = performance.now();
	for (let step = 1; step <= 1500; step++) {
		yBuf.fill(0);
		for (let b = 0; b < BATCH; b++) {
			const row = Math.floor(rand() * meta.train);
			xBuf.set(trainX.subarray(row * DIM, (row + 1) * DIM), b * DIM);
			yBuf[b * CLASSES + labels[row]] = 1;
		}
		const x = np.array(xBuf).reshape([BATCH, DIM]);
		const y = np.array(yBuf).reshape([BATCH, CLASSES]);
		const [lossVal, grads] = jitStep(tree.ref(params), x, y);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		if (step % 150 === 0 || step === 1) {
			const acc = testAccuracy();
			const ms = (performance.now() - t0) / step;
			log(
				`step ${String(step).padStart(4)}  loss ${lossVal.dataSync()[0].toFixed(3)}  ` +
					`test acc ${(acc * 100).toFixed(1)}%  ·  ${ms.toFixed(1)} ms/step`
			);
			await new Promise((r) => setTimeout(r));
		} else {
			lossVal.dispose();
		}
	}
	log('done — untrained was 10%; a linear model plateaus near 92%; this MLP should clear 95%.');
}

void main().catch((e) => log(`error: ${e?.message ?? e}`));
