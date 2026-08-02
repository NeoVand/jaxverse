// Lab 4 — a variational autoencoder squeezes MNIST through two numbers.
//
// The hourglass: 784 → 256 → 64 → (waist) → 64 → 256 → 784, with the waist
// variational — the encoder proposes a Gaussian, the decoder gets a sample,
// and z = μ + σ·ε keeps the whole thing differentiable. Every few hundred
// steps the lab encodes 300 test digits and prints the latent map as ASCII:
// each character is a real digit, printed at its own address. Watch the
// digits sort themselves — no label ever enters the loss.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, numpy as np, nn, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const out = document.getElementById('out') as HTMLPreElement;
let head = '';
let map = '';
const render = () => (out.textContent = `${head}\n\n${map}`);

const DIM = 784;
const K = 2; // the waist
const ENC = [DIM, 256, 64];
const DEC = [K, 64, 256, DIM];
const BATCH = 128;
const BETA = 1 / DIM;

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
	const outBuf = new Float32Array(count * DIM);
	for (let t = 0; t < count; t++) {
		const ox = (t % meta.cols) * meta.side;
		const oy = Math.floor(t / meta.cols) * meta.side;
		for (let y = 0; y < meta.side; y++) {
			const src = ((oy + y) * meta.cols * meta.side + ox) * 4;
			const dst = t * DIM + y * meta.side;
			for (let x = 0; x < meta.side; x++) outBuf[dst + x] = rgba[src + x * 4] / 255;
		}
	}
	return outBuf;
}

function initLayers(sizes: number[], rand: () => number) {
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < sizes.length - 1; i++) {
		const fout = sizes[i + 1];
		const fin = sizes[i];
		const limit = Math.sqrt(6 / (fin + fout));
		const buf = new Float32Array(fin * fout);
		for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}

function half(p: any, x: any, bendLast: boolean) {
	let h = x;
	for (let k = 0; k < p.w.length; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < p.w.length - 1 || bendLast) h = nn.gelu(h);
	}
	return h;
}

/** encoder → waist stats. The extra matrix maps 64 → [μ | log σ²]. */
function encodeStats(p: any, x: any) {
	const h = half(p.enc, x, true);
	return np.dot(h, p.waist.w).add(p.waist.b); // [B, 2K]
}

function lossFn(p: any, x: any, eps: any) {
	const stats = encodeStats(p, x.ref);
	const [mu, lv] = np.split(stats, 2, 1);
	// the reparameterization trick: z = μ + σ·ε, gradient flows through μ, σ
	const z = mu.ref.add(np.exp(lv.ref.mul(0.5)).mul(eps));
	// KL(N(μ,σ²) ‖ N(0,1)) — the rent that keeps the map centred and smooth
	const kl = np.mean(np.sum(np.square(mu).add(np.exp(lv.ref)).sub(1).sub(lv), 1)).mul(0.5);
	const rebuilt = half(p.dec, z, false);
	return np.mean(np.square(rebuilt.sub(x))).add(kl.mul(BETA));
}

async function main() {
	const devices = await init();
	const device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as any);
	head = `device ${device} · ${ENC.join('→')} → waist ${K} → ${DEC.join('→')} · decoding sheets…`;
	render();

	const meta = (await (await fetch('/data/mnist-meta.json')).json()) as Meta;
	const [trainX, testX, labelsBuf] = await Promise.all([
		loadSheet('/data/mnist-train.png', meta, meta.train),
		loadSheet('/data/mnist-test.png', meta, meta.test),
		fetch('/data/mnist-labels.bin').then((r) => r.arrayBuffer())
	]);
	const labels = new Uint8Array(labelsBuf);

	let s = 7;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	let params: any = {
		enc: initLayers(ENC, rand),
		waist: (() => {
			const limit = Math.sqrt(6 / (64 + 2 * K));
			const buf = new Float32Array(64 * 2 * K);
			for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
			return { w: np.array(buf).reshape([64, 2 * K]), b: np.zeros([2 * K]) };
		})(),
		dec: initLayers(DEC, rand)
	};
	const solver = adam(2e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));
	const jitStep = jit((p: any, x: any, e: any) => valueAndGrad((pp: any) => lossFn(pp, x, e))(p));
	const jitEncode = jit((p: any, x: any) => encodeStats(p, x));

	// 300 test digits for the ASCII map
	const MAPN = 300;
	const mapX = np.array(testX.slice(0, MAPN * DIM)).reshape([MAPN, DIM]);

	function asciiMap(): string {
		const stats = jitEncode(tree.ref(params), mapX.ref).dataSync() as Float32Array;
		const [GW, GH] = [64, 24];
		const grid: string[][] = Array.from({ length: GH }, () => Array(GW).fill(' '));
		// frame the cloud: middle 95% of each axis
		const xsAll = Array.from({ length: MAPN }, (_, i) => stats[i * 2 * K]);
		const ysAll = Array.from({ length: MAPN }, (_, i) => stats[i * 2 * K + 1]);
		const lo = (a: number[]) => a.slice().sort((p, q) => p - q)[Math.floor(a.length * 0.025)];
		const hi = (a: number[]) => a.slice().sort((p, q) => p - q)[Math.floor(a.length * 0.975)];
		const [x0, x1, y0, y1] = [lo(xsAll), hi(xsAll), lo(ysAll), hi(ysAll)];
		for (let i = 0; i < MAPN; i++) {
			const gx = Math.round(((xsAll[i] - x0) / (x1 - x0 || 1)) * (GW - 1));
			const gy = Math.round((1 - (ysAll[i] - y0) / (y1 - y0 || 1)) * (GH - 1));
			if (gx >= 0 && gx < GW && gy >= 0 && gy < GH) grid[gy][gx] = String(labels[meta.train + i]);
		}
		return grid.map((r) => r.join('')).join('\n');
	}

	const xBuf = new Float32Array(BATCH * DIM);
	const eBuf = new Float32Array(BATCH * K);
	for (let step = 1; step <= 3000; step++) {
		for (let b = 0; b < BATCH; b++) {
			const row = Math.floor(rand() * meta.train);
			xBuf.set(trainX.subarray(row * DIM, (row + 1) * DIM), b * DIM);
		}
		// ε drawn OUTSIDE the traced function — that is the whole trick
		for (let i = 0; i < eBuf.length; i++)
			eBuf[i] = Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
		const x = np.array(xBuf).reshape([BATCH, DIM]);
		const eps = np.array(eBuf).reshape([BATCH, K]);
		const [lossVal, grads] = jitStep(tree.ref(params), x, eps);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		if (step % 300 === 0 || step === 1) {
			head = `step ${String(step).padStart(4)} · loss ${lossVal.dataSync()[0].toFixed(4)} · each character below is a real test digit at its latent address`;
			map = asciiMap();
			render();
		} else {
			lossVal.dispose();
		}
		if (step % 25 === 0) await new Promise((r) => setTimeout(r)); // keep the page alive
	}
	head += ' · done — the clusters are unsupervised: no label ever entered the loss';
	render();
}

void main().catch((e) => {
	out.textContent = `error: ${e?.message ?? e}`;
});
