// Lab 1 — one small MLP learns a sine wave, live.
//
// The whole jax-js training recipe in one file: parameters as a pytree,
// valueAndGrad over the tree, Adam from optax, and a jitted step. The canvas
// shows the dashed target and the network's current guess.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, numpy as np, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const out = document.getElementById('out') as HTMLPreElement;
const stage = document.getElementById('stage') as HTMLCanvasElement;
stage.hidden = false;
stage.width = 640;
stage.height = 240;
const ctx = stage.getContext('2d')!;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.slice(-14).join('\n');
};

// ── the data: 256 points of a sine, with a little noise ─────────────────────
const N = 256;
const LAYERS = [1, 32, 32, 1];
const target = (x: number) => Math.sin(3.1 * x);
const xs = new Float32Array(N);
const ys = new Float32Array(N);
for (let i = 0; i < N; i++) {
	xs[i] = (i / (N - 1)) * 2 - 1;
	ys[i] = target(xs[i]) + (Math.random() - 0.5) * 0.05;
}

// ── params: a plain object of arrays — a pytree ──────────────────────────────
function mulberry32(seed: number) {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function initParams(seed: number) {
	const rand = mulberry32(seed);
	const w: any[] = [];
	const b: any[] = [];
	for (let i = 0; i < LAYERS.length - 1; i++) {
		const [fin, fout] = [LAYERS[i], LAYERS[i + 1]];
		const limit = Math.sqrt(6 / (fin + fout)); // Glorot
		const buf = new Float32Array(fin * fout);
		for (let j = 0; j < buf.length; j++) buf[j] = (rand() * 2 - 1) * limit;
		w.push(np.array(buf).reshape([fin, fout]));
		b.push(np.zeros([fout]));
	}
	return { w, b };
}

// ── forward fold + MSE loss; the last layer stays linear ────────────────────
function forward(p: any, x: any) {
	let h = x;
	for (let k = 0; k < p.w.length; k++) {
		h = np.dot(h, p.w[k]).add(p.b[k]);
		if (k < p.w.length - 1) h = np.tanh(h);
	}
	return h;
}
const lossFn = (p: any, x: any, y: any) => np.mean(np.square(forward(p, x).sub(y)));

// ── drawing ──────────────────────────────────────────────────────────────────
/** Read a token off the page, so the canvas follows light and dark like the book. */
const token = (name: string, fallback: string) =>
	getComputedStyle(stage).getPropertyValue(name).trim() || fallback;

function draw(pred: Float32Array) {
	const { width: W, height: Hh } = stage;
	ctx.clearRect(0, 0, W, Hh);
	const px = (x: number) => ((x + 1) / 2) * W;
	const py = (y: number) => Hh / 2 - y * (Hh / 2.6);
	ctx.setLineDash([4, 4]);
	ctx.strokeStyle = token('--ink-3', '#a3a094');
	ctx.beginPath();
	for (let i = 0; i < N; i++) {
		const [X, Y] = [px(xs[i]), py(target(xs[i]))];
		if (i === 0) ctx.moveTo(X, Y);
		else ctx.lineTo(X, Y);
	}
	ctx.stroke();
	ctx.setLineDash([]);
	ctx.strokeStyle = token('--accent', '#2b45d8');
	ctx.lineWidth = 2;
	ctx.beginPath();
	for (let i = 0; i < N; i++) {
		const [X, Y] = [px(xs[i]), py(pred[i])];
		if (i === 0) ctx.moveTo(X, Y);
		else ctx.lineTo(X, Y);
	}
	ctx.stroke();
	ctx.lineWidth = 1;
}

// ── the loop ─────────────────────────────────────────────────────────────────
async function main() {
	const devices = await init();
	const device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as any);
	log(`device: ${device} · network ${LAYERS.join(' → ')}`);

	let params: any = initParams(7);
	const solver = adam(3e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));

	const x = np.array(xs).reshape([N, 1]);
	const y = np.array(ys).reshape([N, 1]);

	// jit: the first call traces + compiles; every later call is one dispatch
	const jitStep = jit((p: any, xx: any, yy: any) =>
		valueAndGrad((pp: any) => lossFn(pp, xx, yy))(p)
	);
	const jitPredict = jit((p: any, xx: any) => forward(p, xx));

	for (let step = 1; step <= 4000; step++) {
		const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		if (step % 100 === 0) {
			const pred = jitPredict(tree.ref(params), x.ref).dataSync() as Float32Array;
			draw(pred);
			log(`step ${String(step).padStart(4)}  loss ${lossVal.dataSync()[0].toFixed(5)}`);
			await new Promise((r) => setTimeout(r));
		} else {
			lossVal.dispose();
		}
	}
	log('done — change LAYERS or the target function and save to retrain.');
}

void main().catch((e) => log(`error: ${e?.message ?? e}`));
