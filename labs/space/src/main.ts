// Lab 2 — bending space until one straight cut is enough.
//
// A two-moons classifier, trained with softmax cross-entropy. Every few
// hundred steps the lab asks the network for its verdict over a whole grid
// and prints the decision boundary as ASCII — watch the crease form.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, numpy as np, nn, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const out = document.getElementById('out') as HTMLPreElement;
let head = '';
let map = '';
const render = () => (out.textContent = `${head}\n\n${map}`);

// ── two moons, generated on the spot ─────────────────────────────────────────
const N = 400;
const LAYERS = [2, 16, 16, 2];
const px = new Float32Array(N * 2);
const py = new Float32Array(N * 2); // one-hot targets
{
	let s = 42;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);
	for (let i = 0; i < N; i++) {
		const upper = i % 2 === 0;
		const t = rand() * Math.PI;
		const noise = () => (rand() - 0.5) * 0.22;
		px[i * 2] = (upper ? Math.cos(t) : 1 - Math.cos(t)) * 0.8 + noise();
		px[i * 2 + 1] = (upper ? Math.sin(t) - 0.25 : -Math.sin(t) + 0.25) * 0.8 + noise();
		py[i * 2 + (upper ? 0 : 1)] = 1;
	}
}

// ── the model: same recipe as lab 1, new head ────────────────────────────────
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
		if (k < p.w.length - 1) h = nn.gelu(h);
	}
	return h;
}

// cross-entropy: one-hot targets select the true class's log-probability
function lossFn(p: any, x: any, y: any) {
	const logp = nn.logSoftmax(forward(p, x), -1);
	return np.mean(np.sum(logp.mul(y), -1).neg());
}

// ── the ASCII boundary: a lattice through the same trained weights ──────────
const [GW, GH] = [64, 26];
const gridBuf = new Float32Array(GW * GH * 2);
for (let gy = 0; gy < GH; gy++)
	for (let gx = 0; gx < GW; gx++) {
		gridBuf[(gy * GW + gx) * 2] = (gx / (GW - 1)) * 3 - 1.5;
		gridBuf[(gy * GW + gx) * 2 + 1] = (1 - gy / (GH - 1)) * 2.4 - 1.2;
	}

function asciiMap(logits: Float32Array): string {
	// data points overprint the wash: o = class A, x = class B
	const cells: string[][] = [];
	for (let gy = 0; gy < GH; gy++) {
		cells.push([]);
		for (let gx = 0; gx < GW; gx++) {
			const i = gy * GW + gx;
			const margin = logits[i * 2] - logits[i * 2 + 1];
			cells[gy].push(Math.abs(margin) < 0.25 ? '·' : margin > 0 ? ' ' : '░');
		}
	}
	for (let i = 0; i < N; i++) {
		const gx = Math.round(((px[i * 2] + 1.5) / 3) * (GW - 1));
		const gy = Math.round((1 - (px[i * 2 + 1] + 1.2) / 2.4) * (GH - 1));
		if (gx >= 0 && gx < GW && gy >= 0 && gy < GH) cells[gy][gx] = py[i * 2] === 1 ? 'o' : 'x';
	}
	return cells.map((r) => r.join('')).join('\n');
}

async function main() {
	const devices = await init();
	const device = devices.includes('webgpu') ? 'webgpu' : devices.includes('wasm') ? 'wasm' : 'cpu';
	defaultDevice(device as any);

	let params: any = initParams(7);
	const solver = adam(3e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));
	const x = np.array(px).reshape([N, 2]);
	const y = np.array(py).reshape([N, 2]);
	const grid = np.array(gridBuf).reshape([GW * GH, 2]);

	const jitStep = jit((p: any, xx: any, yy: any) =>
		valueAndGrad((pp: any) => lossFn(pp, xx, yy))(p)
	);
	const jitGrid = jit((p: any, g: any) => forward(p, g));

	for (let step = 1; step <= 3000; step++) {
		const [lossVal, grads] = jitStep(tree.ref(params), x.ref, y.ref);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		if (step % 250 === 0 || step === 1) {
			const logits = jitGrid(tree.ref(params), grid.ref).dataSync() as Float32Array;
			head = `device ${device} · network ${LAYERS.join(' → ')} · step ${step} · loss ${lossVal
				.dataSync()[0]
				.toFixed(4)}`;
			map = asciiMap(logits);
			render();
			await new Promise((r) => setTimeout(r));
		} else {
			lossVal.dispose();
		}
	}
	head += ` · done — o and x are the data; ░ vs blank is the network's verdict`;
	render();
}

void main().catch((e) => {
	out.textContent = `error: ${e?.message ?? e}`;
});
