// Spike: sizing the emoji denoiser.
//
// A conv U-Net and a patch transformer were timed head to head at 32x32 RGBA
// (U-Net 1.58M params / 264 ms per step; DiT 2.49M / 166 ms). The transformer
// wins because every one of its matmuls happens after patchify, on 64 tokens,
// which is jax-js's fast path — the U-Net spends its whole budget on 3x3 convs
// at full resolution. So the remaining question is only which DiT: one small
// enough that a reader watches it learn, and one large enough to ship.

import {
	init,
	defaultDevice,
	numpy as np,
	nn,
	lax,
	random,
	jit,
	valueAndGrad,
	tree
} from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';

const out = document.getElementById('out') as HTMLPreElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.join('\n');
};

const RES = 32;
const CH = 4;
const COND = 128;

// jax-js arrays are consumed on use and typed loosely at this seam.
/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

interface Cfg {
	name: string;
	d: number;
	layers: number;
	heads: number;
	patch: number;
	batch: number;
}

function normal(key: Any, shape: number[], scale: number) {
	return random.normal(key, shape).mul(scale);
}

function initDit(c: Cfg, seed: number) {
	const n = 12 + c.layers * 8;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nk = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const { d, patch, layers } = c;
	const tok = (RES / patch) * (RES / patch);
	const s = Math.sqrt(3 / d);
	const u = (shape: number[], k: number) =>
		random.uniform(nk(), shape, { minval: -k * s, maxval: k * s });
	const p: Any = {
		patchify: normal(nk(), [d, CH, patch, patch], 0.05),
		pos: normal(nk(), [tok, d], 0.02),
		condW: normal(nk(), [COND, d], 0.05),
		condW2: normal(nk(), [d, d], 0.05),
		blocks: [],
		headFilm: normal(nk(), [d, d * 2], 0.02),
		unpatch: normal(nk(), [CH, d, patch, patch], 0.01)
	};
	for (let i = 0; i < layers; i++) {
		p.blocks.push({
			film: normal(nk(), [d, d * 4], 0.02),
			wq: u([d, d], 1),
			wk: u([d, d], 1),
			wv: u([d, d], 1),
			wo: u([d, d], 0.2),
			fc1: u([d, 4 * d], 0.4),
			fc2: u([4 * d, d], 0.2)
		});
	}
	return p;
}

function rmsnorm(x: Any) {
	const ms = np.mean(np.square(x.ref), -1, { keepdims: true });
	return x.div(np.sqrt(ms.add(1e-5)));
}

function ditForward(p: Any, c: Cfg, x: Any, cond: Any) {
	const { d, heads, patch, batch: B } = c;
	const tok = (RES / patch) * (RES / patch);
	const headDim = d / heads;

	const mod = (v: Any, sb: Any, off: number) => {
		const scale = sb.ref.slice([], [off, off + d]).reshape([B, 1, d]);
		const shift = sb.slice([], [off + d, off + 2 * d]).reshape([B, 1, d]);
		return v
			.reshape([B, tok, d])
			.mul(scale.add(1))
			.add(shift)
			.reshape([B * tok, d]);
	};

	const cvec = nn.silu(np.dot(nn.silu(np.dot(cond, p.condW)), p.condW2));

	let h = lax.conv(x, p.patchify, [patch, patch], 'VALID');
	h = np.transpose(h.reshape([B, d, tok]), [0, 2, 1]).reshape([B * tok, d]);
	h = h.add(np.tile(p.pos, [B, 1]));

	for (const layer of p.blocks) {
		const sb = np.dot(cvec.ref, layer.film);
		const res = h.ref;
		const a = mod(rmsnorm(h), sb.ref, 0);
		const q = np.dot(a.ref, layer.wq).reshape([B, tok, heads, headDim]);
		const k = np.dot(a.ref, layer.wk).reshape([B, tok, heads, headDim]);
		const v = np.dot(a, layer.wv).reshape([B, tok, heads, headDim]);
		const o = nn.dotProductAttention(q, k, v);
		h = np.dot(o.reshape([B * tok, d]), layer.wo).add(res);

		const res2 = h.ref;
		let f = mod(rmsnorm(h), sb, 2 * d);
		f = nn.gelu(np.dot(f, layer.fc1));
		h = np.dot(f, layer.fc2).add(res2);
	}

	h = mod(rmsnorm(h), np.dot(cvec, p.headFilm), 0);
	h = np.transpose(h.reshape([B, tok, d]), [0, 2, 1]).reshape([B, d, RES / patch, RES / patch]);
	return lax.convTranspose(h, p.unpatch, [patch, patch], 'VALID');
}

function paramCount(p: Any): number {
	const leaves = tree.leaves(tree.ref(p)) as Any[];
	let total = 0;
	for (const l of leaves) {
		total += l.size;
		l.dispose();
	}
	return total;
}

/**
 * `fused` folds the Adam update inside the jit boundary. Outside it, optax
 * dispatches four small kernels per leaf, and with ~40 leaves that launch
 * traffic — not the matmuls — is what the step time is made of.
 * `syncEvery` > 1 reads the loss back on only some steps.
 */
async function bench(c: Cfg, fused: boolean, syncEvery = 1) {
	let params = initDit(c, 7);
	const B = c.batch;
	const pc = paramCount(params);

	const loss = (p: Any, x: Any, cond: Any, target: Any) =>
		np.mean(np.square(ditForward(p, c, x, cond).sub(target)));

	const solver = adam(3e-4, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));

	const gradStep = jit((p: Any, x: Any, cond: Any, t: Any) =>
		valueAndGrad((pp: Any) => loss(pp, x, cond, t))(p)
	);
	// Hand-rolled Adam so the update can live inside jit. optax's version reads
	// its step counter with .item(), which a tracer cannot answer; the two bias
	// corrections come in instead as traced scalars, so the shape signature —
	// and therefore the compiled kernel — is the same on every step.
	const fusedStep = jit((p: Any, m: Any, v: Any, c: Any, x: Any, cond: Any, t: Any) => {
		const [lv, grads] = valueAndGrad((pp: Any) => loss(pp, x, cond, t))(tree.ref(p));
		const [pl, def] = tree.flatten(p) as [Any[], Any];
		const gl = tree.leaves(grads) as Any[];
		const ml = tree.leaves(m) as Any[];
		const vl = tree.leaves(v) as Any[];
		const c1 = c.ref.slice([0, 1]);
		const c2 = c.slice([1, 2]);
		const nm: Any[] = [];
		const nv: Any[] = [];
		const nextP: Any[] = [];
		for (let i = 0; i < gl.length; i++) {
			const mi = ml[i].mul(0.9).add(gl[i].ref.mul(0.1));
			const vi = vl[i].mul(0.99).add(np.square(gl[i]).mul(0.01));
			const mhat = mi.ref.mul(c1.ref);
			const vhat = vi.ref.mul(c2.ref);
			nextP.push(pl[i].sub(mhat.mul(3e-4).div(np.sqrt(vhat).add(1e-8))));
			nm.push(mi);
			nv.push(vi);
		}
		c1.dispose();
		c2.dispose();
		return [lv, tree.unflatten(def, nextP), tree.unflatten(def, nm), tree.unflatten(def, nv)];
	});
	let mState: Any = tree.map((l: Any) => np.zerosLike(l), tree.ref(params));
	let vState: Any = tree.map((l: Any) => np.zerosLike(l), tree.ref(params));

	const xBuf = new Float32Array(B * CH * RES * RES);
	const cBuf = new Float32Array(B * COND);
	for (let i = 0; i < xBuf.length; i++) xBuf[i] = Math.random() * 2 - 1;
	for (let i = 0; i < cBuf.length; i++) cBuf[i] = Math.random();

	const times: number[] = [];
	let compile = 0;
	const N = 12;
	for (let i = 0; i < N; i++) {
		const x = np.array(xBuf).reshape([B, CH, RES, RES]);
		const cond = np.array(cBuf).reshape([B, COND]);
		const t = np.array(xBuf).reshape([B, CH, RES, RES]);
		const t0 = performance.now();
		let lv: Any;
		if (fused) {
			const k = i + 1;
			const corr = np.array(
				new Float32Array([1 / (1 - Math.pow(0.9, k)), 1 / (1 - Math.pow(0.99, k))])
			);
			const [l, p2, m2, v2] = fusedStep(params, mState, vState, corr, x, cond, t);
			lv = l;
			params = p2;
			mState = m2;
			vState = v2;
		} else {
			const [l, grads] = gradStep(tree.ref(params), x, cond, t);
			lv = l;
			const [updates, newOpt] = solver.update(grads, optState, tree.ref(params));
			params = applyUpdates(params, updates);
			optState = newOpt;
		}
		if (i % syncEvery === 0) lv.item();
		else lv.dispose();
		const dt = performance.now() - t0;
		if (i === 0) compile = dt;
		else times.push(dt);
	}
	// with syncEvery > 1 the queue has to be drained before the clock is fair
	np.mean(np.array(xBuf)).item();
	const steady = times.slice(2);
	const mean = steady.reduce((a, b) => a + b, 0) / steady.length;
	const imgPerSec = Math.round((B * 1000) / mean);
	log(
		`${c.name.padEnd(22)} ${fused ? 'fused' : 'split'} sync${syncEvery}  ` +
			`${(pc / 1e6).toFixed(2)}M  ` +
			`${mean.toFixed(0).padStart(4)} ms/step  ` +
			`${String(imgPerSec).padStart(5)} img/s  ` +
			`compile ${compile.toFixed(0)}ms`
	);

	tree.dispose(params);
	tree.dispose(optState);
	tree.dispose(mState);
	tree.dispose(vState);
}

const MID: Cfg = { name: 'd192 L4 p4', d: 192, layers: 4, heads: 4, patch: 4, batch: 32 };
const RUNS: [Cfg, boolean, number][] = [
	[MID, false, 1],
	[MID, true, 1],
	[MID, true, 4],
	[{ ...MID, batch: 64 }, true, 1],
	[{ ...MID, batch: 128 }, true, 1],
	[{ ...MID, batch: 256 }, true, 1],
	[{ ...MID, name: 'd256 L6 p4', d: 256, layers: 6, heads: 8, batch: 128 }, true, 1],
	[{ ...MID, name: 'd320 L8 p4', d: 320, layers: 8, heads: 8, batch: 128 }, true, 1],
	[{ ...MID, name: 'd192 L4 sample', batch: 8 }, true, 1]
];

async function main() {
	const devices = await init();
	log(`devices: ${devices.join(', ')}`);
	if (!devices.includes('webgpu')) {
		log('error: no WebGPU');
		return;
	}
	defaultDevice('webgpu');
	log('');
	for (const [c, fused, sync] of RUNS) await bench(c, fused, sync);
	log('\nstep 999 done');
}

void main().catch((e) => log(`error: ${e?.stack ?? e}`));
