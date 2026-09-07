// Lab 9 — out of the static.
//
// A denoising diffusion model on 28x28 Fashion-MNIST, in one file. Corrupt a
// garment by a random amount, ask a small transformer which noise was added,
// and score it on squared error. Then walk a fresh sheet of static back into
// a picture by asking that question fifty times.
//
// Everything that matters is a constant at the top or a loop below. Change a
// number and watch what happens.

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

const out = document.getElementById('out') as HTMLPreElement;
const stage = document.getElementById('stage') as HTMLCanvasElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	if (lines.length > 18) lines.shift();
	out.textContent = lines.join('\n');
};

// ------------------------------------------------------------- constants ---

const RES = 28;
const CH = 1; // grayscale ink
const PATCH = 4;
const TOK = (RES / PATCH) ** 2; // 49
const DIM = 192;
const HEADS = 4;
const LAYERS = 4;
const CLASSES = 10;
const TIME = 32;
const CONDW = TIME + CLASSES + 1; // time ‖ one-hot ‖ presence flag
const BATCH = 32;
const LR = 3e-4;
const SAMPLE_EVERY = 500;
const SHOWN = 8;
const PIXELS = CH * RES * RES;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

/** The cosine schedule: how much of the picture survives at noise level τ. */
function alphaBar(tau: number): number {
	const s = 0.008;
	const f = (u: number) => Math.cos(((u + s) / (1 + s)) * (Math.PI / 2)) ** 2;
	return Math.min(Math.max(f(tau) / f(0), 1e-6), 1);
}

function mulberry32(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function gauss(buf: Float32Array, rand: () => number) {
	for (let i = 0; i < buf.length; i += 2) {
		const u = Math.max(rand(), 1e-7);
		const r = Math.sqrt(-2 * Math.log(u));
		const th = 2 * Math.PI * rand();
		buf[i] = r * Math.cos(th);
		if (i + 1 < buf.length) buf[i + 1] = r * Math.sin(th);
	}
}

// ------------------------------------------------------------------ data ---

/** Cut a grayscale spritesheet into one plane per picture. Red carries the ink. */
async function loadSheet(path: string, cols: number, side: number, count: number) {
	const bmp = await createImageBitmap(await (await fetch(path)).blob());
	const rows = Math.ceil(count / cols);
	const c = new OffscreenCanvas(cols * side, rows * side);
	const ctx = c.getContext('2d')!;
	ctx.drawImage(bmp, 0, 0);
	const px = ctx.getImageData(0, 0, cols * side, rows * side).data;
	bmp.close();
	const plane = side * side;
	const imgs = new Uint8Array(count * plane);
	for (let i = 0; i < count; i++) {
		const tx = (i % cols) * side;
		const ty = Math.floor(i / cols) * side;
		for (let y = 0; y < side; y++) {
			for (let x = 0; x < side; x++) {
				imgs[i * plane + y * side + x] = px[((ty + y) * cols * side + tx + x) * 4];
			}
		}
	}
	return imgs;
}

// ----------------------------------------------------------------- model ---

function initParams(seed: number): Arr {
	const n = 8 + LAYERS * 8;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nk = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const g = (shape: number[], s: number) => random.normal(nk(), shape).mul(s);
	const s = Math.sqrt(3 / DIM);
	const u = (shape: number[], k: number) =>
		random.uniform(nk(), shape, { minval: -k * s, maxval: k * s });
	const p: Arr = {
		condIn: g([CONDW, DIM], 0.05),
		condOut: g([DIM, DIM], 0.05),
		patchify: g([DIM, CH, PATCH, PATCH], 0.05),
		pos: g([TOK, DIM], 0.02),
		blocks: [],
		headFilm: g([DIM, 2 * DIM], 0.02),
		unpatch: g([CH, DIM, PATCH, PATCH], 0.01)
	};
	for (let i = 0; i < LAYERS; i++) {
		p.blocks.push({
			film: g([DIM, 4 * DIM], 0.02),
			wq: u([DIM, DIM], 1),
			wk: u([DIM, DIM], 1),
			wv: u([DIM, DIM], 1),
			wo: u([DIM, DIM], 0.2),
			fc1: u([DIM, 4 * DIM], 0.4),
			fc2: u([4 * DIM, DIM], 0.2)
		});
	}
	return p;
}

const rmsnorm = (x: Arr) =>
	x.div(np.sqrt(np.mean(np.square(x.ref), -1, { keepdims: true }).add(1e-5)));

function forward(p: Arr, B: number, x: Arr, cond: Arr): Arr {
	const headDim = DIM / HEADS;
	const mod = (v: Arr, sb: Arr, off: number) => {
		const scale = sb.ref.slice([], [off, off + DIM]).reshape([B, 1, DIM]);
		const shift = sb.slice([], [off + DIM, off + 2 * DIM]).reshape([B, 1, DIM]);
		return v
			.reshape([B, TOK, DIM])
			.mul(scale.add(1))
			.add(shift)
			.reshape([B * TOK, DIM]);
	};
	const cvec = nn.silu(np.dot(nn.silu(np.dot(cond, p.condIn)), p.condOut));
	let h = lax.conv(x, p.patchify, [PATCH, PATCH], 'VALID');
	h = np.transpose(h.reshape([B, DIM, TOK]), [0, 2, 1]).reshape([B * TOK, DIM]);
	h = h.add(np.tile(p.pos, [B, 1]));
	for (const l of p.blocks) {
		const sb = np.dot(cvec.ref, l.film);
		const r1 = h.ref;
		const a = mod(rmsnorm(h), sb.ref, 0);
		const q = np.dot(a.ref, l.wq).reshape([B, TOK, HEADS, headDim]);
		const k = np.dot(a.ref, l.wk).reshape([B, TOK, HEADS, headDim]);
		const v = np.dot(a, l.wv).reshape([B, TOK, HEADS, headDim]);
		h = np.dot(nn.dotProductAttention(q, k, v).reshape([B * TOK, DIM]), l.wo).add(r1);
		const r2 = h.ref;
		let f = mod(rmsnorm(h), sb, 2 * DIM);
		f = nn.gelu(np.dot(f, l.fc1));
		h = np.dot(f, l.fc2).add(r2);
	}
	h = mod(rmsnorm(h), np.dot(cvec, p.headFilm), 0);
	h = np.transpose(h.reshape([B, TOK, DIM]), [0, 2, 1]).reshape([B, DIM, RES / PATCH, RES / PATCH]);
	return lax.convTranspose(h, p.unpatch, [PATCH, PATCH], 'VALID');
}

/**
 * One row of the conditioning vector. The trailing flag says whether the
 * one-hot means anything — without it, "no label" and "class zero" would
 * arrive as the same all-zero block.
 */
function writeCond(dst: Float32Array, row: number, tau: number, label: number | null) {
	const off = row * CONDW;
	dst.fill(0, off, off + CONDW);
	for (let k = 0; k < TIME / 2; k++) {
		const freq = Math.exp((k / (TIME / 2 - 1)) * Math.log(1000));
		dst[off + 2 * k] = Math.sin(tau * freq);
		dst[off + 2 * k + 1] = Math.cos(tau * freq);
	}
	if (label !== null) {
		dst[off + TIME + label] = 1;
		dst[off + TIME + CLASSES] = 1;
	}
}

// ------------------------------------------------------------------- run ---

function parseHex(hex: string): [number, number, number] {
	const h = hex.trim().replace('#', '');
	if (h.length >= 6) {
		return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
	}
	return [128, 128, 128];
}

/** Ink coverage (v+1)/2 onto the page colour — a sneaker is dark by day, pale by night. */
function paint(frame: Float32Array, n: number) {
	stage.hidden = false;
	stage.width = n * RES;
	stage.height = RES;
	const ctx = stage.getContext('2d')!;
	const img = ctx.createImageData(n * RES, RES);
	const css = getComputedStyle(document.documentElement);
	const [br, bg, bb] = parseHex(css.getPropertyValue('--paper'));
	const [ir, ig, ib] = parseHex(css.getPropertyValue('--ink'));
	for (let k = 0; k < n; k++) {
		for (let y = 0; y < RES; y++) {
			for (let x = 0; x < RES; x++) {
				const s = k * PIXELS + y * RES + x;
				const d = (y * n * RES + k * RES + x) * 4;
				const a = Math.max(0, Math.min(1, (frame[s] + 1) / 2));
				img.data[d] = ir * a + br * (1 - a);
				img.data[d + 1] = ig * a + bg * (1 - a);
				img.data[d + 2] = ib * a + bb * (1 - a);
				img.data[d + 3] = 255;
			}
		}
	}
	ctx.putImageData(img, 0, 0);
}

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu'))
		log('note: no WebGPU — falling back to wasm; expect slow steps.');
	defaultDevice(devices.includes('webgpu') ? 'webgpu' : 'wasm');

	const meta = await (await fetch('data/fashion-meta.json')).json();
	log(`loading ${meta.train} garments, ${meta.classes.length} classes…`);
	const [images, labelBuf] = await Promise.all([
		loadSheet('data/fashion-train.png', meta.cols, meta.side, meta.train),
		fetch('data/fashion-labels.bin').then((r) => r.arrayBuffer())
	]);
	const labels = new Uint8Array(labelBuf).subarray(0, meta.train);
	log(`corpus: ${meta.train} pictures at ${RES}x${RES} · ${meta.classes.join(', ')}`);

	let params = initParams(20260905);
	const nParams = (tree.leaves(tree.ref(params)) as Arr[]).reduce((s: number, l: Arr) => {
		const k = l.size;
		l.dispose();
		return s + k;
	}, 0);
	log(`model: ${(nParams / 1e6).toFixed(2)}M params, ${LAYERS} blocks of ${DIM}`);

	// Adam, hand-rolled so the update fits inside jit — optax's reads its step
	// counter with .item(), which a tracer cannot answer, and outside the jit
	// boundary the per-tensor kernel launches cost more than the matmuls.
	let m: Arr = tree.map((l: Arr) => np.zerosLike(l), tree.ref(params));
	let v: Arr = tree.map((l: Arr) => np.zerosLike(l), tree.ref(params));
	const step = jit((p: Arr, mm: Arr, vv: Arr, k: Arr, x: Arr, c: Arr, t: Arr) => {
		const [loss, grads] = valueAndGrad((pp: Arr) =>
			np.mean(np.square(forward(pp, BATCH, x, c).sub(t)))
		)(tree.ref(p));
		const [leaves, def] = tree.flatten(p) as [Arr[], Arr];
		const gl = tree.leaves(grads) as Arr[];
		const ml = tree.leaves(mm) as Arr[];
		const vl = tree.leaves(vv) as Arr[];
		const c1 = k.ref.slice([0, 1]);
		const c2 = k.slice([1, 2]);
		const np2: Arr[] = [];
		const nm: Arr[] = [];
		const nv: Arr[] = [];
		for (let i = 0; i < gl.length; i++) {
			const mi = ml[i].mul(0.9).add(gl[i].ref.mul(0.1));
			const vi = vl[i].mul(0.99).add(np.square(gl[i]).mul(0.01));
			np2.push(
				leaves[i].sub(
					mi.ref
						.mul(c1.ref)
						.mul(LR)
						.div(np.sqrt(vi.ref.mul(c2.ref)).add(1e-8))
				)
			);
			nm.push(mi);
			nv.push(vi);
		}
		c1.dispose();
		c2.dispose();
		return [loss, tree.unflatten(def, np2), tree.unflatten(def, nm), tree.unflatten(def, nv)];
	});

	const sampleNet = jit((p: Arr, x: Arr, c: Arr) => forward(p, SHOWN, x, c));
	const rand = mulberry32(1234);
	const xb = new Float32Array(BATCH * PIXELS);
	const tb = new Float32Array(BATCH * PIXELS);
	const cb = new Float32Array(BATCH * CONDW);
	const noise = new Float32Array(PIXELS);

	/** Walk pure static back into pictures, deterministically (DDIM, η = 0). */
	async function sample(steps: number) {
		const state = new Float32Array(SHOWN * PIXELS);
		const cond = new Float32Array(SHOWN * CONDW);
		gauss(state, mulberry32(99));
		for (let s = 0; s < steps; s++) {
			const tau = 1 - s / steps;
			const tauNext = 1 - (s + 1) / steps;
			// unconditional: the one-hot stays empty, the flag stays off
			for (let n = 0; n < SHOWN; n++) writeCond(cond, n, tau, null);
			const eps = await sampleNet(
				tree.ref(params),
				np.array(state).reshape([SHOWN, CH, RES, RES]),
				np.array(cond).reshape([SHOWN, CONDW])
			).data();
			const ab = alphaBar(tau);
			const abN = tauNext <= 0 ? 1 : alphaBar(tauNext);
			const sa = Math.sqrt(ab);
			const sn = Math.sqrt(1 - ab);
			for (let i = 0; i < state.length; i++) {
				const x0 = Math.min(Math.max((state[i] - sn * eps[i]) / sa, -1), 1);
				state[i] = Math.sqrt(abN) * x0 + Math.sqrt(Math.max(1 - abN, 0)) * eps[i];
			}
		}
		return state;
	}

	let t0 = performance.now();
	let ema = NaN;
	for (let it = 1; ; it++) {
		// build a batch: corrupt, and record the noise that would undo it
		for (let b = 0; b < BATCH; b++) {
			const idx = Math.floor(rand() * meta.train);
			const src = idx * PIXELS;
			gauss(noise, rand);
			const tau = rand();
			const ab = alphaBar(tau);
			const sa = Math.sqrt(ab);
			const sn = Math.sqrt(1 - ab);
			const o = b * PIXELS;
			for (let i = 0; i < PIXELS; i++) {
				xb[o + i] = sa * (images[src + i] / 127.5 - 1) + sn * noise[i];
				tb[o + i] = noise[i];
			}
			// a tenth of the rows drop the label, so one set of weights learns
			// both the conditional field and the unconditional one sampling uses
			writeCond(cb, b, tau, rand() < 0.1 ? null : labels[idx]);
		}
		const kk = np.array(
			new Float32Array([1 / (1 - Math.pow(0.9, it)), 1 / (1 - Math.pow(0.99, it))])
		);
		const [lossArr, p2, m2, v2] = step(
			params,
			m,
			v,
			kk,
			np.array(xb).reshape([BATCH, CH, RES, RES]),
			np.array(cb).reshape([BATCH, CONDW]),
			np.array(tb).reshape([BATCH, CH, RES, RES])
		);
		params = p2;
		m = m2;
		v = v2;
		const loss = lossArr.item();
		ema = Number.isNaN(ema) ? loss : ema * 0.98 + loss * 0.02;

		if (it % 100 === 0) {
			const ms = (performance.now() - t0) / 100;
			t0 = performance.now();
			log(`step ${it} · loss ${ema.toFixed(4)} · ${ms.toFixed(0)} ms/step`);
			await new Promise((r) => setTimeout(r, 0));
		}
		if (it % SAMPLE_EVERY === 0) {
			paint(await sample(50), SHOWN);
			log(`  ↑ ${SHOWN} pictures walked out of static in 50 steps`);
			await new Promise((r) => setTimeout(r, 0));
		}
	}
}

void main().catch((e) => log(`error: ${e?.stack ?? e}`));
