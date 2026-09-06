// Lab 10 — the straight path.
//
// Rectified flow on 32x32 emoji, with prompts. The picture and the noise are
// joined by a straight line; the model is trained to predict the velocity
// along it; sampling is Euler's method and nothing else.
//
// The last third of the file is the part worth reading: guidance and
// composition are the same expression, and drawing a cat with a heart for a
// face costs one extra term.

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

const RES = 32;
const CH = 4;
const PATCH = 4;
const TOK = (RES / PATCH) ** 2;
const DIM = 192;
const HEADS = 4;
const LAYERS = 4;
const BATCH = 32;
const LR = 3e-4;
const TIME = 32;
const SAMPLE_EVERY = 500;
const SHOWN = 8;
const PIXELS = CH * RES * RES;

/** Change these. Words outside the vocabulary are reported and ignored. */
const PROMPT_A = 'cat face';
const PROMPT_B = 'heart';
const GUIDE_A = 3;
const GUIDE_B = 3;
const SAMPLE_STEPS = 16;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

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

async function loadSheet(id: string, cols: number, rows: number, count: number) {
	const bmp = await createImageBitmap(await (await fetch(`data/emoji-${id}.png`)).blob());
	const c = new OffscreenCanvas(cols * RES, rows * RES);
	const ctx = c.getContext('2d')!;
	ctx.drawImage(bmp, 0, 0);
	const px = ctx.getImageData(0, 0, cols * RES, rows * RES).data;
	const plane = RES * RES;
	const imgs = new Uint8Array(count * PIXELS);
	for (let i = 0; i < count; i++) {
		const tx = (i % cols) * RES;
		const ty = Math.floor(i / cols) * RES;
		for (let y = 0; y < RES; y++) {
			for (let x = 0; x < RES; x++) {
				const o = ((ty + y) * cols * RES + tx + x) * 4;
				const p = y * RES + x;
				const d = i * PIXELS + p;
				imgs[d] = px[o];
				imgs[d + plane] = px[o + 1];
				imgs[d + 2 * plane] = px[o + 2];
				imgs[d + 3 * plane] = px[o + 3];
			}
		}
	}
	return imgs;
}

// ----------------------------------------------------------------- model ---

let CONDW = 0; // time ‖ tags ‖ styles ‖ two flags — filled in once meta loads
let NTAGS = 0;
let NSTYLES = 0;

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

/** One row of the conditioning vector: noise level, tags, style, two flags. */
function writeCond(
	dst: Float32Array,
	row: number,
	tau: number,
	tags: number[],
	style: number | null
) {
	const off = row * CONDW;
	dst.fill(0, off, off + CONDW);
	for (let k = 0; k < TIME / 2; k++) {
		const freq = Math.exp((k / (TIME / 2 - 1)) * Math.log(1000));
		dst[off + 2 * k] = Math.sin(tau * freq);
		dst[off + 2 * k + 1] = Math.cos(tau * freq);
	}
	const tagBase = off + TIME;
	const flagBase = tagBase + NTAGS + NSTYLES;
	if (tags.length) {
		const w = 1 / Math.sqrt(tags.length);
		for (const t of tags) dst[tagBase + t] = w;
		dst[flagBase] = 1;
	}
	if (style !== null) {
		dst[tagBase + NTAGS + style] = 1;
		dst[flagBase + 1] = 1;
	}
}

// ------------------------------------------------------------------- run ---

function paint(frame: Float32Array, n: number) {
	stage.hidden = false;
	stage.width = n * RES;
	stage.height = RES;
	const ctx = stage.getContext('2d')!;
	const img = ctx.createImageData(n * RES, RES);
	const plane = RES * RES;
	for (let k = 0; k < n; k++) {
		for (let y = 0; y < RES; y++) {
			for (let x = 0; x < RES; x++) {
				const p = y * RES + x;
				const s = k * PIXELS + p;
				const d = (y * n * RES + k * RES + x) * 4;
				const to = (v: number) => Math.max(0, Math.min(255, (v + 1) * 127.5));
				const a = to(frame[s + 3 * plane]) / 255;
				for (let c = 0; c < 3; c++) img.data[d + c] = to(frame[s + c * plane]) + 128 * (1 - a);
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

	const meta = await (await fetch('data/emoji-meta.json')).json();
	const SETS = meta.sets.map((s: { id: string }) => s.id);
	NTAGS = meta.tags.length;
	NSTYLES = SETS.length;
	CONDW = TIME + NTAGS + NSTYLES + 2;
	const tagIndex = new Map<string, number>(meta.tags.map((t: string, i: number) => [t, i]));
	const parse = (text: string) => {
		const ids: number[] = [];
		const missed: string[] = [];
		for (const w of text.toLowerCase().split(/[^a-z]+/)) {
			if (w.length < 2) continue;
			const i = tagIndex.get(w);
			if (i === undefined) missed.push(w);
			else ids.push(i);
		}
		if (missed.length) log(`  (not in vocabulary, ignored: ${missed.join(', ')})`);
		return ids;
	};

	log(`loading ${meta.count} emoji x ${NSTYLES} styles, ${NTAGS} tags…`);
	const sheets: Uint8Array[] = [];
	for (const id of SETS) sheets.push(await loadSheet(id, meta.cols, meta.rows, meta.count));
	const tagsFor: number[][] = meta.emoji.map((e: { tags: number[] }) => e.tags);

	const tagsA = parse(PROMPT_A);
	const tagsB = parse(PROMPT_B);
	log(`prompt A "${PROMPT_A}" → ${tagsA.length} tags · B "${PROMPT_B}" → ${tagsB.length} tags`);

	let params = initParams(20260905);
	const nParams = (tree.leaves(tree.ref(params)) as Arr[]).reduce((s: number, l: Arr) => {
		const k = l.size;
		l.dispose();
		return s + k;
	}, 0);
	log(`model: ${(nParams / 1e6).toFixed(2)}M params`);

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

	// Three branches per picture: no prompt, prompt A, prompt B. Fixing the
	// count keeps one compiled shape whether you compose one prompt or two.
	const BR = 3;
	const sampleNet = jit((p: Arr, x: Arr, c: Arr) => forward(p, BR * SHOWN, x, c));

	async function sample(steps: number) {
		const state = new Float32Array(SHOWN * PIXELS);
		const xb = new Float32Array(BR * SHOWN * PIXELS);
		const cb = new Float32Array(BR * SHOWN * CONDW);
		const field = new Float32Array(SHOWN * PIXELS);
		gauss(state, mulberry32(99));
		for (let s = 0; s < steps; s++) {
			const tau = 1 - s / steps;
			const dt = 1 / steps;
			for (let br = 0; br < BR; br++) xb.set(state, br * SHOWN * PIXELS);
			for (let n = 0; n < SHOWN; n++) {
				writeCond(cb, n, tau, [], null);
				writeCond(cb, SHOWN + n, tau, tagsA, null);
				writeCond(cb, 2 * SHOWN + n, tau, tagsB, null);
			}
			const o = await sampleNet(
				tree.ref(params),
				np.array(xb).reshape([BR * SHOWN, CH, RES, RES]),
				np.array(cb).reshape([BR * SHOWN, CONDW])
			).data();
			// guidance and composition, in one expression: start from what any
			// picture would do, then add each prompt's difference from it
			const aOff = SHOWN * PIXELS;
			const bOff = 2 * SHOWN * PIXELS;
			for (let i = 0; i < field.length; i++) {
				const u = o[i];
				field[i] = u + GUIDE_A * (o[aOff + i] - u) + GUIDE_B * (o[bOff + i] - u);
			}
			for (let i = 0; i < state.length; i++) state[i] -= dt * field[i]; // Euler
		}
		return state;
	}

	const rand = mulberry32(1234);
	const xb = new Float32Array(BATCH * PIXELS);
	const tb = new Float32Array(BATCH * PIXELS);
	const cb = new Float32Array(BATCH * CONDW);
	const noise = new Float32Array(PIXELS);

	let t0 = performance.now();
	let ema = NaN;
	for (let it = 1; ; it++) {
		for (let b = 0; b < BATCH; b++) {
			const style = Math.floor(rand() * NSTYLES);
			const idx = Math.floor(rand() * meta.count);
			const src = idx * PIXELS;
			gauss(noise, rand);
			const tau = rand();
			const o = b * PIXELS;
			for (let i = 0; i < PIXELS; i++) {
				const x0 = sheets[style][src + i] / 127.5 - 1;
				xb[o + i] = (1 - tau) * x0 + tau * noise[i]; // the straight line
				tb[o + i] = noise[i] - x0; // and its slope
			}
			// drop the prompt a tenth of the time, and the style a tenth, so one
			// model learns the conditional and unconditional fields at once
			writeCond(cb, b, tau, rand() < 0.1 ? [] : tagsFor[idx], rand() < 0.1 ? null : style);
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
			paint(await sample(SAMPLE_STEPS), SHOWN);
			log(`  ↑ "${PROMPT_A}" + "${PROMPT_B}" in ${SAMPLE_STEPS} Euler steps`);
			await new Promise((r) => setTimeout(r, 0));
		}
	}
}

void main().catch((e) => log(`error: ${e?.stack ?? e}`));
