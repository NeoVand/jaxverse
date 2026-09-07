// Which recipe is worth ten hours?
//
// The first emoji model shipped at 24,000 steps and its faces were bad enough
// to be unusable. Rather than guess at the cause, this trains several recipes
// for the same wall clock and draws their samples in one sheet, so the choice
// is made by looking rather than by argument. Equal time, not equal steps —
// a recipe that is twice as good per step and three times as slow is not a
// better recipe.
//
// Driven by scripts/ablate-recipe.mjs.

import { blockUntilReady, init, defaultDevice, jit, numpy as np, tree } from '@jax-js/jax';
import { initParams, paramCount, type DiffusionConfig } from '$lib/diffusion/model';
import { loadEmoji, makeVocab, parsePrompt } from '$lib/diffusion/corpus';
import {
	allocBatch,
	makeBatch,
	makeOptimizer,
	makeSampler,
	mulberry32,
	type BatchOptions,
	type Corpus
} from '$lib/diffusion/runtime';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

const out = document.getElementById('out') as HTMLPreElement;
const sheet = document.getElementById('sheet') as HTMLCanvasElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.join('\n');
};

const qs = new URLSearchParams(location.search);
const MINUTES = Number(qs.get('minutes') ?? 10);
const BATCH = Number(qs.get('batch') ?? 64);
const LR = 3e-4;
const WARMUP = 300;
const EMA_DECAY = 0.9995;
const RES = 32;
const SHOWN = 6;
/** How many distinct faces to memorize. */
const FACES = 12;

/** The prompts the sheet is judged on — faces first, since faces are what
 *  went wrong, then a few solid shapes to check colour and silhouette. */
const PROMPTS = ['', 'smiling face', 'grinning face', 'crying face', 'face', 'smiling cat face'];

interface Recipe {
	name: string;
	patch: number;
	dim: number;
	layers: number;
	heads: number;
	batchOpts: BatchOptions;
	/** Patch 2 diverges at the default rate; it needs its own. */
	lr?: number;
}

// Round two, one variable at a time. Round one changed the timestep
// distribution and the augmentation together and the result was worse than
// the control — washed out, with the colour gone — so logit-normal is dropped
// here rather than carried into every row as a confound. Every recipe below
// has the weight average and the gradient clip; those two are settled.
// Round three: can the architecture draw a face AT ALL?
//
// Faces come out smeared while hearts come out clean, and there are five times
// as many face pictures in the corpus as heart pictures — so it is not a
// shortage of examples. The suspicion is the decoder: at patch 4 a single
// token has to produce a 4x4 block of pixels through one linear layer, and an
// eye is two pixels across, so an eye lives entirely inside one token with no
// way for neighbouring tokens to help draw it.
//
// This is the test that settles it. Give each architecture ninety-six face
// pictures and long enough to memorize them outright. Anything that still
// cannot render a sharp eye is bottlenecked by its own shape, and no amount of
// training on the full corpus will rescue it.
const RECIPES: Recipe[] = [
	{
		name: 'A patch 4 (shipping now)',
		patch: 4,
		dim: 192,
		layers: 4,
		heads: 4,
		batchOpts: {}
	},
	{
		name: 'B patch 2, lr 1e-4',
		patch: 2,
		dim: 192,
		layers: 4,
		heads: 4,
		batchOpts: {},
		lr: 1e-4
	},
	{
		name: 'C patch 2, lr 5e-5',
		patch: 2,
		dim: 192,
		layers: 4,
		heads: 4,
		batchOpts: {},
		lr: 5e-5
	},
	{
		name: 'D patch 1, lr 5e-5',
		patch: 1,
		dim: 128,
		layers: 4,
		heads: 4,
		batchOpts: {},
		lr: 5e-5
	}
];

declare global {
	interface Window {
		__sheet?: string;
		__done?: boolean;
	}
}

const rows: { label: string; pixels: Float32Array[] }[] = [];

function drawSheet(): string {
	const pad = 300;
	const zoom = 3;
	const w = pad + SHOWN * RES * zoom;
	const h = rows.length * (RES * zoom + 6);
	sheet.width = w;
	sheet.height = h;
	const ctx = sheet.getContext('2d')!;
	ctx.fillStyle = '#8a8a8a';
	ctx.fillRect(0, 0, w, h);
	const plane = RES * RES;
	const dim = 4 * plane;
	rows.forEach((row, r) => {
		const y0 = r * (RES * zoom + 6);
		ctx.fillStyle = '#111';
		ctx.font = '12px ui-monospace, monospace';
		ctx.fillText(row.label, 6, y0 + RES * zoom * 0.6);
		row.pixels.forEach((px, k) => {
			const img = ctx.createImageData(RES, RES);
			for (let y = 0; y < RES; y++) {
				for (let x = 0; x < RES; x++) {
					const p = y * RES + x;
					const to = (v: number) => Math.max(0, Math.min(255, (v + 1) * 127.5));
					const a = to(px[p + 3 * plane]) / 255;
					const d = p * 4;
					for (let c = 0; c < 3; c++) img.data[d + c] = to(px[p + c * plane]) + 138 * (1 - a);
					img.data[d + 3] = 255;
				}
			}
			const tmp = new OffscreenCanvas(RES, RES);
			tmp.getContext('2d')!.putImageData(img, 0, 0);
			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(tmp, pad + k * RES * zoom, y0, RES * zoom, RES * zoom);
		});
		void dim;
	});
	return sheet.toDataURL('image/png');
}

async function runRecipe(
	r: Recipe,
	corpus: Corpus,
	tagsFor: Record<string, number[]>,
	nTags: number
) {
	const cfg: DiffusionConfig = {
		res: RES,
		channels: 4,
		patch: r.patch,
		dim: r.dim,
		layers: r.layers,
		heads: r.heads,
		tags: nTags,
		styles: corpus.styles,
		objective: 'flow'
	};
	let weights: Arr = initParams(cfg, 20260906);
	let ema: Arr = tree.map((l: Arr) => l.mul(1), tree.ref(weights));
	const emaStep = jit((e: Arr, p: Arr, k: Arr) => {
		const [el, def] = tree.flatten(e) as [Arr[], Arr];
		const pl = tree.leaves(p) as Arr[];
		const keep = k.ref.slice([0, 1]);
		const take = k.slice([1, 2]);
		const o: Arr[] = [];
		for (let i = 0; i < el.length; i++) o.push(el[i].mul(keep.ref).add(pl[i].mul(take.ref)));
		keep.dispose();
		take.dispose();
		return tree.unflatten(def, o);
	});

	const opt = makeOptimizer(cfg, BATCH, weights);
	let batch = allocBatch(cfg, BATCH);
	let spare = allocBatch(cfg, BATCH);
	const rand = mulberry32(1234);
	makeBatch(corpus, cfg, BATCH, rand, batch, r.batchOpts);

	const deadline = performance.now() + MINUTES * 60_000;
	let step = 0;
	let loss = NaN;
	while (performance.now() < deadline) {
		const warm = Math.min(1, (step + 1) / WARMUP);
		// anneal over the budget this recipe actually gets
		const frac = 1 - (deadline - performance.now()) / (MINUTES * 60_000);
		const baseLr = r.lr ?? LR;
		const lr = baseLr * warm * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, frac))));
		const [lossArr, next] = opt.step(weights, batch, lr);
		weights = next;
		makeBatch(corpus, cfg, BATCH, rand, spare, r.batchOpts);
		[batch, spare] = [spare, batch];
		const l = lossArr.item();
		step++;
		const d = Math.min(EMA_DECAY, (1 + step) / (10 + step));
		ema = emaStep(ema, tree.ref(weights), np.array(new Float32Array([d, 1 - d])));
		loss = Number.isNaN(loss) ? l : loss * 0.98 + l * 0.02;
		if (step % 200 === 0) log(`   ${r.name} · step ${step} · loss ${loss.toFixed(4)}`);
		if (step % 20 === 0) {
			// Force the average. Reading the loss each step realizes the weight
			// graph, but nothing ever reads the average, so its deferred
			// operations pile up one per step and the stack overflows a few
			// thousand steps in. Draining it here keeps the graph shallow.
			await blockUntilReady(tree.ref(ema));
			// Yield long enough that the driver's CDP calls get scheduled: a
			// zero-millisecond yield leaves the main thread busy essentially all
			// the time and the run looks hung from outside.
			await new Promise((res) => setTimeout(res, 2));
		}
	}

	// One picture per prompt, same seed and same guidance in every row, so the
	// only thing differing down a column is the recipe.
	const sampler = makeSampler(cfg, 1);
	const pixels: Float32Array[] = [];
	for (const prompt of PROMPTS) {
		const frame = await sampler.run({
			params: tree.ref(ema),
			steps: 24,
			a: { tags: tagsFor[prompt] ?? [], style: null },
			guidanceA: 2,
			seed: 4242
		});
		pixels.push(frame.slice());
	}

	log(
		`✓ ${r.name} — ${(paramCount(cfg) / 1e6).toFixed(2)}M params, ${step} steps, loss ${loss.toFixed(4)}`
	);
	rows.push({ label: `${r.name} · ${step} steps`, pixels });
	window.__sheet = drawSheet();

	opt.dispose();
	tree.dispose(weights);
	tree.dispose(ema);
	return { steps: step, loss };
}

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu')) {
		log('error: no WebGPU');
		return;
	}
	defaultDevice('webgpu');

	const emoji = await loadEmoji('');
	const nTags = emoji.meta.tags.length;
	const vocab = makeVocab(emoji.meta.tags);
	const tagsFor: Record<string, number[]> = {};
	for (const p of PROMPTS) tagsFor[p] = parsePrompt(p, vocab).tags;

	// Cut the corpus down to a handful of faces, every style. Small enough to
	// memorize, so what is left on the page is the architecture's ceiling
	// rather than its progress.
	const faceTag = vocab.index.get('face')!;
	const keep: number[] = [];
	for (let i = 0; i < emoji.count && keep.length < FACES; i++) {
		if (emoji.tags[i].includes(faceTag)) keep.push(i);
	}
	const dim = 4 * RES * RES;
	const small: Corpus = {
		images: new Uint8Array(keep.length * emoji.styles * dim),
		tags: keep.map((i) => emoji.tags[i]),
		count: keep.length,
		styles: emoji.styles
	};
	for (let s = 0; s < emoji.styles; s++) {
		keep.forEach((src, k) => {
			small.images.set(
				emoji.images.subarray((s * emoji.count + src) * dim, (s * emoji.count + src + 1) * dim),
				(s * keep.length + k) * dim
			);
		});
	}
	log(
		`overfit set: ${keep.length} faces x ${emoji.styles} styles = ${keep.length * emoji.styles} pictures · ${MINUTES} min each`
	);
	log(`faces: ${keep.map((i) => emoji.meta.emoji[i].cp).join(' ')}`);

	for (const r of RECIPES) {
		log(`\n▸ ${r.name}`);
		await runRecipe(r, small, tagsFor, nTags);
	}
	log('\ndone');
	window.__done = true;
}

void main().catch((e) => log(`error: ${e?.stack ?? e}`));
