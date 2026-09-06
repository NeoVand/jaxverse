// Proof sheets for the two shipped checkpoints.
//
// Two of this book's captions make empirical claims — that the velocity model
// survives a four-step budget the diffusion model does not, and that prompts
// and styles steer independently. Claims like that have to be looked at before
// they are written down. This page draws the evidence; scripts/verify-emoji.mjs
// pulls the PNGs out so they can be inspected as files.

import { init, defaultDevice, tree } from '@jax-js/jax';
import { EMOJI_SHAPE, initParams, type DiffusionConfig } from '$lib/diffusion/model';
import { loadEmoji, makeVocab, parsePrompt } from '$lib/diffusion/corpus';
import { makeSampler, NOTHING, type Condition } from '$lib/diffusion/runtime';
import { unpack } from '$lib/diffusion/checkpoint';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

const out = document.getElementById('out') as HTMLPreElement;
const sheet = document.getElementById('sheet') as HTMLCanvasElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.join('\n');
};

const RES = 32;
const DIM = 4 * RES * RES;
const BUDGETS = [50, 20, 10, 4, 2];
const SEEDS = 5;

declare global {
	interface Window {
		__sheets?: Record<string, string>;
		__done?: boolean;
	}
}

/** Compose tiles into an offscreen canvas over mid grey, with row labels. */
function draw(
	rows: { label: string; pixels: (Float32Array | null)[] }[],
	perRow: number,
	zoom = 3
) {
	const pad = 150;
	const w = pad + perRow * RES * zoom;
	const h = rows.length * (RES * zoom + 6);
	sheet.width = w;
	sheet.height = h;
	const ctx = sheet.getContext('2d')!;
	ctx.fillStyle = '#8a8a8a';
	ctx.fillRect(0, 0, w, h);
	const plane = RES * RES;
	rows.forEach((row, r) => {
		const y0 = r * (RES * zoom + 6);
		ctx.fillStyle = '#111';
		ctx.font = '13px ui-monospace, monospace';
		ctx.fillText(row.label, 6, y0 + RES * zoom * 0.6);
		row.pixels.forEach((px, k) => {
			if (!px) return;
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
	});
	return sheet.toDataURL('image/png');
}

/** Split a [count · DIM] block into one array per picture. */
const split = (px: Float32Array, n: number) =>
	Array.from({ length: n }, (_, i) => px.slice(i * DIM, (i + 1) * DIM));

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu')) {
		log('error: no WebGPU');
		return;
	}
	defaultDevice('webgpu');
	window.__sheets = {};

	const emoji = await loadEmoji('');
	const vocab = makeVocab(emoji.meta.tags);
	const base = { ...EMOJI_SHAPE, tags: emoji.meta.tags.length, styles: emoji.meta.sets.length };

	async function load(
		objective: 'eps' | 'flow'
	): Promise<{ cfg: DiffusionConfig; w: Arr; steps: number }> {
		const cfg: DiffusionConfig = { ...base, objective };
		const buf = await (await fetch(`data/emoji-${objective}.bin`)).arrayBuffer();
		const { params, steps } = unpack(buf, initParams(cfg, 0));
		log(`${objective}: checkpoint at step ${steps}`);
		return { cfg, w: params, steps };
	}

	const eps = await load('eps');
	const flow = await load('flow');

	// ── sheet 1: the step-count budget, both models ──
	const budgetRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const [name, m] of [
		['noise', eps],
		['velocity', flow]
	] as const) {
		const sampler = makeSampler(m.cfg, SEEDS);
		for (const steps of BUDGETS) {
			const px = await sampler.run({
				params: tree.ref(m.w),
				steps,
				a: NOTHING,
				guidanceA: 1,
				seed: 31
			});
			budgetRows.push({ label: `${name} · ${steps} steps`, pixels: split(px, SEEDS) });
			log(`  ${name} @ ${steps} steps`);
		}
	}
	window.__sheets.budget = draw(budgetRows, SEEDS);

	// ── sheet 2: prompts, at the studio's default guidance ──
	const PROMPTS = [
		'smiling cat face',
		'red heart',
		'star',
		'ghost',
		'fire',
		'moon',
		'flower',
		'tree'
	];
	const promptSampler = makeSampler(flow.cfg, 6);
	const promptRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const p of PROMPTS) {
		const parsed = parsePrompt(p, vocab);
		const px = await promptSampler.run({
			params: tree.ref(flow.w),
			steps: 20,
			a: { tags: parsed.tags, style: null },
			guidanceA: 3,
			seed: 2026
		});
		promptRows.push({ label: `${p} (${parsed.matched.length})`, pixels: split(px, 6) });
		log(`  prompt "${p}" → ${parsed.matched.join(',') || 'no tags'}`);
	}
	window.__sheets.prompts = draw(promptRows, 6);

	// ── sheet 3: one idea in every style ──
	const styleRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	const ghost = parsePrompt('ghost', vocab).tags;
	for (let s = 0; s < emoji.meta.sets.length; s++) {
		const px = await promptSampler.run({
			params: tree.ref(flow.w),
			steps: 20,
			a: { tags: ghost, style: s },
			guidanceA: 3,
			seed: 1212
		});
		styleRows.push({ label: emoji.meta.sets[s].label, pixels: split(px, 6) });
	}
	window.__sheets.styles = draw(styleRows, 6);

	// ── sheet 4: composing two prompts ──
	const comboRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	const COMBOS: [string, string][] = [
		['cat face', 'heart'],
		['ghost', 'flower'],
		['moon', 'smiling face'],
		['star', 'clock'],
		['bird', 'fire']
	];
	for (const [pa, pb] of COMBOS) {
		const a: Condition = { tags: parsePrompt(pa, vocab).tags, style: null };
		const b: Condition = { tags: parsePrompt(pb, vocab).tags, style: null };
		for (const [label, cond] of [
			[pa, { a, b: undefined, wb: 0 }],
			[pb, { a: b, b: undefined, wb: 0 }],
			[`${pa} + ${pb}`, { a, b, wb: 3 }]
		] as const) {
			const px = await promptSampler.run({
				params: tree.ref(flow.w),
				steps: 20,
				a: cond.a,
				b: cond.b,
				guidanceA: 3,
				guidanceB: cond.wb,
				seed: 515
			});
			comboRows.push({ label, pixels: split(px, 6) });
		}
	}
	window.__sheets.combos = draw(comboRows, 6);

	// ── sheet 5: the guidance sweep, whose caption names specific behaviour ──
	const guideSampler = makeSampler(flow.cfg, 4);
	const heart = parsePrompt('red heart', vocab).tags;
	const guideRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const w of [0, 1, 2, 4, 8]) {
		const px = await guideSampler.run({
			params: tree.ref(flow.w),
			steps: 20,
			a: { tags: heart, style: null },
			guidanceA: w,
			seed: 808
		});
		guideRows.push({ label: `w = ${w}`, pixels: split(px, 4) });
	}
	window.__sheets.guidance = draw(guideRows, 4);

	// ── sheet 6: is it copying? ──
	//
	// The chapters claim these models draw things that are not in the corpus.
	// With 2.6M parameters and 8,656 pictures that claim has to be checked, not
	// assumed. For each sample, find the nearest training picture by plain
	// squared distance over all 8,656 and put them side by side.
	const nnSampler = makeSampler(flow.cfg, 8);
	const samples = await nnSampler.run({
		params: tree.ref(flow.w),
		steps: 24,
		a: NOTHING,
		guidanceA: 1,
		seed: 77
	});
	const nnRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	const drawn = split(samples, 8);
	const neighbours: (Float32Array | null)[] = [];
	const dists: number[] = [];
	for (const s of drawn) {
		let best = Infinity;
		let bestIdx = 0;
		for (let i = 0; i < emoji.images.length / DIM; i++) {
			let d = 0;
			const off = i * DIM;
			for (let k = 0; k < DIM; k += 4) {
				const diff = s[k] - (emoji.images[off + k] / 127.5 - 1);
				d += diff * diff;
				if (d > best) break;
			}
			if (d < best) {
				best = d;
				bestIdx = i;
			}
		}
		const nn = new Float32Array(DIM);
		for (let k = 0; k < DIM; k++) nn[k] = emoji.images[bestIdx * DIM + k] / 127.5 - 1;
		neighbours.push(nn);
		dists.push(best);
	}
	nnRows.push({ label: 'drawn', pixels: drawn });
	nnRows.push({ label: 'nearest real', pixels: neighbours });
	window.__sheets.nearest = draw(nnRows, 8);
	log(
		`nearest-neighbour distances (sampled quarter of channels): ${dists.map((d) => d.toFixed(1)).join(' ')}`
	);

	log('done');
	window.__done = true;
}

void main().catch((e) => log(`error: ${e?.stack ?? e}`));
