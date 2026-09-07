// Proof sheets for the two shipped checkpoints.
//
// Both chapters make claims that are empirical rather than arguable — that the
// velocity model survives a step budget the diffusion model does not, that the
// label steers what gets drawn, that the samples are not copies of the
// training set. Claims like that have to be looked at before they are written
// down. This page draws the evidence; scripts/verify-fashion.mjs pulls the
// PNGs out so they can be inspected as files.

import { init, defaultDevice, tree } from '@jax-js/jax';
import { FASHION_SHAPE, initParams, type DiffusionConfig } from '$lib/diffusion/model';
import { loadFashion } from '$lib/diffusion/corpus';
import { makeSampler, NOTHING, toUnit } from '$lib/diffusion/runtime';
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

const RES = 28;
const DIM = RES * RES;
const BUDGETS = [50, 20, 10, 4, 2];
const SEEDS = 6;

declare global {
	interface Window {
		__sheets?: Record<string, string>;
		__done?: boolean;
	}
}

/** Compose tiles into an offscreen canvas, ink on paper, with row labels. */
function draw(
	rows: { label: string; pixels: (Float32Array | null)[] }[],
	perRow: number,
	zoom = 3
) {
	const pad = 130;
	const w = pad + perRow * RES * zoom;
	const h = rows.length * (RES * zoom + 6);
	sheet.width = w;
	sheet.height = h;
	const ctx = sheet.getContext('2d')!;
	ctx.fillStyle = '#e8e6e0';
	ctx.fillRect(0, 0, w, h);
	rows.forEach((row, r) => {
		const y0 = r * (RES * zoom + 6);
		ctx.fillStyle = '#111';
		ctx.font = '13px ui-monospace, monospace';
		ctx.fillText(row.label, 6, y0 + RES * zoom * 0.6);
		row.pixels.forEach((px, k) => {
			if (!px) return;
			const img = ctx.createImageData(RES, RES);
			for (let p = 0; p < DIM; p++) {
				// ink coverage over paper, the same reading the plates use
				const a = Math.max(0, Math.min(1, (px[p] + 1) / 2));
				const v = Math.round(232 * (1 - a) + 17 * a);
				const d = p * 4;
				img.data[d] = v;
				img.data[d + 1] = v;
				img.data[d + 2] = v;
				img.data[d + 3] = 255;
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

	const fashion = await loadFashion('');
	const names = fashion.meta.classes;

	// A sheet is only worth drawing if it can be drawn from any candidate, not
	// just the one the book currently ships: the whole point is comparing them.
	const qs = new URLSearchParams(location.search);
	const tag = qs.get('tag') ?? '';
	const base = {
		...FASHION_SHAPE,
		dim: Number(qs.get('dim') ?? FASHION_SHAPE.dim),
		layers: Number(qs.get('layers') ?? FASHION_SHAPE.layers),
		heads: Number(qs.get('heads') ?? FASHION_SHAPE.heads),
		patch: Number(qs.get('patch') ?? FASHION_SHAPE.patch),
		classes: names.length
	};

	async function load(
		objective: 'eps' | 'flow'
	): Promise<{ cfg: DiffusionConfig; w: Arr; steps: number }> {
		const cfg: DiffusionConfig = { ...base, objective };
		const file = `data/fashion-${objective}${tag ? `-${tag}` : ''}.bin`;
		const buf = await (await fetch(file)).arrayBuffer();
		const { params, steps } = unpack(buf, initParams(cfg, 0));
		log(`${objective}: checkpoint at step ${steps}`);
		return { cfg, w: params, steps };
	}

	// A candidate architecture is trained one objective at a time, so the
	// sheets have to be drawable from whichever half exists yet.
	const only = qs.get('only');
	const eps = only === 'flow' ? null : await load('eps');
	const flow = only === 'eps' ? null : await load('flow');
	const main = flow ?? eps!;

	// ── sheet 1: every class, which is the whole promise of the demo ──
	const classSampler = makeSampler(main.cfg, SEEDS);
	const classRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (let c = 0; c < names.length; c++) {
		const px = await classSampler.run({
			params: tree.ref(main.w),
			steps: 20,
			a: { label: c },
			guidanceA: 2,
			seed: 2026
		});
		classRows.push({ label: names[c], pixels: split(px, SEEDS) });
		log(`  class ${names[c]}`);
	}
	window.__sheets.classes = draw(classRows, SEEDS);

	// ── sheet 2: what guidance actually buys, on one class ──
	const guideRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const w of [0, 1, 1.5, 2, 3, 5]) {
		const px = await classSampler.run({
			params: tree.ref(main.w),
			steps: 20,
			a: { label: 9 },
			guidanceA: w,
			seed: 404
		});
		guideRows.push({ label: `ankle boot · w = ${w}`, pixels: split(px, SEEDS) });
	}
	window.__sheets.guidance = draw(guideRows, SEEDS);

	// ── sheet 3: the step budget, both objectives ──
	const budgetRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const [name, m] of (eps && flow
		? [
				['noise', eps],
				['velocity', flow]
			]
		: [[flow ? 'velocity' : 'noise', main]]) as [string, { cfg: DiffusionConfig; w: Arr }][]) {
		const sampler = makeSampler(m.cfg, SEEDS);
		for (const steps of BUDGETS) {
			const px = await sampler.run({
				params: tree.ref(m.w),
				steps,
				a: { label: 7 },
				guidanceA: 2,
				seed: 88
			});
			budgetRows.push({ label: `${name} · ${steps} steps`, pixels: split(px, SEEDS) });
			log(`  ${name} @ ${steps} steps`);
		}
	}
	window.__sheets.budget = draw(budgetRows, SEEDS);

	// ── sheet 4: walking the label from one garment to another ──
	// The conditioning is a one-hot, so a partial mix is simply a vector the
	// model was never trained on — which is exactly the question worth asking.
	const morphRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	for (const [a, b] of [
		[7, 9],
		[0, 3],
		[8, 5]
	]) {
		const frames: (Float32Array | null)[] = [];
		for (let i = 0; i < SEEDS; i++) {
			const t = i / (SEEDS - 1);
			const px = await classSampler.run({
				params: tree.ref(main.w),
				steps: 20,
				a: { label: a, other: b, mix: t },
				guidanceA: 2,
				seed: 5150
			});
			frames.push(px.slice(0, DIM));
		}
		morphRows.push({ label: `${names[a]} → ${names[b]}`, pixels: frames });
		log(`  morph ${names[a]} → ${names[b]}`);
	}
	window.__sheets.morph = draw(morphRows, SEEDS);

	// ── sheet 5: is it copying? ──
	const nnSampler = makeSampler(main.cfg, 8);
	const drawn = await nnSampler.run({
		params: tree.ref(main.w),
		steps: 24,
		a: NOTHING,
		guidanceA: 1,
		seed: 31337
	});
	const nnRows: { label: string; pixels: (Float32Array | null)[] }[] = [];
	const made = split(drawn, 8);
	const nearest: (Float32Array | null)[] = [];
	const dists: number[] = [];
	for (const g of made) {
		let best = Infinity;
		let bestIdx = 0;
		for (let i = 0; i < fashion.count; i++) {
			let d = 0;
			// a quarter of the pixels is enough to rank neighbours and is four
			// times faster over twelve thousand candidates
			for (let p = 0; p < DIM; p += 4) {
				const diff = g[p] - toUnit(fashion.images[i * DIM + p]);
				d += diff * diff;
			}
			if (d < best) {
				best = d;
				bestIdx = i;
			}
		}
		const real = new Float32Array(DIM);
		for (let p = 0; p < DIM; p++) real[p] = toUnit(fashion.images[bestIdx * DIM + p]);
		nearest.push(real);
		dists.push(Math.sqrt(best));
	}
	nnRows.push({ label: 'drawn', pixels: made });
	nnRows.push({ label: 'nearest real', pixels: nearest });
	window.__sheets.nearest = draw(nnRows, 8);
	log(`nearest-neighbour distances: ${dists.map((d) => d.toFixed(1)).join(' ')}`);

	log('done');
	window.__done = true;
}

main().catch((e) => {
	log(`error: ${e}`);
	window.__done = true;
});
