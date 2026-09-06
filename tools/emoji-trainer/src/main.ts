// Offline trainer for the shipped emoji checkpoints.
//
// It runs in headless Chromium so that the weights the book ships were
// produced by the same jax-js, the same kernels and the same code as the
// weights a reader trains in their own tab. scripts/train-emoji.mjs drives it
// and pulls snapshots out through window.__snapshot().

import { init, defaultDevice, tree } from '@jax-js/jax';
import {
	EMOJI_SHAPE,
	initParams,
	type DiffusionConfig,
	type Objective
} from '$lib/diffusion/model';
import { loadEmoji } from '$lib/diffusion/corpus';
import {
	allocBatch,
	makeBatch,
	makeOptimizer,
	makeSampler,
	mulberry32,
	type Corpus
} from '$lib/diffusion/runtime';
import { packFloat32, packInt8, unpack } from '$lib/diffusion/checkpoint';

/* eslint-disable @typescript-eslint/no-explicit-any */
type Arr = any;

const out = document.getElementById('out') as HTMLPreElement;
const preview = document.getElementById('preview') as HTMLCanvasElement;
let head = '';
let tail: string[] = [];
const log = (s: string) => {
	tail.push(s);
	if (tail.length > 14) tail = tail.slice(-14);
	out.textContent = `${head}\n${tail.join('\n')}`;
};

const params = new URLSearchParams(location.search);
const OBJECTIVE = (params.get('objective') ?? 'flow') as Objective;
const BATCH = Number(params.get('batch') ?? 32);
const LR = Number(params.get('lr') ?? 3e-4);
const WARMUP = 300;
/** Horizon for the cosine decay. Stopping early just means it never finished
 *  decaying, which costs a little polish and nothing else. */
const HORIZON = Number(params.get('horizon') ?? 55000);

declare global {
	interface Window {
		__snapshot?: (kind: 'f32' | 'i8') => string;
		__resume?: (b64: string) => void;
		__stop?: () => void;
		__state?: { step: number; loss: number; ready: boolean };
	}
}

function toBase64(buf: ArrayBuffer): string {
	const bytes = new Uint8Array(buf);
	let s = '';
	for (let i = 0; i < bytes.length; i += 0x8000) {
		s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
	}
	return btoa(s);
}

function fromBase64(b64: string): ArrayBuffer {
	const bin = atob(b64);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	return bytes.buffer;
}

/** Draw eight samples so a human watching the run can see it working. */
function paint(frame: Float32Array, cfg: DiffusionConfig, n: number) {
	const { res, channels } = cfg;
	const dim = channels * res * res;
	const ctx = preview.getContext('2d')!;
	const img = ctx.createImageData(res * n, res);
	for (let k = 0; k < n; k++) {
		for (let y = 0; y < res; y++) {
			for (let x = 0; x < res; x++) {
				const p = y * res + x;
				const s = k * dim + p;
				const d = (y * res * n + k * res + x) * 4;
				const a = Math.min(Math.max((frame[s + 3 * res * res] + 1) * 127.5, 0), 255);
				for (let c = 0; c < 3; c++) {
					const v = Math.min(Math.max((frame[s + c * res * res] + 1) * 127.5, 0), 255);
					// premultiplied over the mid grey the canvas is painted with
					img.data[d + c] = Math.round(v + 136 * (1 - a / 255));
				}
				img.data[d + 3] = 255;
			}
		}
	}
	ctx.putImageData(img, 0, 0);
	ctx.imageSmoothingEnabled = false;
	ctx.drawImage(preview, 0, 0, res * n, res, 0, 0, preview.width, preview.height);
}

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu')) {
		log('error: no WebGPU');
		return;
	}
	defaultDevice('webgpu');

	log('loading corpus…');
	const emoji = await loadEmoji('');
	const cfg: DiffusionConfig = {
		...EMOJI_SHAPE,
		tags: emoji.meta.tags.length,
		styles: emoji.meta.sets.length,
		objective: OBJECTIVE
	};
	const corpus: Corpus = emoji;

	let weights: Arr = initParams(cfg, 20260905);
	let step = 0;
	const nParams = (tree.leaves(tree.ref(weights)) as Arr[]).reduce((s: number, l: Arr) => {
		const n = l.size;
		l.dispose();
		return s + n;
	}, 0);
	head =
		`objective ${OBJECTIVE} · ${(nParams / 1e6).toFixed(2)}M params · batch ${BATCH} · lr ${LR}\n` +
		`${emoji.count} emoji x ${emoji.styles} styles · ${cfg.tags} tags`;

	const opt = makeOptimizer(cfg, BATCH, weights);
	// Two batch buffers, ping-ponged: the CPU builds the next one while the GPU
	// is still busy with this one. Corrupting a picture is a few hundred
	// thousand transcendentals per step, and unoverlapped it was most of the
	// wall clock.
	let batch = allocBatch(cfg, BATCH);
	let spare = allocBatch(cfg, BATCH);
	const rand = mulberry32(1234);
	const sampler = makeSampler(cfg, 8);
	makeBatch(corpus, cfg, BATCH, rand, batch);

	let stopped = false;
	window.__stop = () => {
		stopped = true;
	};
	window.__snapshot = (kind) =>
		toBase64(
			kind === 'i8' ? packInt8(weights, step, OBJECTIVE) : packFloat32(weights, step, OBJECTIVE)
		);
	window.__resume = (b64) => {
		const loaded = unpack(fromBase64(b64), weights);
		weights = loaded.params;
		step = loaded.steps;
		log(`resumed at step ${step}`);
	};
	window.__state = { step: 0, loss: NaN, ready: true };

	let ema = NaN;
	let sinceLog = performance.now();
	// Warmup counts from when THIS process started, not from the checkpoint's
	// step: a resumed run has restored weights but empty Adam moments, and a
	// full-rate step on top of empty moments is how a run dies in its first
	// second. Decay, by contrast, follows the absolute step.
	let sinceStart = 0;
	for (;;) {
		if (stopped) break;
		const warm = Math.min(1, ++sinceStart / WARMUP);
		const decay = 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, step / HORIZON)));
		const lr = LR * warm * decay;
		const [lossArr, next] = opt.step(weights, batch, lr);
		weights = next;
		// step() has already copied the batch to the device, so the spare can be
		// filled while that work is in flight; the loss read below is the sync.
		makeBatch(corpus, cfg, BATCH, rand, spare);
		[batch, spare] = [spare, batch];
		const loss = lossArr.item();
		step++;
		ema = Number.isNaN(ema) ? loss : ema * 0.98 + loss * 0.02;
		window.__state = { step, loss: ema, ready: true };

		if (step % 200 === 0) {
			const now = performance.now();
			const ms = (now - sinceLog) / 200;
			sinceLog = now;
			log(`step ${step} · loss ${ema.toFixed(4)} · ${ms.toFixed(0)} ms/step · lr ${lr.toFixed(6)}`);
			// yield so the driver's evaluate() calls get a turn
			await new Promise((r) => setTimeout(r, 0));
		}
		if (step % 2000 === 0) {
			const frame = await sampler.run({
				params: tree.ref(weights),
				steps: OBJECTIVE === 'flow' ? 24 : 40,
				a: { tags: [], style: null },
				guidanceA: 1,
				seed: 99
			});
			paint(frame, cfg, 8);
			await new Promise((r) => setTimeout(r, 0));
		}
	}
	opt.dispose();
	log('stopped');
}

void main().catch((e) => {
	log(`error: ${e?.stack ?? e}`);
});
