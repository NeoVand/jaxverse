// Lab 5 — a tiny GPT learns to write, one character at a time.
//
// The corpus in this zip is the same story stream the book's scribe reads
// (about 1.5 million characters). The model is a 2-layer, 4-head transformer
// — see model.ts, condensed from the book's own — and every few hundred
// steps it writes a specimen so you can watch gibberish become prose.
// WebGPU strongly recommended; wasm works but each step takes seconds.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import {
	initParams,
	forwardLogprobs,
	lossFn,
	makeBatchOH,
	forwardSeq,
	sampleFromRow,
	type ModelConfig
} from './model';

const out = document.getElementById('out') as HTMLPreElement;
const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.slice(-30).join('\n');
};

const BATCH = 8;

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu'))
		log('note: no WebGPU — falling back to wasm; expect seconds per step.');
	defaultDevice(devices.includes('webgpu') ? 'webgpu' : 'wasm');

	// ── the corpus: one byte per character, ids indexing a char vocabulary ────
	const [tokRes, vocabRes] = await Promise.all([
		fetch('/data/text-tokens.bin'),
		fetch('/data/text-vocab.json')
	]);
	const bytes = new Uint8Array(await tokRes.arrayBuffer());
	const { chars } = (await vocabRes.json()) as { chars: string[] };
	const data = Uint16Array.from(bytes);
	log(`corpus: ${data.length.toLocaleString()} characters · vocab ${chars.length}`);

	const cfg: ModelConfig = {
		nLayer: 2,
		nEmbd: 96,
		nHead: 4,
		blockSize: 96,
		vocab: chars.length
	};
	let params: any = initParams(cfg, 7);
	const solver = adam(1.2e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));

	const jitStep = jit((p: any, tok: any, pos: any, tgt: any) =>
		valueAndGrad((pp: any) => lossFn(pp, cfg, tok, pos, tgt))(p)
	);
	// params go in as an ARGUMENT — a jitted closure would bake them in as
	// trace-time constants and the specimens would never improve
	const jitF = jit((p: any, tok: any, pos: any) => forwardLogprobs(p, cfg, tok, pos));
	const jitForward = (tok: any, pos: any) => jitF(tree.ref(params), tok, pos);

	let s = 1234;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);

	function writeSpecimen(n = 160): string {
		const prompt = 'Once upon a time';
		const ids = [...prompt].map((c) => Math.max(0, chars.indexOf(c)));
		for (let i = 0; i < n; i++) {
			const rows = forwardSeq(jitForward, cfg, ids.slice(-cfg.blockSize));
			const at = Math.min(ids.length, cfg.blockSize) - 1;
			const row = rows.subarray(at * cfg.vocab, (at + 1) * cfg.vocab) as Float32Array;
			ids.push(sampleFromRow(row, 0.8, 12, rand));
		}
		return ids.map((id) => chars[id] ?? '').join('');
	}

	log(`model: ${cfg.nLayer} layers · ${cfg.nEmbd} wide · ${cfg.nHead} heads — compiling…`);
	const t0 = performance.now();
	for (let step = 1; step <= 1200; step++) {
		const { tokenOH, posOH, targetOH } = makeBatchOH(cfg, data, rand, BATCH);
		const [lossVal, grads] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		const loss = lossVal.dataSync()[0];
		if (step % 200 === 0 || step === 1) {
			const ms = (performance.now() - t0) / step;
			log('');
			log(
				`── step ${step} · loss ${loss.toFixed(3)} nats (uniform guess would be ${Math.log(cfg.vocab).toFixed(2)}) · ${ms.toFixed(0)} ms/step`
			);
			log(writeSpecimen());
			await new Promise((r) => setTimeout(r));
		}
	}
	log('');
	log('done — raise the step budget, widen the model, or lower the temperature and re-save.');
}

void main().catch((e) => log(`error: ${e?.message ?? e}`));
