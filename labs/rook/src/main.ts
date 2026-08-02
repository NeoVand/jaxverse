// Lab 7 — Rook: legality emerges from next-token prediction.
//
// The token stream in this zip is ~360k tokens of real chess games in UCI
// notation; token 0 is <game>, the fresh-board marker. The model (model.ts,
// the same transformer as Lab 5) is never told the rules. Every hundred
// steps this lab samples a game from the empty board and lets chess.js —
// a referee that KNOWS the rules — count how many plies survive before the
// first illegal move. Watch that number climb. WebGPU strongly recommended.

/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, valueAndGrad, tree } from '@jax-js/jax';
import { adam, applyUpdates } from '@jax-js/optax';
import { Chess } from 'chess.js';
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
	out.textContent = lines.slice(-24).join('\n');
};

const BATCH = 8;
const GAME = 0; // the <game> token

async function main() {
	const devices = await init();
	if (!devices.includes('webgpu'))
		log('note: no WebGPU — falling back to wasm; expect seconds per step.');
	defaultDevice(devices.includes('webgpu') ? 'webgpu' : 'wasm');

	const [tokRes, vocabRes] = await Promise.all([
		fetch('/data/rook-tokens.bin'),
		fetch('/data/rook-vocab.json')
	]);
	const data = new Uint16Array(await tokRes.arrayBuffer());
	const { moves, vocabSize } = (await vocabRes.json()) as { moves: string[]; vocabSize: number };
	log(`stream: ${data.length.toLocaleString()} tokens of real games · vocab ${vocabSize} moves`);

	const cfg: ModelConfig = { nLayer: 2, nEmbd: 64, nHead: 4, blockSize: 64, vocab: vocabSize };
	let params: any = initParams(cfg, 7);
	const solver = adam(1.2e-3, { b1: 0.9, b2: 0.99 });
	let optState = solver.init(tree.ref(params));

	const jitStep = jit((p: any, tok: any, pos: any, tgt: any) =>
		valueAndGrad((pp: any) => lossFn(pp, cfg, tok, pos, tgt))(p)
	);
	const jitF = jit((p: any, tok: any, pos: any) => forwardLogprobs(p, cfg, tok, pos));
	const jitForward = (tok: any, pos: any) => jitF(tree.ref(params), tok, pos);

	let s = 1234;
	const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0), s / 4294967296);

	/** Sample one game from <game>; the referee counts legal plies. */
	function refereedGame(maxPlies = 40): { plies: number; line: string } {
		const chess = new Chess();
		const ids = [GAME];
		let plies = 0;
		const played: string[] = [];
		for (let i = 0; i < maxPlies; i++) {
			const rows = forwardSeq(jitForward, cfg, ids.slice(-cfg.blockSize));
			const at = Math.min(ids.length, cfg.blockSize) - 1;
			const row = rows.subarray(at * cfg.vocab, (at + 1) * cfg.vocab) as Float32Array;
			const id = sampleFromRow(row, 0.7, 8, rand);
			if (id === GAME) break; // the model chose to end the game
			const uci = moves[id - 1];
			try {
				chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
			} catch {
				return { plies, line: `${played.join(' ')} ✗${uci}` };
			}
			played.push(uci);
			plies++;
			ids.push(id);
			if (chess.isGameOver()) break;
		}
		return { plies, line: played.join(' ') };
	}

	log(`model: ${cfg.nLayer} layers · ${cfg.nEmbd} wide · never told the rules — compiling…`);
	const t0 = performance.now();
	for (let step = 1; step <= 1000; step++) {
		const { tokenOH, posOH, targetOH } = makeBatchOH(cfg, data, rand, BATCH);
		const [lossVal, grads] = jitStep(tree.ref(params), tokenOH, posOH, targetOH);
		const [updates, newState] = solver.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates);
		optState = newState;
		const loss = lossVal.dataSync()[0];
		if (step % 100 === 0 || step === 1) {
			// three refereed games; report the best
			const games = [refereedGame(), refereedGame(), refereedGame()];
			const best = games.reduce((a, b) => (b.plies > a.plies ? b : a));
			const ms = (performance.now() - t0) / step;
			log('');
			log(
				`── step ${String(step).padStart(4)} · loss ${loss.toFixed(3)} · best of 3 games: ` +
					`${best.plies} legal plies · ${ms.toFixed(0)} ms/step`
			);
			log(`   ${best.line || '(no legal move)'}`);
			await new Promise((r) => setTimeout(r));
		}
	}
	log('');
	log('done — an untrained model averages under 1 legal ply; ✗ marks the first illegal move.');
}

void main().catch((e) => log(`error: ${e?.message ?? e}`));
