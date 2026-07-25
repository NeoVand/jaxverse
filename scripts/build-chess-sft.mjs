// SFT corpus for Rook: games where both sides play greedy material — take
// the most valuable capture on the board, otherwise give check half the
// time, otherwise wander. Same token language as the random-legal
// pretraining stream (build-rook-data.mjs in llmvibes), so fine-tuning on
// this shifts the model's style without touching the rules it already
// knows. Encoded against the SHIPPED vocab (run copy-rook-data.mjs first);
// any game that uses a move the vocab has never seen is dropped whole.
//
// Written assets (static/data/):
//   rook-sft-tokens.bin  Uint16 stream: <game> m1 m2 … <game> …
//
// Usage: node scripts/build-chess-sft.mjs [games=2500] [maxPlies=60] [seed=11]

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Chess } from 'chess.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const N_GAMES = Number(process.argv[2] ?? 2500);
const MAX_PLIES = Number(process.argv[3] ?? 60);
let seed = Number(process.argv[4] ?? 11);

// mulberry32
function rand() {
	seed |= 0;
	seed = (seed + 0x6d2b79f5) | 0;
	let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
	t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const VALUE = { q: 9, r: 5, b: 3, n: 3, p: 1 };
const pickFrom = (arr) => arr[Math.floor(rand() * arr.length)];

const outDir = join(ROOT, 'static', 'data');
const vocab = JSON.parse(readFileSync(join(outDir, 'rook-vocab.json'), 'utf8'));
const idOf = new Map(vocab.moves.map((m, i) => [m, i + 1])); // 0 = <game>

const stream = [];
let kept = 0;
let dropped = 0;
let plies = 0;
let captures = 0;
let mates = 0;
for (let g = 0; g < N_GAMES; g++) {
	const chess = new Chess();
	const game = [];
	let gameCaptures = 0;
	for (let ply = 0; ply < MAX_PLIES; ply++) {
		const legal = chess.moves({ verbose: true });
		if (legal.length === 0) break;
		const caps = legal.filter((m) => m.captured);
		let move;
		if (caps.length > 0) {
			const best = Math.max(...caps.map((m) => VALUE[m.captured]));
			move = pickFrom(caps.filter((m) => VALUE[m.captured] === best));
			gameCaptures++;
		} else {
			const checks = legal.filter((m) => m.san.includes('+') || m.san.includes('#'));
			move = checks.length > 0 && rand() < 0.5 ? pickFrom(checks) : pickFrom(legal);
		}
		game.push(move.from + move.to + (move.promotion ?? ''));
		chess.move(move);
	}
	if (game.some((uci) => !idOf.has(uci))) {
		dropped++;
		continue;
	}
	stream.push(0);
	for (const uci of game) stream.push(idOf.get(uci));
	kept++;
	plies += game.length;
	captures += gameCaptures;
	if (chess.isCheckmate()) mates++;
	if ((g + 1) % 500 === 0) console.log(`generated ${g + 1}/${N_GAMES} games`);
}

writeFileSync(join(outDir, 'rook-sft-tokens.bin'), Buffer.from(new Uint16Array(stream).buffer));
console.log(
	`kept ${kept}, dropped ${dropped} (out-of-vocab), tokens: ${stream.length.toLocaleString()}`
);
console.log(
	`avg plies: ${(plies / kept).toFixed(1)}, capture rate: ${((captures / plies) * 100).toFixed(1)}%/move, ` +
		`checkmate rate: ${((mates / kept) * 100).toFixed(1)}%`
);
