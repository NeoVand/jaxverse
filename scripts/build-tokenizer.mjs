// The vocabulary the scribe reads by default: 300 byte-pair merges over the
// corpus built by build-corpus.mjs — the same 300 merges Plate II's first run
// elects, so "the scribe reads the vocabulary you grew" is literally true for
// a reader who never presses anything.
//
// This mirrors BpeTrainer (src/lib/components/demos/language/bpe-live.ts) merge
// for merge, including the flat counts table, its stride growth and its
// tie-breaking, because the two must agree exactly. src/lib/data/bpe.test.ts
// fails if they ever drift.
//
// Written asset (static/data/):
//   text-merges.json  {baseVocab, merges: [[a,b]…], pieces, vocabSize,
//                      corpusTokens, charsPerToken}
//
// Usage: node scripts/build-tokenizer.mjs [merges]

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = join(ROOT, 'static', 'data');
const MERGES = Number(process.argv[2] ?? 300);

const bytes = new Uint8Array(readFileSync(join(DATA, 'text-tokens.bin')));
const { chars } = JSON.parse(readFileSync(join(DATA, 'text-vocab.json'), 'utf8'));

// ── word-piece boundaries: an optional leading space, then a run of one class ──
const LETTER = /[A-Za-z]/;
const DIGIT = /[0-9]/;
const classOf = (c) => (c === '\n' ? 0 : LETTER.test(c) ? 1 : DIGIT.test(c) ? 2 : 3);
function opensPiece(prev, c) {
	if (prev === undefined) return true;
	if (c === '\n' || prev === '\n') return true;
	if (c === ' ') return true;
	if (prev === ' ') return false;
	return classOf(c) !== classOf(prev);
}

const originalLen = bytes.length;
const seq = Uint16Array.from(bytes);
const bow = new Uint8Array(originalLen);
{
	let prev;
	for (let i = 0; i < originalLen; i++) {
		const c = chars[seq[i]];
		if (opensPiece(prev, c)) bow[i] = 1;
		prev = c;
	}
}

const base = chars.length;
const table = [...chars];
const merges = [];
let len = originalLen;
let S = 256;
let counts = new Int32Array(S * S);
let ready = false;

for (let m = 0; m < MERGES; m++) {
	const newId = base + m;
	if (newId >= S) {
		S *= 2;
		counts = new Int32Array(S * S);
		ready = false;
	}
	if (!ready) {
		counts.fill(0);
		for (let i = 0; i < len - 1; i++) if (!bow[i + 1]) counts[seq[i] * S + seq[i + 1]]++;
		ready = true;
	}
	let best = 1; // a pair must repeat to be worth a token
	let bestKey = -1;
	for (let key = 0; key < counts.length; key++)
		if (counts[key] > best) {
			best = counts[key];
			bestKey = key;
		}
	if (bestKey < 0) throw new Error(`no pair repeats at merge ${m}`);
	const a = Math.floor(bestKey / S);
	const b = bestKey % S;

	counts.fill(0);
	let w = 0;
	let prev = -1;
	for (let i = 0; i < len;) {
		const fuse = i + 1 < len && seq[i] === a && seq[i + 1] === b && !bow[i + 1];
		const t = fuse ? newId : seq[i];
		const opens = bow[i];
		i += fuse ? 2 : 1;
		seq[w] = t;
		bow[w] = opens;
		w++;
		if (prev >= 0 && !opens) counts[prev * S + t]++;
		prev = t;
	}
	len = w;
	table[newId] = table[a] + table[b];
	merges.push([a, b]);
}

const out = {
	baseVocab: base,
	merges,
	pieces: table.slice(base),
	vocabSize: table.length,
	corpusTokens: len,
	charsPerToken: +(originalLen / len).toFixed(4)
};
writeFileSync(join(DATA, 'text-merges.json'), JSON.stringify(out));

console.log(
	`${merges.length} merges · vocab ${out.vocabSize} · ` +
		`corpus ${originalLen.toLocaleString()} chars → ${len.toLocaleString()} tokens ` +
		`(${out.charsPerToken} chars/token)`
);
console.log(
	`first 24 pieces: ${out.pieces
		.slice(0, 24)
		.map((p) => JSON.stringify(p))
		.join(' ')}`
);
console.log(
	`last 8 pieces:   ${out.pieces
		.slice(-8)
		.map((p) => JSON.stringify(p))
		.join(' ')}`
);
