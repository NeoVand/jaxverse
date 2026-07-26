import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { BpeTrainer } from '$lib/components/demos/language/bpe-live';
import { makeVocabulary, wordStarts } from './bpe';

/** The shipped snapshot and the live trainer must elect the same vocabulary:
 * the scribe's default vocabulary is sold to the reader as the one Plate II
 * grows, so any drift between scripts/build-tokenizer.mjs and BpeTrainer is a
 * lie in the prose, not just a mismatch. */

const bytes = new Uint8Array(readFileSync('static/data/text-tokens.bin'));
const { chars } = JSON.parse(readFileSync('static/data/text-vocab.json', 'utf8')) as {
	chars: string[];
};
const snapshot = JSON.parse(readFileSync('static/data/text-merges.json', 'utf8')) as {
	baseVocab: number;
	merges: Array<[number, number]>;
	pieces: string[];
	vocabSize: number;
	corpusTokens: number;
	charsPerToken: number;
};

describe('the shipped vocabulary', () => {
	const trainer = new BpeTrainer(Uint16Array.from(bytes), chars);
	for (let i = 0; i < snapshot.merges.length; i++) trainer.step();

	it('is the one the live trainer elects, merge for merge', () => {
		expect(trainer.merges.map((m) => [m.a, m.b])).toEqual(snapshot.merges);
		expect(trainer.table.slice(snapshot.baseVocab)).toEqual(snapshot.pieces);
	});

	it('tokenizes the corpus to the length the snapshot claims', () => {
		expect(trainer.tokenizedCorpus().length).toBe(snapshot.corpusTokens);
		expect(bytes.length / snapshot.corpusTokens).toBeCloseTo(snapshot.charsPerToken, 3);
	});

	it('never elects a piece that straddles two words', () => {
		for (const piece of snapshot.pieces) {
			// a piece may open with a space, but may not contain a further
			// boundary — " the" is legal, "e t" and "ce upon" are not
			const inner = wordStarts([...piece]).slice(1);
			expect(
				inner.some((flag) => flag === 1),
				`"${piece}" straddles a boundary`
			).toBe(false);
		}
	});
});

describe('encoding with the shipped merges', () => {
	const pairs = snapshot.merges.map(([a, b], i) => ({
		a,
		b,
		newId: snapshot.baseVocab + i
	}));
	const vocab = makeVocabulary(chars, pairs, snapshot.charsPerToken);

	it('round-trips text that is inside the alphabet', () => {
		const text = 'Once upon a time there was a tiny cat who wanted to fly.';
		expect(vocab.decode(vocab.encode(text))).toBe(text);
	});

	it('carries whole words where the corpus voted for them', () => {
		const pieces = vocab.encode('the cat was little').map((id) => vocab.table[id]);
		expect(pieces).toContain(' was');
		expect(pieces).toContain(' little');
	});

	it('drops characters outside the alphabet, keeping what surrounds them', () => {
		expect(vocab.decode(vocab.encode('caf\u00e9 \u2014 ok'))).toBe('caf  ok');
	});
});
