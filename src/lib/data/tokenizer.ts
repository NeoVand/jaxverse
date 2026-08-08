// The two vocabularies the scribe can read, and the corpus in each of them.
//
// Word pieces are the default: the 300 merges in static/data/text-merges.json
// are exactly the ones the tokenizer plate's first run elects (guarded by bpe.test.ts), so
// the model really is reading the vocabulary the reader grows. Characters stay
// available because the chapter's spelling-acquisition arc needs them — with
// word pieces, most of English's spelling is bought before training starts.

import { base } from '$app/paths';
import { loadCorpus } from './corpus';
import {
	fuseStream,
	makeVocabulary,
	wordStartsOfTokens,
	type MergePair,
	type Vocabulary
} from './bpe';

export type TokenizerKind = 'chars' | 'pieces';

export interface TokenizedCorpus {
	kind: TokenizerKind;
	vocab: Vocabulary;
	/** The corpus in this vocabulary — what the worker samples batches from. */
	tokens: Uint16Array;
	/** Merges applied, 0 for characters. */
	mergeCount: number;
}

interface SnapshotFile {
	baseVocab: number;
	merges: Array<[number, number]>;
	pieces: string[];
	vocabSize: number;
	corpusTokens: number;
	charsPerToken: number;
}

let snapshotCache: Promise<SnapshotFile> | null = null;
let charCache: Promise<TokenizedCorpus> | null = null;
let pieceCache: Promise<TokenizedCorpus> | null = null;

async function loadSnapshot(): Promise<SnapshotFile> {
	snapshotCache ??= (async () => {
		const res = await fetch(`${base}/data/text-merges.json`);
		if (!res.ok) throw new Error(`tokenizer: merges fetch failed (${res.status})`);
		return (await res.json()) as SnapshotFile;
	})();
	return snapshotCache;
}

/** Merge records as the encoders want them, from the flat snapshot pairs. */
export function pairsOf(snapshot: {
	baseVocab: number;
	merges: Array<[number, number]>;
}): MergePair[] {
	return snapshot.merges.map(([a, b], i) => ({ a, b, newId: snapshot.baseVocab + i }));
}

/** One token per character: the chapter's slow, legible baseline. */
export async function charTokenizer(): Promise<TokenizedCorpus> {
	charCache ??= (async () => {
		const corpus = await loadCorpus();
		return {
			kind: 'chars' as const,
			vocab: makeVocabulary(corpus.vocab, [], 1),
			tokens: corpus.tokens,
			mergeCount: 0
		};
	})();
	return charCache;
}

/** The shipped word-piece vocabulary, with the corpus retokenized to match.
 * Applying 300 merges to 1.5M tokens is ~0.5 s of arithmetic, so it runs in
 * slices with the event loop free in between — the page must stay alive. */
export async function pieceTokenizer(
	onProgress?: (done: number, total: number) => void
): Promise<TokenizedCorpus> {
	pieceCache ??= (async () => {
		const [corpus, snapshot] = await Promise.all([loadCorpus(), loadSnapshot()]);
		const pairs = pairsOf(snapshot);
		let stream = {
			tokens: corpus.tokens,
			starts: wordStartsOfTokens(corpus.tokens, corpus.vocab)
		};
		const SLICE = 25;
		for (let i = 0; i < pairs.length; i += SLICE) {
			stream = fuseStream(stream, pairs.slice(i, i + SLICE));
			onProgress?.(Math.min(i + SLICE, pairs.length), pairs.length);
			await new Promise((r) => setTimeout(r, 0));
		}
		return {
			kind: 'pieces' as const,
			vocab: makeVocabulary(corpus.vocab, pairs, corpus.tokens.length / stream.tokens.length),
			tokens: stream.tokens,
			mergeCount: pairs.length
		};
	})();
	return pieceCache;
}

/** A vocabulary the reader grew past the snapshot, from the tokenizer plate's trainer. */
export function adoptedTokenizer(
	chars: readonly string[],
	pairs: readonly MergePair[],
	tokens: Uint16Array,
	originalLen: number
): TokenizedCorpus {
	return {
		kind: 'pieces',
		vocab: makeVocabulary(chars, pairs, originalLen / tokens.length),
		tokens,
		mergeCount: pairs.length
	};
}

/** How many merges the shipped snapshot carries, without fetching the corpus. */
export async function snapshotMergeCount(): Promise<number> {
	return (await loadSnapshot()).merges.length;
}
