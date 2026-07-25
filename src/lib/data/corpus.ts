// The char-level story corpus packed by scripts/build-corpus.mjs: one byte
// per character on the wire, widened to Uint16Array here because the LM
// stack consumes u16 token streams.

import { base } from '$app/paths';

export interface Corpus {
	/** One token per character of the corpus. */
	tokens: Uint16Array;
	/** Token id → character, sorted; id = index. */
	vocab: string[];
	decode(ids: number[]): string;
	/** Characters outside the vocabulary are skipped. */
	encode(s: string): number[];
}

interface CorpusVocabFile {
	chars: string[];
	vocabSize: number;
}

let cache: Promise<Corpus> | null = null;

/** Fetch + decode once; every demo shares the same in-flight promise. */
export async function loadCorpus(): Promise<Corpus> {
	cache ??= (async () => {
		const [tokRes, vocabRes] = await Promise.all([
			fetch(`${base}/data/text-tokens.bin`),
			fetch(`${base}/data/text-vocab.json`)
		]);
		if (!tokRes.ok) throw new Error(`corpus: tokens fetch failed (${tokRes.status})`);
		if (!vocabRes.ok) throw new Error(`corpus: vocab fetch failed (${vocabRes.status})`);
		const bytes = new Uint8Array(await tokRes.arrayBuffer());
		const { chars } = (await vocabRes.json()) as CorpusVocabFile;
		const idOf = new Map(chars.map((c, i) => [c, i]));
		return {
			tokens: Uint16Array.from(bytes),
			vocab: chars,
			decode: (ids) => ids.map((id) => chars[id] ?? '').join(''),
			encode: (s) => [...s].map((c) => idOf.get(c)).filter((id): id is number => id !== undefined)
		};
	})();
	return cache;
}
