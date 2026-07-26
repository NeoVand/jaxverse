// Live byte-pair encoding over the real corpus — the algorithm, not a replay.
// One merge = one pass over the token stream: the pass that rewrites the
// sequence for merge k simultaneously counts the pairs of the NEW sequence, so
// merge k+1 needs no extra scan (a full separate pass measured ~8 ms; fusing
// keeps a merge comfortably under one frame of work).
//
// Merges never straddle two words (see $lib/data/bpe): a parallel flag array
// marks where each word piece begins, and a pair whose right half opens a word
// is never counted. Without it this elects "ce upon a tim" — tokens no reader
// or attention row can make sense of.
//
// Pair counts live in a flat Int32Array indexed by a·S+b, where S is a power
// of two grown as the vocabulary grows (S=256 → 256 KB of counts; S=2048 →
// 16 MB, the hard ceiling at 1,979 merges). No Map, no hashing; the argmax is
// a linear scan of the table. Dependency-free on purpose: this exact file runs
// under node for the timing bench.

import {
	fuseIds,
	makeVocabulary,
	wordStarts,
	wordStartsOfTokens,
	type MergePair,
	type Vocabulary
} from '$lib/data/bpe';

/** Merges per run: the default first run, and each "keep merging" increment. */
export const RUN_MERGES = 300;
/** Id-space ceiling (the counts table caps at 2048² ints ≈ 16 MB). */
export const HARD_MAX_MERGES = 2048 - 69;

export interface MergeRecord {
	/** 0-based merge number; newId = baseVocab + index. */
	index: number;
	a: number;
	b: number;
	newId: number;
	/** Adjacent-pair frequency at the moment of the vote. */
	count: number;
	aText: string;
	bText: string;
	text: string;
	/** Corpus length in tokens AFTER applying this merge. */
	corpusLen: number;
	/** Wall-clock ms this merge took to count + rewrite — the liveness receipt. */
	ms: number;
}

export class BpeTrainer {
	readonly baseVocab: number;
	readonly originalLen: number;
	/** id → decoded text, extended as merges land. */
	readonly table: string[];
	readonly merges: MergeRecord[] = [];
	readonly chars: readonly string[];

	private seq: Uint16Array;
	/** 1 where the token at this position opens a word piece. */
	private bow: Uint8Array;
	private len: number;
	private S = 256; // current key stride; counts is S×S
	private counts = new Int32Array(256 * 256);
	private countsReady = false;
	private idOf = new Map<string, number>();

	constructor(tokens: Uint16Array, vocab: string[]) {
		this.seq = tokens.slice();
		this.bow = wordStartsOfTokens(tokens, vocab);
		this.len = tokens.length;
		this.originalLen = tokens.length;
		this.baseVocab = vocab.length;
		this.chars = vocab;
		this.table = [...vocab];
		vocab.forEach((c, i) => this.idOf.set(c, i));
	}

	/** The corpus as tokenized right now — what the scribe trains on when the
	 * reader hands its vocabulary over. */
	tokenizedCorpus(): Uint16Array {
		return this.seq.slice(0, this.len);
	}

	/** The merge list as plain pairs, for encoders and for the snapshot. */
	pairs(): MergePair[] {
		return this.merges.map((m) => ({ a: m.a, b: m.b, newId: m.newId }));
	}

	vocabulary(): Vocabulary {
		return makeVocabulary(this.chars, this.pairs(), this.originalLen / this.len);
	}

	/** Corpus length in tokens once the first k merges are applied. */
	corpusLenAt(k: number): number {
		if (k <= 0) return this.originalLen;
		const m = this.merges[Math.min(k, this.merges.length) - 1];
		return m ? m.corpusLen : this.originalLen;
	}

	decode(id: number): string {
		return this.table[id] ?? '';
	}

	/** One real merge: vote (argmax over pair counts), fuse, rewrite + recount.
	 * Returns null when the id space is spent or no pair repeats. */
	step(): MergeRecord | null {
		if (this.merges.length >= HARD_MAX_MERGES) return null;
		const t0 = performance.now();
		const newId = this.baseVocab + this.merges.length;
		if (newId >= this.S) {
			// grow the key space; old keys no longer decode under the new stride
			this.S *= 2;
			this.counts = new Int32Array(this.S * this.S);
			this.countsReady = false;
		}
		const { seq, bow, counts, S } = this;

		if (!this.countsReady) {
			counts.fill(0);
			// a pair whose right half opens a word would fuse across a boundary
			for (let i = 0; i < this.len - 1; i++) if (!bow[i + 1]) counts[seq[i] * S + seq[i + 1]]++;
			this.countsReady = true;
		}

		let best = 1; // a pair must appear at least twice to be worth a token
		let bestKey = -1;
		for (let key = 0; key < counts.length; key++) {
			if (counts[key] > best) {
				best = counts[key];
				bestKey = key;
			}
		}
		if (bestKey < 0) return null;

		const a = Math.floor(bestKey / S);
		const b = bestKey % S;

		// greedy left-to-right collapse; count the output's pairs as it is emitted.
		// w never overtakes i, so writing the flags in place is safe.
		counts.fill(0);
		let w = 0;
		let prev = -1;
		for (let i = 0; i < this.len;) {
			const fuse = i + 1 < this.len && seq[i] === a && seq[i + 1] === b && !bow[i + 1];
			const t = fuse ? newId : seq[i];
			const opens = bow[i];
			i += fuse ? 2 : 1;
			seq[w] = t;
			bow[w] = opens;
			w++;
			if (prev >= 0 && !opens) counts[prev * S + t]++;
			prev = t;
		}
		this.len = w;

		this.table[newId] = this.table[a] + this.table[b];
		const rec: MergeRecord = {
			index: this.merges.length,
			a,
			b,
			newId,
			count: best,
			aText: this.table[a],
			bText: this.table[b],
			text: this.table[newId],
			corpusLen: w,
			ms: performance.now() - t0
		};
		this.merges.push(rec);
		return rec;
	}

	/** Tokenize a short string using only the first k merges — replaying the
	 * merge list over ~60 characters is trivial, so scrubbing back is instant. */
	encodeAt(text: string, k: number): number[] {
		const kept: string[] = [];
		for (const c of text) if (this.idOf.has(c)) kept.push(c); // strays drop
		const ids = kept.map((c) => this.idOf.get(c)!);
		const upto = Math.min(k, this.merges.length);
		return fuseIds(ids, wordStarts(kept), this.pairs().slice(0, upto));
	}
}

/** Drive merges up to `maxMerges`, one per beat. A merge computes in ~1.4 ms —
 * faster than the eye — so `paceMs` spaces the beats out for reading. The
 * merge itself is still computed live on its turn; only the wait between
 * merges is a display choice. Re-entrant: call again with the same trainer
 * and a higher target to keep merging where it left off. */
export async function trainBpe(
	trainer: BpeTrainer,
	opts: { maxMerges?: number; paceMs?: number | (() => number); shouldStop?: () => boolean },
	onMerge: (m: MergeRecord) => void
): Promise<'done' | 'stopped' | 'exhausted'> {
	const max = Math.min(opts.maxMerges ?? RUN_MERGES, HARD_MAX_MERGES);
	const pace = () => (typeof opts.paceMs === 'function' ? opts.paceMs() : (opts.paceMs ?? 0));
	while (trainer.merges.length < max) {
		if (opts.shouldStop?.()) return 'stopped';
		await new Promise((r) => setTimeout(r, pace()));
		if (opts.shouldStop?.()) return 'stopped';
		const m = trainer.step();
		if (!m) return 'exhausted';
		onMerge(m);
	}
	return 'done';
}
