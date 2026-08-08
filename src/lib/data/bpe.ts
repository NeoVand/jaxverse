// Byte-pair encoding shared by the live trainer (the tokenizer plate), the shipped
// snapshot the scribe reads by default, and the encoders both hand the model.
//
// The one rule that makes the vocabulary readable: a merge may never straddle
// two words. GPT-2 enforces this with a regex pre-tokenizer; we do the same
// thing with a flag per position, which costs one byte per token and keeps the
// trainer's honest "count every pair in the whole corpus" scan intact.
//
// Without the rule, greedy merging over a flat character stream elects tokens
// like "ce upon a tim" — and, past a thousand merges on a small corpus, whole
// sentences. Those would make the model's attention rows unreadable, which is
// the opposite of what the plates are for.

/** A word piece is an optional leading space, then a run of one class. */
const enum Cls {
	Newline,
	Letter,
	Digit,
	Other
}

function classOf(c: string): Cls {
	if (c === '\n') return Cls.Newline;
	if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Cls.Letter;
	if (c >= '0' && c <= '9') return Cls.Digit;
	return Cls.Other;
}

/** Does `c` open a new word piece, given the character before it? */
function opensPiece(prev: string | undefined, c: string): boolean {
	if (prev === undefined) return true;
	if (c === '\n' || prev === '\n') return true;
	if (c === ' ') return true;
	if (prev === ' ') return false; // the space opened this piece; c continues it
	return classOf(c) !== classOf(prev);
}

/** One flag per character: 1 where a word piece begins. */
export function wordStarts(chars: readonly string[]): Uint8Array {
	const out = new Uint8Array(chars.length);
	for (let i = 0; i < chars.length; i++) if (opensPiece(chars[i - 1], chars[i])) out[i] = 1;
	return out;
}

/** The same, straight off a char-token stream — the corpus arrives this way. */
export function wordStartsOfTokens(
	tokens: ArrayLike<number>,
	vocab: readonly string[]
): Uint8Array {
	const out = new Uint8Array(tokens.length);
	let prev: string | undefined;
	for (let i = 0; i < tokens.length; i++) {
		const c = vocab[tokens[i]] ?? '';
		if (opensPiece(prev, c)) out[i] = 1;
		prev = c;
	}
	return out;
}

export interface MergePair {
	a: number;
	b: number;
	newId: number;
}

/** Apply merges in order to one short id sequence, respecting boundaries.
 * `starts` is the flag array for `ids` and is read, not written. */
export function fuseIds(ids: number[], starts: Uint8Array, merges: readonly MergePair[]): number[] {
	let cur = ids;
	let flags = starts;
	for (const { a, b, newId } of merges) {
		const out: number[] = [];
		const outFlags = new Uint8Array(cur.length);
		for (let i = 0; i < cur.length;) {
			const fuse = i + 1 < cur.length && cur[i] === a && cur[i + 1] === b && !flags[i + 1];
			outFlags[out.length] = flags[i];
			out.push(fuse ? newId : cur[i]);
			i += fuse ? 2 : 1;
		}
		cur = out;
		flags = outFlags;
	}
	return cur;
}

/** A token stream and the flags that say where its words begin. */
export interface Stream {
	tokens: Uint16Array;
	starts: Uint8Array;
}

/** Apply merges to a whole corpus: one pass per merge over a 1.5M-token stream
 * costs ~1.5 ms, so a few hundred merges add up to about half a second. Feed
 * the result back in to continue with more merges. */
export function fuseStream(stream: Stream, merges: readonly MergePair[]): Stream {
	const seq = stream.tokens.slice();
	const bow = stream.starts.slice();
	let len = seq.length;
	for (const { a, b, newId } of merges) {
		let w = 0;
		for (let i = 0; i < len;) {
			const fuse = i + 1 < len && seq[i] === a && seq[i + 1] === b && !bow[i + 1];
			bow[w] = bow[i];
			seq[w] = fuse ? newId : seq[i];
			w++;
			i += fuse ? 2 : 1;
		}
		len = w;
	}
	return { tokens: seq.slice(0, len), starts: bow.slice(0, len) };
}

/** What a model needs to read and write text: a table, and the two maps. */
export interface Vocabulary {
	/** id → the text it stands for. Base ids are single characters. */
	table: string[];
	size: number;
	/** How many characters one token carries, on this corpus — the exchange
	 * rate that makes losses in nats/token comparable to nats/char. */
	charsPerToken: number;
	encode(text: string): number[];
	decode(ids: readonly number[]): string;
}

/** Build a vocabulary from the base characters plus a list of merges. */
export function makeVocabulary(
	chars: readonly string[],
	merges: readonly MergePair[],
	charsPerToken: number
): Vocabulary {
	const table = [...chars];
	for (const m of merges) table[m.newId] = table[m.a] + table[m.b];
	const idOf = new Map(chars.map((c, i) => [c, i]));
	return {
		table,
		size: table.length,
		charsPerToken,
		encode(text: string): number[] {
			const kept: string[] = [];
			for (const c of text) if (idOf.has(c)) kept.push(c);
			const ids = kept.map((c) => idOf.get(c)!);
			return merges.length === 0 ? ids : fuseIds(ids, wordStarts(kept), merges);
		},
		decode(ids: readonly number[]): string {
			let out = '';
			for (const id of ids) out += table[id] ?? '';
			return out;
		}
	};
}
