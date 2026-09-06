// Loading the emoji corpus: eight spritesheets and a metadata file, decoded
// once and shared by every plate on the page.
//
// The sheets are PNG because PNG is a decompressor every browser already
// ships. `scripts/build-emoji.mjs` writes them; nothing here downloads from
// the internet.

import type { Corpus } from './runtime';

export interface EmojiSet {
	id: string;
	label: string;
	credit: string;
	license: string;
	url: string;
}

export interface EmojiEntry {
	/** The character itself, for labels and for the reader to recognize. */
	cp: string;
	name: string;
	group: string;
	tags: number[];
}

export interface EmojiMeta {
	tile: number;
	cols: number;
	rows: number;
	count: number;
	sets: EmojiSet[];
	tags: string[];
	emoji: EmojiEntry[];
}

export interface EmojiCorpus extends Corpus {
	meta: EmojiMeta;
	/** Tag string to its index, for turning a typed prompt into a condition. */
	tagIndex: Map<string, number>;
}

const CHANNELS = 4;

async function rasterize(url: string, w: number, h: number): Promise<Uint8ClampedArray> {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${res.status} loading ${url}`);
	const bitmap = await createImageBitmap(await res.blob());
	const canvas =
		typeof OffscreenCanvas !== 'undefined'
			? new OffscreenCanvas(w, h)
			: Object.assign(document.createElement('canvas'), { width: w, height: h });
	const ctx = canvas.getContext('2d', {
		willReadFrequently: true
	}) as CanvasRenderingContext2D | null;
	if (!ctx) throw new Error('no 2d context for the emoji sheets');
	ctx.clearRect(0, 0, w, h);
	ctx.drawImage(bitmap, 0, 0);
	bitmap.close();
	return ctx.getImageData(0, 0, w, h).data;
}

let pending: Promise<EmojiCorpus> | null = null;

/** Fetch and decode every sheet. Cached — several plates share one download. */
export function loadEmoji(base = ''): Promise<EmojiCorpus> {
	pending ??= (async () => {
		const meta: EmojiMeta = await fetch(`${base}/data/emoji-meta.json`).then((r) => r.json());
		const { tile, cols, count, sets } = meta;
		const dim = CHANNELS * tile * tile;
		const images = new Uint8Array(sets.length * count * dim);

		await Promise.all(
			sets.map(async (set, s) => {
				const px = await rasterize(
					`${base}/data/emoji-${set.id}.png`,
					cols * tile,
					meta.rows * tile
				);
				for (let i = 0; i < count; i++) {
					const tx = (i % cols) * tile;
					const ty = Math.floor(i / cols) * tile;
					// planar CHW, because the network wants channels on axis 1
					const dst = (s * count + i) * dim;
					for (let y = 0; y < tile; y++) {
						for (let x = 0; x < tile; x++) {
							const o = ((ty + y) * cols * tile + tx + x) * 4;
							const p = y * tile + x;
							images[dst + p] = px[o];
							images[dst + tile * tile + p] = px[o + 1];
							images[dst + 2 * tile * tile + p] = px[o + 2];
							images[dst + 3 * tile * tile + p] = px[o + 3];
						}
					}
				}
			})
		);

		return {
			meta,
			images,
			tags: meta.emoji.map((e) => e.tags),
			count,
			styles: sets.length,
			tagIndex: new Map(meta.tags.map((t, i) => [t, i]))
		};
	})();
	return pending;
}

/** Undo the plural folding the corpus builder applied, the same way it did. */
function singular(w: string): string {
	if (/(ss|us|is|as)$/.test(w) || w.length < 5) return w;
	if (/ies$/.test(w)) return `${w.slice(0, -3)}y`;
	if (/(ch|sh|s|x|z)es$/.test(w)) return w.slice(0, -2);
	if (/ves$/.test(w)) return `${w.slice(0, -3)}f`;
	if (/s$/.test(w)) return w.slice(0, -1);
	return w;
}

export interface TagVocab {
	tags: string[];
	index: Map<string, number>;
}

export function makeVocab(tags: string[]): TagVocab {
	return { tags, index: new Map(tags.map((t, i) => [t, i])) };
}

export interface ParsedPrompt {
	tags: number[];
	matched: string[];
	ignored: string[];
}

/**
 * Turn what the reader typed into tag indices.
 *
 * Words the vocabulary has never seen are reported rather than dropped in
 * silence — a prompt that quietly did nothing is the most confusing thing a
 * demo like this can do to someone.
 */
export function parsePrompt(text: string, vocab: TagVocab): ParsedPrompt {
	const tags: number[] = [];
	const matched: string[] = [];
	const ignored: string[] = [];
	const seen = new Set<number>();
	for (const raw of text.toLowerCase().split(/[^a-z]+/)) {
		if (raw.length < 2) continue;
		const idx = vocab.index.get(raw) ?? vocab.index.get(singular(raw));
		if (idx === undefined) {
			ignored.push(raw);
		} else if (!seen.has(idx)) {
			seen.add(idx);
			tags.push(idx);
			matched.push(vocab.tags[idx]);
		}
	}
	return { tags, matched, ignored };
}
