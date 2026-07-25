// Rook's chess data, shipped by scripts/copy-rook-data.mjs: the random-legal
// pretraining token stream, the UCI move vocabulary (id 0 = <game>), and the
// time-machine weight snapshots — int8 blobs dequantized here per leaf with
// the scales the manifest carries for each waypoint.

import { base } from '$app/paths';

export interface RookVocab {
	/** Distinct UCI strings; token id = index + 1 (id 0 is <game>). */
	moves: string[];
	vocabSize: number;
}

export interface RookData {
	/** <game> m1 m2 … <game> …, one token per ply. */
	tokens: Uint16Array;
	vocab: RookVocab;
	decode(id: number): string;
	/** UCI (and '<game>') → token id. */
	idOf: Map<string, number>;
}

export interface RookWaypoint {
	step: number;
	trainLoss: number;
	valLoss: number;
	/** Per-leaf int8 dequant scales, aligned with manifest.leafSizes. */
	scales: number[];
	bytes: number;
	legalRate: number;
	/** Snapshot basename under /data/timemachine/. */
	file: string;
	scenes?: unknown;
	game?: string[];
	illegalAt?: number | null;
}

export interface RookManifest {
	bird: string;
	config: { nLayer: number; nEmbd: number; nHead: number; blockSize: number; vocab: number };
	params: number;
	/** Flat f32 parameter count per pytree leaf, in serialization order. */
	leafSizes: number[];
	quant: string;
	steps: number;
	waypoints: RookWaypoint[];
}

const GAME = '<game>';

let rookCache: Promise<RookData> | null = null;

/** Fetch + decode once; every demo shares the same in-flight promise. */
export async function loadRook(): Promise<RookData> {
	rookCache ??= (async () => {
		const [tokRes, vocabRes] = await Promise.all([
			fetch(`${base}/data/rook-tokens.bin`),
			fetch(`${base}/data/rook-vocab.json`)
		]);
		if (!tokRes.ok) throw new Error(`rook: tokens fetch failed (${tokRes.status})`);
		if (!vocabRes.ok) throw new Error(`rook: vocab fetch failed (${vocabRes.status})`);
		const tokens = new Uint16Array(await tokRes.arrayBuffer());
		const vocab = (await vocabRes.json()) as RookVocab;
		const idOf = new Map<string, number>([[GAME, 0]]);
		vocab.moves.forEach((m, i) => idOf.set(m, i + 1));
		return {
			tokens,
			vocab,
			decode: (id) => (id === 0 ? GAME : (vocab.moves[id - 1] ?? '?')),
			idOf
		};
	})();
	return rookCache;
}

let manifestCache: Promise<RookManifest> | null = null;

export async function loadRookManifest(): Promise<RookManifest> {
	manifestCache ??= (async () => {
		const res = await fetch(`${base}/data/timemachine/rook-manifest.json`);
		if (!res.ok) throw new Error(`rook: manifest fetch failed (${res.status})`);
		return (await res.json()) as RookManifest;
	})();
	return manifestCache;
}

const waypointCache = new Map<string, Promise<Float32Array>>();

/**
 * Fetch one weight snapshot and dequantize: symmetric per-tensor int8, so
 * f32[i] = i8[i] · scales[leaf], with leaf boundaries from manifest.leafSizes.
 */
export async function loadRookWaypoint(
	file: string,
	manifest: RookManifest
): Promise<Float32Array> {
	let pending = waypointCache.get(file);
	if (!pending) {
		pending = (async () => {
			const wp = manifest.waypoints.find((w) => w.file === file);
			if (!wp) throw new Error(`rook: no waypoint named ${file} in manifest`);
			const res = await fetch(`${base}/data/timemachine/${file}`);
			if (!res.ok) throw new Error(`rook: weights fetch failed (${res.status})`);
			const q = new Int8Array(await res.arrayBuffer());
			const flat = new Float32Array(q.length);
			let off = 0;
			for (let li = 0; li < manifest.leafSizes.length; li++) {
				const s = wp.scales[li];
				const sz = manifest.leafSizes[li];
				for (let k = 0; k < sz; k++) flat[off + k] = q[off + k] * s;
				off += sz;
			}
			if (off !== q.length) throw new Error(`rook: leafSizes sum ${off} ≠ blob ${q.length}`);
			return flat;
		})();
		waypointCache.set(file, pending);
	}
	return pending;
}
