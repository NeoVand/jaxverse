// Live skip-gram word embeddings with negative sampling — plain TS on the
// main thread, trained on the same story corpus the scribe reads. Small on
// purpose: top-220 words, 16 dimensions, sigmoid dot products — the whole
// learning rule is ~40 lines, and a browser does ~1M pairs/second of it.
// Dependency-free so the node bench can run this exact file.

export interface WordCorpus {
	/** Top-V words by frequency, id = index. */
	vocab: string[];
	/** Corpus frequency per vocab word. */
	counts: number[];
	/** The corpus as vocab ids, out-of-vocab words dropped. */
	seq: Int32Array;
	/** Word count before the vocabulary filter. */
	totalWords: number;
}

/** Lowercase, strip everything but letters, keep the top `vocabSize` words.
 * Single letters are dropped except the real words "a" and "i" (contractions
 * like "don't" would otherwise shed bare "t"s into the vocabulary). */
export function buildWordCorpus(text: string, vocabSize = 220): WordCorpus {
	const words = (text.toLowerCase().match(/[a-z]+/g) ?? []).filter(
		(w) => w.length > 1 || w === 'a' || w === 'i'
	);
	const freq = new Map<string, number>();
	for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
	const top = [...freq.entries()].sort((x, y) => y[1] - x[1]).slice(0, vocabSize);
	const idOf = new Map(top.map(([w], i) => [w, i]));
	const ids: number[] = [];
	for (const w of words) {
		const id = idOf.get(w);
		if (id !== undefined) ids.push(id);
	}
	return {
		vocab: top.map(([w]) => w),
		counts: top.map(([, c]) => c),
		seq: Int32Array.from(ids),
		totalWords: words.length
	};
}

function mulberry32(seed: number) {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export interface SkipGramOpts {
	dim?: number;
	seed?: number;
	/** Max context offset; each pair draws its window uniformly from 1..window. */
	window?: number;
	negatives?: number;
	lrStart?: number;
	lrEnd?: number;
	/** Pairs over which the learning rate decays and training counts as done. */
	budget?: number;
}

export class SkipGram {
	readonly dim: number;
	readonly V: number;
	readonly budget: number;
	/** Input vectors [V×dim] — the embeddings everything downstream reads. */
	readonly wIn: Float32Array;
	private wOut: Float32Array;
	private corpus: WordCorpus;
	private negTable: Int32Array;
	private rng: () => number;
	private window: number;
	private negatives: number;
	private lrStart: number;
	private lrEnd: number;
	pairsSeen = 0;
	/** EMA of the per-pair NCE loss — the demo's live readout. */
	lossEma = NaN;

	constructor(corpus: WordCorpus, opts: SkipGramOpts = {}) {
		this.corpus = corpus;
		this.dim = opts.dim ?? 16;
		this.V = corpus.vocab.length;
		this.window = opts.window ?? 3;
		this.negatives = opts.negatives ?? 5;
		this.lrStart = opts.lrStart ?? 0.05;
		this.lrEnd = opts.lrEnd ?? 0.01;
		this.budget = opts.budget ?? 1_200_000;
		this.rng = mulberry32(opts.seed ?? 7);
		this.wIn = new Float32Array(this.V * this.dim);
		this.wOut = new Float32Array(this.V * this.dim);
		for (let i = 0; i < this.wIn.length; i++) this.wIn[i] = (this.rng() - 0.5) / this.dim;
		// negative-sampling table: unigram^0.75, the word2vec standard
		const T = 100_000;
		this.negTable = new Int32Array(T);
		const pow = corpus.counts.map((c) => Math.pow(c, 0.75));
		const total = pow.reduce((a, b) => a + b, 0);
		let w = 0;
		let acc = pow[0] / total;
		for (let i = 0; i < T; i++) {
			this.negTable[i] = w;
			if (i / T > acc && w < this.V - 1) {
				w++;
				acc += pow[w] / total;
			}
		}
	}

	/** Run n training pairs synchronously (one pair ≈ 200 flops at dim 16). */
	trainPairs(n: number): void {
		const { wIn, wOut, dim, negatives } = this;
		const seq = this.corpus.seq;
		const len = seq.length;
		const dv = new Float32Array(dim);
		for (let p = 0; p < n; p++) {
			const t = Math.min(1, this.pairsSeen / this.budget);
			const lr = this.lrStart + (this.lrEnd - this.lrStart) * t;
			const i = 1 + Math.floor(this.rng() * (len - 2));
			const win = 1 + Math.floor(this.rng() * this.window);
			let off = 1 + Math.floor(this.rng() * win);
			if (this.rng() < 0.5) off = -off;
			const j = Math.min(len - 1, Math.max(0, i + off));
			const c = seq[i]; // center word
			const target = seq[j]; // its real neighbor
			const cBase = c * dim;
			dv.fill(0);
			let loss = 0;
			for (let k = 0; k <= negatives; k++) {
				const positive = k === 0;
				const o = positive ? target : this.negTable[Math.floor(this.rng() * this.negTable.length)];
				if (!positive && o === target) continue;
				const oBase = o * dim;
				let s = 0;
				for (let d = 0; d < dim; d++) s += wIn[cBase + d] * wOut[oBase + d];
				const pred = sigmoid(s);
				const g = ((positive ? 1 : 0) - pred) * lr;
				for (let d = 0; d < dim; d++) {
					dv[d] += g * wOut[oBase + d];
					wOut[oBase + d] += g * wIn[cBase + d];
				}
				loss += positive ? -Math.log(pred + 1e-9) : -Math.log(1 - pred + 1e-9);
			}
			for (let d = 0; d < dim; d++) wIn[cBase + d] += dv[d];
			this.lossEma = Number.isNaN(this.lossEma) ? loss : this.lossEma * 0.999 + loss * 0.001;
			this.pairsSeen++;
		}
	}

	vector(id: number): Float32Array {
		return this.wIn.subarray(id * this.dim, (id + 1) * this.dim);
	}

	cosine(a: number, b: number): number {
		const va = this.vector(a);
		const vb = this.vector(b);
		let d = 0;
		let na = 0;
		let nb = 0;
		for (let i = 0; i < this.dim; i++) {
			d += va[i] * vb[i];
			na += va[i] * va[i];
			nb += vb[i] * vb[i];
		}
		return d / (Math.sqrt(na * nb) + 1e-12);
	}

	/** k nearest vocabulary words by cosine, excluding the word itself. */
	neighbors(id: number, k = 8): Array<{ id: number; sim: number }> {
		const sims: Array<{ id: number; sim: number }> = [];
		for (let j = 0; j < this.V; j++) if (j !== id) sims.push({ id: j, sim: this.cosine(id, j) });
		sims.sort((x, y) => y.sim - x.sim);
		return sims.slice(0, k);
	}

	/** a − b + c ≈ ? — top-k completions by cosine, excluding a, b, c. */
	analogy(a: number, b: number, c: number, k = 3): Array<{ id: number; sim: number }> {
		const { dim } = this;
		const q = new Float32Array(dim);
		const va = this.vector(a);
		const vb = this.vector(b);
		const vc = this.vector(c);
		for (let d = 0; d < dim; d++) q[d] = va[d] - vb[d] + vc[d];
		let nq = 0;
		for (let d = 0; d < dim; d++) nq += q[d] * q[d];
		nq = Math.sqrt(nq) + 1e-12;
		const out: Array<{ id: number; sim: number }> = [];
		for (let j = 0; j < this.V; j++) {
			if (j === a || j === b || j === c) continue;
			const vj = this.vector(j);
			let dj = 0;
			let nj = 0;
			for (let d = 0; d < dim; d++) {
				dj += q[d] * vj[d];
				nj += vj[d] * vj[d];
			}
			out.push({ id: j, sim: dj / (nq * Math.sqrt(nj) + 1e-12) });
		}
		out.sort((x, y) => y.sim - x.sim);
		return out.slice(0, k);
	}
}

// ── 2-component PCA ──────────────────────────────────────────────────────────
// Ported from LangX (src/lib/runtime/rag/pca.ts): power iteration with the
// implicit covariance product Cu = Xᵀ(Xu) — the d×d covariance is never
// materialized, and the deterministic seed keeps the map's orientation stable
// across refits, so the scatter drifts instead of jumping.

export interface Pca2 {
	mean: number[];
	c1: number[];
	c2: number[];
}

function dot(a: number[], b: number[]): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

function norm(v: number[]): number {
	return Math.sqrt(dot(v, v));
}

function pcaSeed(d: number, salt: number): number[] {
	const v = new Array<number>(d);
	for (let i = 0; i < d; i++) v[i] = Math.sin((i + 1) * 12.9898 + salt * 78.233);
	const n = norm(v) || 1;
	return v.map((x) => x / n);
}

export function fitPca2(vectors: number[][]): Pca2 {
	const n = vectors.length;
	const d = vectors[0]?.length ?? 0;
	const mean = new Array<number>(d).fill(0);
	for (const v of vectors) for (let i = 0; i < d; i++) mean[i] += v[i];
	for (let i = 0; i < d; i++) mean[i] /= Math.max(1, n);
	const X = vectors.map((v) => v.map((x, i) => x - mean[i]));

	const cmul = (u: number[]): number[] => {
		const xu = X.map((row) => dot(row, u));
		const out = new Array<number>(d).fill(0);
		for (let r = 0; r < n; r++) {
			const s = xu[r];
			const row = X[r];
			for (let i = 0; i < d; i++) out[i] += row[i] * s;
		}
		return out;
	};

	const iterate = (salt: number, ortho: number[][]): number[] => {
		let v = pcaSeed(d, salt);
		for (let t = 0; t < 64; t++) {
			const w = cmul(v);
			for (const o of ortho) {
				const p = dot(w, o);
				for (let i = 0; i < d; i++) w[i] -= p * o[i];
			}
			const wn = norm(w);
			if (wn < 1e-9) break;
			v = w.map((x) => x / wn);
		}
		return v;
	};

	const c1 = d ? iterate(1, []) : [];
	const c2 = d ? iterate(2, [c1]) : [];
	return { mean, c1, c2 };
}

export function project(v: ArrayLike<number>, pca: Pca2): [number, number] {
	const { mean, c1, c2 } = pca;
	let x = 0;
	let y = 0;
	for (let i = 0; i < mean.length; i++) {
		const cv = v[i] - mean[i];
		x += cv * c1[i];
		y += cv * c2[i];
	}
	return [x, y];
}
