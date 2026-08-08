// The shared scribe: one training WorkerEngine, one corpus, one training
// history that three plates of this chapter read. the scribe trains it; the
// surprise meter and the forward-pass walkthrough interrogate the same
// weights, so "ask the model you trained" is literal. A second WorkerEngine
// (the sampler) writes the desk's samples from couriered checkpoints so the
// loss curve never pauses for them. Module-level so the model survives
// scrolling; the chapter page calls disposeAll() on unmount — GPU memory is
// no souvenir.

import { detectWebGPU } from '$lib/nn/engine';
import { WorkerEngine } from '$lib/llm/worker-engine';
import type { ModelConfig, PerTokenInfo } from '$lib/llm/engine';
import { progress } from '$lib/data/progress.svelte';
import {
	adoptedTokenizer,
	charTokenizer,
	pieceTokenizer,
	type TokenizedCorpus,
	type TokenizerKind
} from '$lib/data/tokenizer';
import type { MergePair } from '$lib/data/bpe';

/** Everything but the vocabulary is fixed; the vocabulary is whatever the
 * tokenizer hands over, which is the point of the chapter. */
export const SCRIBE_SHAPE = {
	name: 'scribe',
	nLayer: 2,
	nEmbd: 96,
	nHead: 4,
	blockSize: 96
} as const;

export function configFor(vocabSize: number): ModelConfig {
	return { ...SCRIBE_SHAPE, vocab: vocabSize };
}

export const AUTO_PROMPT = 'Once upon a time';
/** Steps per training burst — held-out eval and a fresh desk sample after each.
 * (The LM worker syncs the loss every step by design — pipelining measured
 * slower in jax-js, upstream #151 — so chunk size is the only pacing knob.) */
export const TRAIN_CHUNK = 40;
const SAMPLE_CHARS = 160;
/** The worker keeps only the last blockSize/2 prompt tokens; mirror that cap. */
const MAX_PROMPT = SCRIBE_SHAPE.blockSize / 2;
const MILESTONE_STEP = 800;

export type ScribePhase = 'idle' | 'loading' | 'ready' | 'training' | 'error' | 'no-webgpu';

export interface DeskSample {
	id: number;
	step: number;
	prompt: string;
	text: string;
	temperature: number;
	auto: boolean;
}

export interface AttentionSnap {
	seqLen: number;
	nHead: number;
	blockSize: number;
	layers: Float32Array[];
}

/** A stalled fetch or a GPU device that never arrives would otherwise leave the
 * plate saying "loading…" forever. Give every boot step a deadline so the
 * reader gets an error and a Retry instead of a spinner with no end. */
function guard<T>(what: string, p: Promise<T>, ms = 25_000): Promise<T> {
	return Promise.race([
		p,
		new Promise<never>((_, reject) =>
			setTimeout(() => reject(new Error(`${what} timed out — try again`)), ms)
		)
	]);
}

function countParams(c: ModelConfig): number {
	// wte + lmHead (2·V·d) + wpe (S·d), plus per layer wq/wk/wv/wo (4·d²) and
	// the 4×-wide MLP pair (8·d²).
	return 2 * c.vocab * c.nEmbd + c.blockSize * c.nEmbd + c.nLayer * 12 * c.nEmbd * c.nEmbd;
}

class ScribeLab {
	phase = $state<ScribePhase>('idle');
	loadNote = $state('');
	errorMsg = $state('');
	step = $state(0);
	lossNow = $state(NaN);
	tokPerSec = $state(0);
	/** [step, train loss], one entry per optimizer step. */
	trainLoss = $state<Array<[number, number]>>([]);
	/** [step, loss] on fixed held-out batches — step 0, then one per burst. */
	valPoints = $state<Array<[number, number]>>([]);
	/** Newest first, capped at 4: the desk's latest sample plus three archived. */
	samples = $state<DeskSample[]>([]);
	sampling = $state(false);
	spoke = $state(false);

	/** Which vocabulary the model reads, and how big it turned out. */
	kind = $state<TokenizerKind>('pieces');
	vocabSize = $state(0);
	mergeCount = $state(0);
	/** Characters per token on this corpus: 1 for characters, ~2.4 for pieces.
	 * Losses in nats/token divide by it to become nats/character, the only way
	 * to compare two tokenizers honestly. */
	charsPerToken = $state(1);
	corpusTokens = $state(0);
	paramCount = $state(0);
	readonly device = 'webgpu';

	/** The vocabulary in force: encode/decode for every plate in the chapter. */
	private tokenizer: TokenizedCorpus | null = null;
	/** Set by the tokenizer plate when the reader hands over a vocabulary of your own. */
	private pending: TokenizedCorpus | null = null;
	private engine: WorkerEngine | null = null;
	/** The desk's own scribe: a second worker with its own GPU device. Each
	 * burst the trainer exports a checkpoint (one quick readback) and this
	 * worker writes the sample from it, so the loss curve never stops for the
	 * desk — and every sample is exactly the weights of the step it names.
	 * Until it boots (or if it can't), sampling rides the training worker and
	 * the loop waits, as it always used to. */
	private sampler: WorkerEngine | null = null;
	private samplerReady = false;
	private samplePromise: Promise<void> | null = null;
	private bootPromise: Promise<void> | null = null;
	private initCkpt: ArrayBuffer | null = null;
	private webgpu: boolean | null = null;
	private playing = false;
	private trainPromise: Promise<void> | null = null;
	private gen = 0;
	private sampleSeq = 0;

	/** Which vocabulary the next boot should use: one the reader handed over,
	 * else the shipped snapshot, else characters. */
	private async resolveTokenizer(): Promise<TokenizedCorpus> {
		if (this.pending) {
			const t = this.pending;
			this.pending = null;
			return t;
		}
		if (this.kind === 'chars') return charTokenizer();
		return pieceTokenizer((done, total) => {
			if (done < total) this.loadNote = `applying ${total} merges to the corpus (${done})…`;
		});
	}

	private adopt(tok: TokenizedCorpus): void {
		this.tokenizer = tok;
		this.kind = tok.kind;
		this.vocabSize = tok.vocab.size;
		this.mergeCount = tok.mergeCount;
		this.charsPerToken = tok.vocab.charsPerToken;
		this.corpusTokens = tok.tokens.length;
		this.paramCount = countParams(configFor(tok.vocab.size));
	}

	/** Loss of knowing nothing, in nats per token: every door held equally open. */
	get uniformNats(): number {
		return Math.log(Math.max(2, this.vocabSize));
	}

	/** Nats per token → bits per character, the unit that survives a change of
	 * tokenizer: a token carries charsPerToken characters, and a nat is 1/ln2 bits. */
	bitsPerChar(natsPerToken: number): number {
		return natsPerToken / Math.LN2 / this.charsPerToken;
	}

	/** Switch vocabulary: a new embedding table means a new model, so this
	 * rebuilds from scratch. */
	async setKind(kind: TokenizerKind): Promise<void> {
		if (kind === this.kind && this.tokenizer?.kind === kind && !this.pending) return;
		this.kind = kind;
		this.pending = null;
		await this.rebuild();
	}

	/** the tokenizer plate handing its vocabulary to the model, merges and corpus and all. */
	async useVocabulary(
		chars: readonly string[],
		pairs: readonly MergePair[],
		tokens: Uint16Array,
		originalLen: number
	): Promise<void> {
		this.pending = adoptedTokenizer(chars, pairs, tokens, originalLen);
		this.kind = 'pieces';
		await this.rebuild();
	}

	/** A vocabulary change means a new model. Let any boot already in flight
	 * finish first — two boots racing would leave one worker holding the GPU
	 * device that the other is waiting for. */
	private async rebuild(): Promise<void> {
		await this.pause();
		if (this.bootPromise) await this.bootPromise.catch(() => {});
		this.phase = 'idle';
		await this.boot();
	}

	/** Cheap adapter probe on mount, so no-WebGPU readers get prose, not a dead button. */
	async probe(): Promise<void> {
		if (this.webgpu !== null) return;
		this.webgpu = await detectWebGPU();
		if (!this.webgpu && this.phase === 'idle') this.phase = 'no-webgpu';
	}

	async boot(): Promise<void> {
		if (this.phase === 'loading' || this.phase === 'ready' || this.phase === 'training') return;
		this.bootPromise = this.bootOnce();
		await this.bootPromise;
	}

	private async bootOnce(): Promise<void> {
		const myGen = ++this.gen;
		this.errorMsg = '';
		this.phase = 'loading';
		try {
			if (this.webgpu === null) this.webgpu = await detectWebGPU();
			if (myGen !== this.gen) return;
			if (!this.webgpu) {
				this.phase = 'no-webgpu';
				return;
			}
			const stale = this.engine;
			this.engine = null;
			const staleSampler = this.sampler;
			this.sampler = null;
			this.samplerReady = false;
			// awaited, not fired and forgotten: the old workers must let go of
			// their GPU devices before the new ones ask (dispose has a deadline)
			if (stale) await stale.dispose();
			if (staleSampler) await staleSampler.dispose();
			if (myGen !== this.gen) return;
			this.loadNote = 'fetching the story corpus (about 1.5 MB)…';
			const tok = await guard('the corpus', this.resolveTokenizer());
			if (myGen !== this.gen) return;
			this.adopt(tok);
			this.loadNote = 'building a fresh transformer on your GPU…';
			const config = configFor(tok.vocab.size);
			const engine = new WorkerEngine({
				tokenData: tok.tokens,
				decode: (ids) => tok.vocab.decode(ids),
				decodeOne: (id) => tok.vocab.table[id] ?? '',
				seed: 42,
				lr: 1.5e-3
			});
			this.engine = engine;
			// Superseded from here on means: hand the device back. A worker dropped
			// without dispose() keeps its GPU device, and the next boot waits forever.
			const superseded = () => {
				if (myGen === this.gen) return false;
				if (this.engine === engine) this.engine = null;
				void engine.dispose();
				return true;
			};
			await guard('the GPU', engine.init(config));
			if (superseded()) return;
			// Kept so Reset can restore step-0 weights without a re-init.
			this.initCkpt = await guard('the first weights', engine.exportCheckpoint());
			if (superseded()) return;
			this.loadNote = 'asking the untrained model to write…';
			const v0 = await engine.valLoss();
			if (superseded()) return;
			this.step = 0;
			this.lossNow = NaN;
			this.tokPerSec = 0;
			this.trainLoss = [];
			this.valPoints = [[0, v0]];
			this.samples = [];
			this.spoke = false;
			this.phase = 'ready';
			void this.autoSample(); // the step-0 baseline: uniform noise, on record
			void this.bootSampler(tok, config, myGen); // the desk's worker, in the background
		} catch (err) {
			if (myGen !== this.gen) return;
			this.errorMsg = err instanceof Error ? err.message : String(err);
			this.phase = 'error';
		}
	}

	/** Boot the sampling worker. Silent on failure: sampling then stays on the
	 * training worker, which merely brings back the pause it used to cause. */
	private async bootSampler(tok: TokenizedCorpus, config: ModelConfig, myGen: number) {
		const s = new WorkerEngine({
			tokenData: tok.tokens,
			decode: (ids) => tok.vocab.decode(ids),
			decodeOne: (id) => tok.vocab.table[id] ?? '',
			seed: 43,
			lr: 1.5e-3
		});
		try {
			await guard('the sampler', s.init(config));
			if (myGen !== this.gen) {
				void s.dispose();
				return;
			}
			this.sampler = s;
			this.samplerReady = true;
		} catch {
			void s.dispose();
		}
	}

	/** Which engine writes the next sample: the sampler, freshly loaded with
	 * the trainer's current weights (training keeps running), or the training
	 * worker itself when the sampler is missing (the caller then waits). */
	private async sampleEngine(): Promise<WorkerEngine | null> {
		const e = this.engine;
		const s = this.samplerReady ? this.sampler : null;
		if (!e || !s) return e;
		try {
			const ckpt = await e.exportCheckpoint();
			await s.loadWeights(ckpt);
			return s;
		} catch {
			// the sampler died mid-courier — retire it and wait inline again
			this.samplerReady = false;
			this.sampler = null;
			void s.dispose();
			return this.engine;
		}
	}

	toggle(): void {
		if (this.phase === 'training') void this.pause();
		else this.start();
	}

	start(): void {
		if (this.playing || this.phase !== 'ready' || !this.engine) return;
		this.playing = true;
		this.phase = 'training';
		this.trainPromise = this.loop();
	}

	async pause(): Promise<void> {
		this.playing = false;
		const e = this.engine;
		if (e) await e.stop().catch(() => {});
		if (this.trainPromise) await this.trainPromise;
		this.trainPromise = null;
		if (this.phase === 'training') this.phase = 'ready';
	}

	private async loop(): Promise<void> {
		const myGen = this.gen;
		while (this.playing && this.engine && myGen === this.gen) {
			const e = this.engine;
			try {
				await e.train(TRAIN_CHUNK, (m) => {
					if (myGen !== this.gen) return;
					this.step = m.step;
					this.lossNow = m.loss;
					this.tokPerSec = this.tokPerSec
						? this.tokPerSec * 0.8 + m.tokensPerSec * 0.2
						: m.tokensPerSec;
					this.trainLoss.push([m.step, m.loss]);
				});
				if (myGen !== this.gen || !this.engine) return;
				const v = await e.valLoss();
				if (myGen !== this.gen) return;
				this.valPoints.push([this.step, v]);
				if (!this.spoke && this.step >= MILESTONE_STEP) {
					this.spoke = true;
					progress.reach('language:spoke');
				}
				// With the sampling worker up, the desk writes on its own device
				// while the next burst runs — the curve never pauses. (The
				// checkpoint export posts before the next train call, so the
				// sample is exactly this burst's weights.) Without it, wait, as
				// the loop always used to.
				if (this.samplerReady) void this.autoSample();
				else await this.autoSample();
			} catch (err) {
				if (myGen !== this.gen) return;
				this.playing = false;
				this.errorMsg = err instanceof Error ? err.message : String(err);
				this.phase = 'error';
				return;
			}
		}
	}

	/** Back to the exact step-0 weights (Adam moments reset with them). */
	async reset(): Promise<void> {
		const e = this.engine;
		if (!e || !this.initCkpt || this.phase === 'loading') return;
		const wasPlaying = this.playing;
		await this.pause();
		// a desk sample may still be in flight on the sampling worker; let it
		// land (it gets cleared below) rather than have it surface post-reset
		if (this.samplePromise) await this.samplePromise.catch(() => {});
		const myGen = this.gen;
		try {
			await e.loadWeights(this.initCkpt.slice(0));
			if (myGen !== this.gen) return;
			this.step = 0;
			this.lossNow = NaN;
			this.tokPerSec = 0;
			this.trainLoss = [];
			this.samples = [];
			this.spoke = false;
			const v0 = await e.valLoss();
			if (myGen !== this.gen) return;
			this.valPoints = [[0, v0]];
			await this.autoSample();
			if (myGen !== this.gen) return;
			if (wasPlaying) this.start();
		} catch {
			// engine disposed mid-reset — the gen guard already stopped state writes
		}
	}

	/** One sample() call yields at most blockSize−1 tokens; chain calls, feeding
	 * the tail back as prompt, until SAMPLE_CHARS characters exist. */
	private async sampleChars(
		e: WorkerEngine,
		promptIds: number[],
		temperature: number
	): Promise<string | null> {
		const myGen = this.gen;
		let ids = promptIds.slice(-MAX_PROMPT);
		let out = '';
		while (out.length < SAMPLE_CHARS) {
			const r = await e.sample(ids, {
				temperature,
				topK: 40,
				maxTokens: Math.min(SCRIBE_SHAPE.blockSize - 1, SAMPLE_CHARS - out.length)
			});
			if (myGen !== this.gen) return null;
			if (r.tokens.length === 0) break;
			out += r.text;
			ids = [...ids, ...r.tokens].slice(-MAX_PROMPT);
		}
		// The last chunk can overshoot by a whole burst of word-pieces; trim back
		// to the target at a word boundary so every sample is the same length and
		// the desk's cards never change height.
		if (out.length > SAMPLE_CHARS) {
			const cut = out.slice(0, SAMPLE_CHARS).replace(/\s+\S*$/, '');
			out = `${cut} …`;
		}
		return out;
	}

	/** The desk's fixed question: same prompt, same temperature, every burst —
	 * so the only thing that changes between samples is the weights. */
	private autoSample(): Promise<void> {
		const c = this.tokenizer?.vocab;
		if (!c || this.sampling) return Promise.resolve();
		const p = this.writeSample(c, AUTO_PROMPT, 0.8, true);
		this.samplePromise = p;
		return p;
	}

	sampleNow(promptText: string, temperature: number): Promise<void> {
		const c = this.tokenizer?.vocab;
		if (!c || this.sampling || !this.engine) return Promise.resolve();
		const p = this.writeSample(c, promptText, temperature, false);
		this.samplePromise = p;
		return p;
	}

	private async writeSample(
		c: TokenizedCorpus['vocab'],
		promptText: string,
		temperature: number,
		auto: boolean
	): Promise<void> {
		this.sampling = true;
		const myGen = this.gen;
		const atStep = this.step;
		try {
			let ids = c.encode(promptText).slice(0, MAX_PROMPT);
			if (ids.length === 0) ids = c.encode(AUTO_PROMPT);
			const shown = auto ? promptText : c.decode(ids); // what the model actually read
			const via = await this.sampleEngine();
			if (!via || myGen !== this.gen) return;
			const text = await this.sampleChars(via, ids, temperature);
			if (text === null || myGen !== this.gen) return;
			this.pushSample({ step: atStep, prompt: shown, text, temperature, auto });
		} catch {
			// disposed mid-sample
		} finally {
			this.sampling = false;
		}
	}

	private pushSample(s: Omit<DeskSample, 'id'>): void {
		// one on the desk + four in the archive, so its 2×2 grid is always full
		this.samples = [{ id: ++this.sampleSeq, ...s }, ...this.samples].slice(0, 5);
	}

	async inspect(tokens: number[]): Promise<PerTokenInfo[] | null> {
		const e = this.engine;
		if (!e || tokens.length === 0) return null;
		const myGen = this.gen;
		try {
			const r = await e.inspect(tokens);
			return myGen === this.gen ? r : null;
		} catch {
			return null;
		}
	}

	async attention(tokens: number[]): Promise<AttentionSnap | null> {
		const e = this.engine;
		if (!e || tokens.length === 0) return null;
		const myGen = this.gen;
		try {
			const r = await e.attention(tokens);
			return myGen === this.gen ? r : null;
		} catch {
			return null;
		}
	}

	/** The real next-token log-prob row for a context — the walkthrough's final
	 * stage re-softmaxes this row client-side for its temperature control. */
	async nextDistribution(tokens: number[]): Promise<Float32Array | null> {
		const e = this.engine;
		if (!e || tokens.length === 0) return null;
		const myGen = this.gen;
		try {
			const r = await e.nextDistribution(tokens);
			return myGen === this.gen ? r : null;
		} catch {
			return null;
		}
	}

	encode(s: string): number[] {
		return this.tokenizer?.vocab.encode(s) ?? [];
	}

	/** The text one token stands for: a character, or a word piece. */
	textOf(id: number): string {
		return this.tokenizer?.vocab.table[id] ?? '';
	}

	/** Page unmount: terminate the worker, return to the powered-off state.
	 * Also runs after server prerender — must stay browser-safe. */
	disposeAll(): void {
		this.gen++;
		this.playing = false;
		const e = this.engine;
		this.engine = null;
		if (e) void e.dispose();
		const s = this.sampler;
		this.sampler = null;
		this.samplerReady = false;
		if (s) void s.dispose();
		this.samplePromise = null;
		this.trainPromise = null;
		this.initCkpt = null;
		this.tokenizer = null; // the loaders cache; this is just our handle
		this.pending = null;
		this.phase = this.webgpu === false ? 'no-webgpu' : 'idle';
		this.loadNote = '';
		this.errorMsg = '';
		this.step = 0;
		this.lossNow = NaN;
		this.tokPerSec = 0;
		this.trainLoss = [];
		this.valPoints = [];
		this.samples = [];
		this.sampling = false;
		this.spoke = false;
	}
}

export const scribe = new ScribeLab();
