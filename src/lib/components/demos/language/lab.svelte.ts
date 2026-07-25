// The shared scribe: one WorkerEngine, one corpus, one training history that
// three plates of this chapter read. Plate III trains it; the surprise meter
// and the forward-pass walkthrough interrogate the same weights, so "ask the
// model you trained" is literal. Module-level so the model survives scrolling;
// the chapter page calls disposeAll() on unmount — GPU memory is no souvenir.

import { loadCorpus, type Corpus } from '$lib/data/corpus';
import { detectWebGPU } from '$lib/nn/engine';
import { WorkerEngine } from '$lib/llm/worker-engine';
import type { ModelConfig, PerTokenInfo } from '$lib/llm/engine';
import { progress } from '$lib/data/progress.svelte';

export const SCRIBE_CONFIG: ModelConfig = {
	name: 'scribe',
	nLayer: 2,
	nEmbd: 96,
	nHead: 4,
	blockSize: 96,
	vocab: 69
};

/** Loss of knowing nothing: −log(1/69). The chart's dashed reference line. */
export const UNIFORM_NATS = Math.log(SCRIBE_CONFIG.vocab);
export const AUTO_PROMPT = 'Once upon a time';
/** Steps per training burst — held-out eval and a fresh desk sample after each.
 * (The LM worker syncs the loss every step by design — pipelining measured
 * slower in jax-js, upstream #151 — so chunk size is the only pacing knob.) */
export const TRAIN_CHUNK = 40;
const SAMPLE_CHARS = 160;
/** The worker keeps only the last blockSize/2 prompt tokens; mirror that cap. */
const MAX_PROMPT = SCRIBE_CONFIG.blockSize / 2;
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

	readonly paramCount = countParams(SCRIBE_CONFIG);
	readonly device = 'webgpu';

	corpus: Corpus | null = null;
	private engine: WorkerEngine | null = null;
	private initCkpt: ArrayBuffer | null = null;
	private webgpu: boolean | null = null;
	private playing = false;
	private trainPromise: Promise<void> | null = null;
	private gen = 0;
	private sampleSeq = 0;

	/** Cheap adapter probe on mount, so no-WebGPU readers get prose, not a dead button. */
	async probe(): Promise<void> {
		if (this.webgpu !== null) return;
		this.webgpu = await detectWebGPU();
		if (!this.webgpu && this.phase === 'idle') this.phase = 'no-webgpu';
	}

	async boot(): Promise<void> {
		if (this.phase === 'loading' || this.phase === 'ready' || this.phase === 'training') return;
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
			if (stale) void stale.dispose();
			this.loadNote = 'fetching the story corpus (about 1.5 MB)…';
			const corpus = await loadCorpus();
			if (myGen !== this.gen) return;
			this.corpus = corpus;
			this.loadNote = 'building a fresh transformer on your GPU…';
			const engine = new WorkerEngine({
				tokenData: corpus.tokens,
				decode: (ids) => corpus.decode(ids),
				decodeOne: (id) => corpus.vocab[id] ?? '',
				seed: 42,
				lr: 1.5e-3
			});
			this.engine = engine;
			await engine.init(SCRIBE_CONFIG);
			if (myGen !== this.gen) return;
			// Kept so Reset can restore step-0 weights without a re-init.
			this.initCkpt = await engine.exportCheckpoint();
			if (myGen !== this.gen) return;
			this.loadNote = 'asking the untrained model to write…';
			const v0 = await engine.valLoss();
			if (myGen !== this.gen) return;
			this.step = 0;
			this.lossNow = NaN;
			this.tokPerSec = 0;
			this.trainLoss = [];
			this.valPoints = [[0, v0]];
			this.samples = [];
			this.spoke = false;
			this.phase = 'ready';
			void this.autoSample(); // the step-0 baseline: uniform noise, on record
		} catch (err) {
			if (myGen !== this.gen) return;
			this.errorMsg = err instanceof Error ? err.message : String(err);
			this.phase = 'error';
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
				await this.autoSample();
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
	private async sampleChars(promptIds: number[], temperature: number): Promise<string | null> {
		const e = this.engine;
		if (!e) return null;
		const myGen = this.gen;
		let ids = promptIds.slice(-MAX_PROMPT);
		let out = '';
		while (out.length < SAMPLE_CHARS) {
			const r = await e.sample(ids, {
				temperature,
				topK: 40,
				maxTokens: Math.min(SCRIBE_CONFIG.blockSize - 1, SAMPLE_CHARS - out.length)
			});
			if (myGen !== this.gen) return null;
			if (r.tokens.length === 0) break;
			out += r.text;
			ids = [...ids, ...r.tokens].slice(-MAX_PROMPT);
		}
		return out;
	}

	/** The desk's fixed question: same prompt, same temperature, every burst —
	 * so the only thing that changes between samples is the weights. */
	private async autoSample(): Promise<void> {
		const c = this.corpus;
		if (!c || this.sampling) return;
		this.sampling = true;
		const myGen = this.gen;
		const atStep = this.step;
		try {
			const text = await this.sampleChars(c.encode(AUTO_PROMPT), 0.8);
			if (text === null || myGen !== this.gen) return;
			this.pushSample({ step: atStep, prompt: AUTO_PROMPT, text, temperature: 0.8, auto: true });
		} catch {
			// disposed mid-sample
		} finally {
			this.sampling = false;
		}
	}

	async sampleNow(promptText: string, temperature: number): Promise<void> {
		const c = this.corpus;
		if (!c || this.sampling || !this.engine) return;
		this.sampling = true;
		const myGen = this.gen;
		const atStep = this.step;
		try {
			let ids = c.encode(promptText).slice(0, MAX_PROMPT);
			if (ids.length === 0) ids = c.encode(AUTO_PROMPT);
			const shown = c.decode(ids); // what the model actually read, post-drop
			const text = await this.sampleChars(ids, temperature);
			if (text === null || myGen !== this.gen) return;
			this.pushSample({ step: atStep, prompt: shown, text, temperature, auto: false });
		} catch {
			// disposed mid-sample
		} finally {
			this.sampling = false;
		}
	}

	private pushSample(s: Omit<DeskSample, 'id'>): void {
		this.samples = [{ id: ++this.sampleSeq, ...s }, ...this.samples].slice(0, 4);
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
		return this.corpus?.encode(s) ?? [];
	}

	charOf(id: number): string {
		return this.corpus?.vocab[id] ?? '';
	}

	/** Page unmount: terminate the worker, return to the powered-off state.
	 * Also runs after server prerender — must stay browser-safe. */
	disposeAll(): void {
		this.gen++;
		this.playing = false;
		const e = this.engine;
		this.engine = null;
		if (e) void e.dispose();
		this.trainPromise = null;
		this.initCkpt = null;
		this.corpus = null; // loadCorpus caches the fetch; this is just our handle
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
