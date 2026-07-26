// Main-thread LlmEngine backed by the training worker (worker.ts) over a tiny
// promise-based RPC. One WorkerEngine per mounted playground.

import type {
	EngineCapability,
	LlmEngine,
	ModelConfig,
	PerTokenInfo,
	SampleResult,
	TrainStepMetrics
} from './engine';

interface Pending {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
	onMetrics?: (m: TrainStepMetrics) => void;
}

export interface WorkerEngineOptions {
	/** Pretokenized corpus the worker samples batches from. */
	tokenData: Uint16Array;
	seed?: number;
	lr?: number;
	/** Decode token ids to text (for SampleResult.text / PerTokenInfo.text). */
	decode: (ids: number[]) => string;
	/** Decode a single token id (inspector labels). */
	decodeOne: (id: number) => string;
	stopToken?: number;
}

export class WorkerEngine implements LlmEngine {
	readonly capability: EngineCapability = 'webgpu';
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private opts: WorkerEngineOptions;

	constructor(opts: WorkerEngineOptions) {
		this.opts = opts;
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (e) => this.onMessage(e);
		this.worker.onerror = (e) => {
			const err = new Error(e.message || 'worker error');
			for (const p of this.pending.values()) p.reject(err);
			this.pending.clear();
		};
	}

	private onMessage(e: MessageEvent) {
		const msg = e.data;
		const p = this.pending.get(msg.id);
		if (!p) return;
		if (msg.event === 'metrics') {
			p.onMetrics?.(msg.m as TrainStepMetrics);
			return;
		}
		this.pending.delete(msg.id);
		if (msg.ok) p.resolve(msg.result);
		else p.reject(new Error(msg.error));
	}

	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		transfer: Transferable[] = [],
		onMetrics?: (m: TrainStepMetrics) => void
	): Promise<T> {
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onMetrics });
			this.worker.postMessage({ id, op, ...payload }, transfer);
		});
	}

	async init(config: ModelConfig, checkpoint?: ArrayBuffer): Promise<void> {
		// Copy tokenData: the buffer is transferred and would otherwise detach.
		const copy = this.opts.tokenData.slice();
		const payload: Record<string, unknown> = {
			config,
			tokenData: copy.buffer,
			seed: this.opts.seed ?? 42,
			lr: this.opts.lr ?? 3e-4
		};
		const transfer: Transferable[] = [copy.buffer];
		if (checkpoint) {
			payload.checkpoint = checkpoint;
			transfer.push(checkpoint);
		}
		await this.call('init', payload, transfer);
	}

	train(steps: number, onMetrics: (m: TrainStepMetrics) => void): Promise<void> {
		return this.call('train', { steps }, [], onMetrics).then(() => undefined);
	}

	async stop(): Promise<void> {
		await this.call('stop');
	}

	async sample(
		promptTokens: number[],
		opts?: { temperature?: number; topK?: number; maxTokens?: number }
	): Promise<SampleResult> {
		const r = await this.call<{ tokens: number[] }>('sample', {
			promptTokens: [...promptTokens],
			temperature: opts?.temperature,
			topK: opts?.topK,
			maxTokens: opts?.maxTokens,
			stopToken: this.opts.stopToken
		});
		return { tokens: r.tokens, text: this.opts.decode(r.tokens) };
	}

	/** Full next-token log-prob row for a context (legal-move masking, etc.). */
	async nextDistribution(tokens: number[]): Promise<Float32Array> {
		const r = await this.call<{ row: ArrayBuffer }>('nextdist', { tokens: [...tokens] });
		return new Float32Array(r.row);
	}

	/** Swap the training corpus in place; weights and optimizer survive.
	 * Continuing to train after this IS supervised fine-tuning. */
	async setTokens(tokens: Uint16Array): Promise<number> {
		const copy = tokens.slice();
		const r = await this.call<{ tokens: number }>('settokens', { tokenData: copy.buffer }, [
			copy.buffer
		]);
		return r.tokens;
	}

	/** Sample G continuations of one prompt in lockstep (for RLVR groups). */
	async sampleGroup(
		promptTokens: number[],
		opts?: { g?: number; maxNew?: number; temperature?: number; topK?: number }
	): Promise<number[][]> {
		const r = await this.call<{ seqs: number[][] }>('samplegroup', {
			promptTokens: [...promptTokens],
			g: opts?.g,
			maxNew: opts?.maxNew,
			temperature: opts?.temperature,
			topK: opts?.topK,
			stopToken: this.opts.stopToken
		});
		return r.seqs;
	}

	/** One verifier-rewarded policy-gradient update (REINFORCE with the
	 * group-relative advantages passed in). seqs is [G, blockSize+1] Int32,
	 * -1 padded; starts[g] = index of the first generated token in row g. */
	async rlStep(
		seqs: Int32Array,
		advs: number[],
		starts: number[]
	): Promise<{ loss: number; step: number; skipped?: boolean }> {
		const copy = seqs.slice();
		return this.call('rlstep', { seqs: copy.buffer, advs, starts }, [copy.buffer]);
	}

	async inspect(tokens: number[]): Promise<PerTokenInfo[]> {
		const r = await this.call<{ perToken: PerTokenInfo[] }>('inspect', { tokens: [...tokens] });
		return r.perToken.map((t) => ({ ...t, text: this.opts.decodeOne(t.id) }));
	}

	/** Change the learning rate (Adam moments reset). */
	async setLr(lr: number): Promise<void> {
		await this.call('setlr', { lr });
	}

	/** Residual stream after each block for a token sequence. layers[li] is a
	 * flat [seqLen · nEmbd] Float32Array (row-major by position). The surface a
	 * linear probe reads to test whether the board is linearly decodable. */
	async residuals(tokens: number[]): Promise<{
		seqLen: number;
		nEmbd: number;
		nLayer: number;
		layers: Float32Array[];
	}> {
		const r = await this.call<{
			layers: ArrayBuffer[];
			seqLen: number;
			nEmbd: number;
			nLayer: number;
		}>('residual', { tokens: [...tokens] });
		return {
			seqLen: r.seqLen,
			nEmbd: r.nEmbd,
			nLayer: r.nLayer,
			layers: r.layers.map((b) => new Float32Array(b))
		};
	}

	/** Mean loss over fixed held-out batches (deterministic across calls). */
	async valLoss(): Promise<number> {
		const r = await this.call<{ valLoss: number }>('valloss');
		return r.valLoss;
	}

	/** Per-tensor gradient L2 norms on a fixed diagnostic batch. */
	async gradNorms(): Promise<{ loss: number; norms: Array<{ name: string; norm: number }> }> {
		return this.call('gradnorms');
	}

	/** Real attention patterns for a prompt. layers[li] is the [H·S, S] block
	 * for layer li; row h·S+i holds head h's attention from position i. */
	async attention(tokens: number[]): Promise<{
		seqLen: number;
		nHead: number;
		blockSize: number;
		layers: Float32Array[];
	}> {
		const r = await this.call<{
			layers: ArrayBuffer[];
			nHead: number;
			blockSize: number;
			seqLen: number;
		}>('attention', { tokens: [...tokens] });
		return {
			seqLen: r.seqLen,
			nHead: r.nHead,
			blockSize: r.blockSize,
			layers: r.layers.map((b) => new Float32Array(b))
		};
	}

	async exportCheckpoint(): Promise<ArrayBuffer> {
		const r = await this.call<{ checkpoint: ArrayBuffer }>('export');
		return r.checkpoint;
	}

	/** Swap the resident weights for a checkpoint in place (cheaper than a full
	 * re-init: no tokenData re-transfer). The buffer is transferred. Used by the
	 * time-machine scrubber to re-roll a waypoint's dequantized weights live. */
	async loadWeights(checkpoint: ArrayBuffer): Promise<void> {
		await this.call('load', { checkpoint }, [checkpoint]);
	}

	/** A worker mid-jit answers no RPC promptly, and a worker that never releases
	 * its GPU device blocks the next one from getting it — so the graceful
	 * request gets a deadline, and termination happens either way. */
	async dispose(): Promise<void> {
		try {
			await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]);
		} finally {
			this.worker.terminate();
			this.pending.clear();
		}
	}
}
