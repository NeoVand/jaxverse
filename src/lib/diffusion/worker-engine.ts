// Main-thread wrapper around the diffusion worker: one promise per request,
// training metrics delivered on a side channel that does not resolve it.

import type { DiffusionInit, SampleOptions, SampleResult, TrainStepMetrics } from './engine';
import type { EmojiSet } from './corpus';

interface Pending {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
	onMetrics?: (m: TrainStepMetrics) => void;
}

export interface CorpusInfo {
	count: number;
	styles: number;
	tags: string[];
	sets: EmojiSet[];
	emoji: { cp: string; name: string }[];
	hasShipped: boolean;
}

export class DiffusionEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;

	constructor() {
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (e) => this.onMessage(e);
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

	init(base: string, opts: DiffusionInit): Promise<CorpusInfo> {
		const transfer = opts.checkpoint ? [opts.checkpoint] : [];
		return this.call<CorpusInfo>('init', { base, ...opts }, transfer);
	}

	train(steps: number, onMetrics: (m: TrainStepMetrics) => void): Promise<{ step: number }> {
		return this.call<{ step: number }>('train', { steps }, [], onMetrics);
	}

	stop(): Promise<void> {
		return this.call<void>('stop');
	}

	async sample(count: number, opts: SampleOptions): Promise<SampleResult> {
		const r = await this.call<{
			pixels: ArrayBuffer;
			count: number;
			frames?: ArrayBuffer[];
			ms: number;
		}>('sample', { count, ...opts });
		return {
			pixels: new Float32Array(r.pixels),
			count: r.count,
			frames: r.frames?.map((f) => new Float32Array(f)),
			ms: r.ms
		};
	}

	async tiles(indices: number[], style: number): Promise<Float32Array> {
		const r = await this.call<{ pixels: ArrayBuffer }>('tiles', { indices, style });
		return new Float32Array(r.pixels);
	}

	reset(seed: number): Promise<{ step: number }> {
		return this.call<{ step: number }>('reset', { seed });
	}

	setLr(lr: number): Promise<void> {
		return this.call<void>('setlr', { lr });
	}

	/** Always await this before making another engine: one GPU device per worker,
	 *  and a leaked worker blocks the next boot for the life of the tab. */
	async dispose(): Promise<void> {
		try {
			await Promise.race([this.call('dispose'), new Promise((r) => setTimeout(r, 400))]);
		} finally {
			this.worker.terminate();
			this.pending.clear();
		}
	}
}
