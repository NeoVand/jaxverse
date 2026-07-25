// Main-thread client for the MLP training worker — a promise RPC with a
// streaming side-channel for per-step metrics. One MlpEngine per mounted demo.

import type { DataSpec, EvalResult, LayerWeights, MlpConfig, TrainMetrics } from './engine';

interface Pending {
	resolve: (v: unknown) => void;
	reject: (e: Error) => void;
	onMetrics?: (m: TrainMetrics) => void;
}

export class MlpEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	device: string = 'unknown';
	paramCount = 0;

	constructor() {
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
			p.onMetrics?.(msg.m as TrainMetrics);
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
		onMetrics?: (m: TrainMetrics) => void
	): Promise<T> {
		const id = this.nextId++;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onMetrics });
			this.worker.postMessage({ id, op, ...payload }, transfer);
		});
	}

	async init(config: MlpConfig, data: DataSpec, checkpoint?: ArrayBuffer): Promise<void> {
		const x = data.x.slice();
		const y = data.y.slice();
		const payload: Record<string, unknown> = {
			config,
			x: x.buffer,
			y: y.buffer,
			n: data.n
		};
		const transfer: Transferable[] = [x.buffer, y.buffer];
		if (checkpoint) {
			payload.checkpoint = checkpoint;
			transfer.push(checkpoint);
		}
		const r = await this.call<{ device: string; paramCount: number }>('init', payload, transfer);
		this.device = r.device;
		this.paramCount = r.paramCount;
	}

	/** Swap the dataset (same dims) — weights and jit caches survive. */
	async setData(data: DataSpec): Promise<void> {
		const x = data.x.slice();
		const y = data.y.slice();
		await this.call('setdata', { x: x.buffer, y: y.buffer, n: data.n }, [x.buffer, y.buffer]);
	}

	/** Fresh random weights; optimizer state resets too. */
	async reset(seed?: number): Promise<void> {
		await this.call('reset', { seed });
	}

	/** Runs `steps` updates; metrics stream on every synced step. For tiny
	 * models pass syncEvery 4–8: loss is read (and posted) only that often,
	 * which removes most GPU-sync stalls. */
	train(steps: number, onMetrics: (m: TrainMetrics) => void, syncEvery = 1): Promise<void> {
		return this.call('train', { steps, syncEvery }, [], onMetrics).then(() => undefined);
	}

	async stop(): Promise<void> {
		await this.call('stop');
	}

	/** Forward n rows; returns n × outDim predictions (logits for xent). */
	async predict(x: Float32Array, n: number, chunk = 256): Promise<Float32Array> {
		const copy = x.slice();
		const r = await this.call<{ y: ArrayBuffer }>('predict', { x: copy.buffer, n, chunk }, [
			copy.buffer
		]);
		return new Float32Array(r.y);
	}

	/** Post-activation values of every layer for the given rows. */
	async activations(
		x: Float32Array,
		n: number,
		chunk = 1024
	): Promise<{ layers: Float32Array[]; widths: number[] }> {
		const copy = x.slice();
		const r = await this.call<{ layers: ArrayBuffer[]; widths: number[] }>(
			'activations',
			{ x: copy.buffer, n, chunk },
			[copy.buffer]
		);
		return { layers: r.layers.map((b) => new Float32Array(b)), widths: r.widths };
	}

	/** Run the network tail from layer `from` given that layer's activations. */
	async forwardFrom(from: number, h: Float32Array, n: number, chunk = 256): Promise<Float32Array> {
		const copy = h.slice();
		const r = await this.call<{ y: ArrayBuffer }>(
			'forwardfrom',
			{ from, h: copy.buffer, n, chunk },
			[copy.buffer]
		);
		return new Float32Array(r.y);
	}

	/** Saliency: d(class score)/d(input) for one row. */
	async inputGrad(x: Float32Array, target: number): Promise<Float32Array> {
		const copy = x.slice();
		const r = await this.call<{ g: ArrayBuffer }>('inputgrad', { x: copy.buffer, target }, [
			copy.buffer
		]);
		return new Float32Array(r.g);
	}

	/** Held-out loss (and accuracy for classifiers) — deterministic. */
	eval(): Promise<EvalResult> {
		return this.call<EvalResult>('eval');
	}

	/** Loss on a fixed training batch — comparable across calls. */
	async trainLoss(): Promise<number> {
		const r = await this.call<{ loss: number }>('trainloss');
		return r.loss;
	}

	/** Every layer's weight matrix and bias, for weight visualizations. */
	async weights(): Promise<LayerWeights[]> {
		const r = await this.call<{ buffers: ArrayBuffer[]; layers: number[] }>('weights');
		const out: LayerWeights[] = [];
		for (let k = 0; k + 1 < r.layers.length; k++) {
			out.push({
				w: new Float32Array(r.buffers[2 * k]),
				b: new Float32Array(r.buffers[2 * k + 1]),
				inDim: r.layers[k],
				outDim: r.layers[k + 1]
			});
		}
		return out;
	}

	async exportCheckpoint(): Promise<ArrayBuffer> {
		const r = await this.call<{ checkpoint: ArrayBuffer }>('export');
		return r.checkpoint;
	}

	async setLr(lr: number): Promise<void> {
		await this.call('setlr', { lr });
	}

	async dispose(): Promise<void> {
		try {
			await this.call('dispose');
		} finally {
			this.worker.terminate();
			this.pending.clear();
		}
	}
}
