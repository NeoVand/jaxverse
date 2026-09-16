import type {
	WorldInit,
	WorldInfo,
	WorldMetrics,
	WorldEvaluation,
	ReadoutMetrics,
	WorldForecastRequest,
	WorldForecast,
	WorldPlanRequest,
	WorldPlan
} from './engine';

export interface Comparison {
	/** Target updates. Only complete comparisons have both runs at this step. */
	steps: number;
	regularizedStep: number;
	unregularizedStep: number;
	status: 'running' | 'paused' | 'complete';
	regularized: WorldEvaluation;
	unregularized: WorldEvaluation | null;
}

export type WorldWorkerMessage =
	| { id: number; event: 'metrics'; metrics: WorldMetrics }
	| { id: number; event: 'comparison'; comparison: Comparison }
	| { id: number; ok: true; result?: unknown }
	| { id: number; ok: false; error: string };

interface Pending {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	onMetrics?: (metrics: WorldMetrics) => void;
	onComparison?: (comparison: Comparison) => void;
}

/** The only main-thread seam to weights; every expensive operation stays in the worker. */
export class WorldEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 0;
	private disposed = false;
	private failure: Error | null = null;

	constructor() {
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = ({ data }: MessageEvent<WorldWorkerMessage>) => {
			const pending = this.pending.get(data.id);
			if (!pending) return;
			if ('event' in data) {
				if (data.event === 'metrics') pending.onMetrics?.(data.metrics);
				else pending.onComparison?.(data.comparison);
				return;
			}
			this.pending.delete(data.id);
			if (data.ok) pending.resolve(data.result);
			else pending.reject(new Error(data.error));
		};
		this.worker.onerror = (event) =>
			this.fail(new Error(event.message || 'The world-model worker stopped.'));
		this.worker.onmessageerror = () =>
			this.fail(new Error('The world-model worker returned an unreadable message.'));
	}
	private fail(error: Error) {
		this.failure = error;
		this.worker.terminate();
		this.rejectAll(error);
	}

	private rejectAll(error: Error) {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		onMetrics?: (metrics: WorldMetrics) => void,
		onComparison?: (comparison: Comparison) => void
	): Promise<T> {
		if (this.disposed) return Promise.reject(new Error('This experiment has been closed.'));
		if (this.failure) return Promise.reject(this.failure);
		const id = ++this.nextId;
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, {
				resolve: resolve as (value: unknown) => void,
				reject,
				onMetrics,
				onComparison
			});
			this.worker.postMessage({ id, op, ...payload });
		});
	}

	init(options: WorldInit = {}): Promise<WorldInfo & { backend: string }> {
		return this.call('init', { options });
	}
	train(steps: number, onMetrics?: (metrics: WorldMetrics) => void): Promise<WorldMetrics> {
		return this.call('train', { steps }, onMetrics);
	}
	stop(): Promise<void> {
		return this.call('stop');
	}
	evaluate(): Promise<WorldEvaluation> {
		return this.call('evaluate');
	}
	fitReadout(): Promise<ReadoutMetrics> {
		return this.call('readout');
	}
	forecast(request: WorldForecastRequest): Promise<WorldForecast> {
		return this.call('forecast', { request });
	}
	plan(request: WorldPlanRequest): Promise<WorldPlan> {
		return this.call('plan', { request });
	}
	compare(
		onMetrics?: (metrics: WorldMetrics) => void,
		onComparison?: (comparison: Comparison) => void
	): Promise<Comparison> {
		return this.call('compare', {}, onMetrics, onComparison);
	}
	reset(seed?: number): Promise<WorldInfo> {
		return this.call('reset', { seed });
	}
	async dispose(): Promise<void> {
		if (this.disposed) return;
		this.disposed = true;
		this.worker.terminate();
		this.rejectAll(new Error('This experiment has been closed.'));
	}
}
