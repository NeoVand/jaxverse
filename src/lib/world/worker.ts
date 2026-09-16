/// <reference lib="webworker" />
import { defaultDevice, init } from '@jax-js/jax';
import {
	WorldCore,
	type WorldInit,
	type WorldForecastRequest,
	type WorldPlanRequest
} from './engine';
import type { Comparison, WorldWorkerMessage } from './worker-engine';

interface Request {
	id: number;
	op: string;
	options?: WorldInit;
	steps?: number;
	seed?: number;
	request?: WorldForecastRequest | WorldPlanRequest;
}
let core: WorldCore | null = null;
let options: WorldInit = {};
let stopped = false;
let stepCount = 0;
let queue: Promise<unknown> = Promise.resolve();
function post(message: WorldWorkerMessage) {
	self.postMessage(message);
}

async function handle(req: Request): Promise<unknown> {
	if (req.op === 'init') {
		const devices = await init();
		const backend = devices.includes('webgpu')
			? 'webgpu'
			: devices.includes('wasm')
				? 'wasm'
				: 'cpu';
		defaultDevice(backend);
		options = req.options ?? {};
		stepCount = 0;
		core?.dispose();
		core = new WorldCore(options);
		return { ...(await core.init()), backend };
	}
	if (!core) throw new Error('The model is still being prepared.');
	if (req.op === 'train') {
		stopped = false;
		return core.train(
			req.steps ?? 20,
			(metrics) => {
				stepCount = metrics.step;
				post({ id: req.id, event: 'metrics', metrics });
			},
			() => stopped
		);
	}
	if (req.op === 'evaluate') return core.evaluate();
	if (req.op === 'readout') return core.fitReadout();
	if (req.op === 'forecast') return core.forecast(req.request as WorldForecastRequest);
	if (req.op === 'plan') return core.plan(req.request as WorldPlanRequest);
	if (req.op === 'reset') {
		options = { ...options, seed: req.seed ?? options.seed };
		stepCount = 0;
		return core.reset(req.seed);
	}
	if (req.op === 'compare') {
		if (!stepCount) throw new Error('Train the model before comparing objectives.');
		stopped = false;
		const target = stepCount;
		let comparison: Comparison = {
			steps: target,
			regularizedStep: target,
			unregularizedStep: 0,
			status: 'running',
			regularized: await core.evaluate(),
			unregularized: null
		};
		const emit = () => post({ id: req.id, event: 'comparison', comparison });
		emit();
		const baseline = await core.createUnregularizedBaseline();
		try {
			const checkpoint = async () => {
				const unregularized = await baseline.evaluate();
				comparison = {
					...comparison,
					unregularized,
					unregularizedStep: unregularized.step,
					status: unregularized.step >= target ? 'complete' : stopped ? 'paused' : 'running'
				};
				emit();
			};
			await checkpoint();
			while (!stopped && comparison.unregularizedStep < target) {
				const next =
					comparison.unregularizedStep < 25
						? 25
						: (Math.floor(comparison.unregularizedStep / 100) + 1) * 100;
				await baseline.train(
					Math.min(target, next) - comparison.unregularizedStep,
					(metrics) => post({ id: req.id, event: 'metrics', metrics }),
					() => stopped
				);
				await checkpoint();
			}
			return comparison;
		} finally {
			baseline.dispose();
		}
	}
	throw new Error(`Unknown experiment operation: ${req.op}`);
}

self.onmessage = ({ data: req }: MessageEvent<Request>) => {
	if (req.op === 'stop') {
		stopped = true;
		post({ id: req.id, ok: true });
		return;
	}
	const next = queue.then(async () => {
		try {
			post({ id: req.id, ok: true, result: await handle(req) });
		} catch (error) {
			post({
				id: req.id,
				ok: false,
				error: error instanceof Error ? error.message : String(error)
			});
		}
	});
	queue = next.catch(() => {});
};
