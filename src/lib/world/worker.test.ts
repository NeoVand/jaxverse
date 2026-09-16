import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Comparison } from './worker-engine';

const fake = vi.hoisted(() => ({ instances: [] as FakeCore[], failAt: -1 }));

class FakeCore {
	step = 0;
	disposed = false;
	constructor(readonly options: { regularization?: number } = {}) {
		fake.instances.push(this);
	}
	async init() {
		return { step: this.step };
	}
	async train(steps: number, emit: (metrics: { step: number }) => void, stop: () => boolean) {
		for (let i = 0; i < steps && !stop(); i++) emit({ step: ++this.step });
		return { step: this.step };
	}
	async evaluate() {
		if (this.options.regularization === 0 && this.step === fake.failAt)
			throw new Error('Diagnostic failed');
		return { step: this.step, projection: [this.step], predictionLoss: 1 / (this.step + 1) };
	}
	async createUnregularizedBaseline() {
		return new FakeCore({ regularization: 0 });
	}
	dispose() {
		this.disposed = true;
	}
}

vi.mock('@jax-js/jax', () => ({ defaultDevice: vi.fn(), init: async () => ['cpu'] }));
vi.mock('./engine', () => ({
	get WorldCore() {
		return FakeCore;
	}
}));

interface Message {
	id: number;
	event?: 'metrics' | 'comparison';
	metrics?: { step: number };
	comparison?: Comparison;
	ok?: boolean;
	result?: Comparison;
	error?: string;
}
type Request = { id: number; op: string; steps?: number };

let messages: Message[];
let pending: Map<number, (message: Message) => void>;
let onPost: ((message: Message) => void) | null;
let scope: {
	onmessage: ((event: { data: Request }) => void) | null;
	postMessage: (message: Message) => void;
};

function request(data: Request): Promise<Message> {
	return new Promise((resolve) => {
		pending.set(data.id, resolve);
		scope.onmessage!({ data });
	});
}

beforeEach(async () => {
	vi.resetModules();
	fake.instances = [];
	fake.failAt = -1;
	messages = [];
	pending = new Map();
	onPost = null;
	scope = {
		onmessage: null,
		postMessage(message) {
			messages.push(structuredClone(message));
			onPost?.(message);
			if (!message.event) pending.get(message.id)?.(message);
		}
	};
	vi.stubGlobal('self', scope);
	await import('./worker');
	await request({ id: 1, op: 'init' });
	await request({ id: 2, op: 'train', steps: 250 });
});

afterEach(() => vi.unstubAllGlobals());

describe('streamed objective comparison', () => {
	it('streams actual initial and intermediate checkpoints without advancing the main model', async () => {
		const result = await request({ id: 3, op: 'compare' });
		const checkpoints = messages
			.filter((m) => m.id === 3 && m.event === 'comparison')
			.map((m) => m.comparison!);
		expect(checkpoints[0].unregularized).toBeNull();
		expect(checkpoints.slice(1).map((c) => c.unregularizedStep)).toEqual([0, 25, 100, 200, 250]);
		expect(checkpoints.every((c) => c.regularizedStep === 250)).toBe(true);
		expect(checkpoints.slice(1).every((c) => c.unregularized?.step === c.unregularizedStep)).toBe(
			true
		);
		expect(result.result?.status).toBe('complete');
		expect(fake.instances[0].step).toBe(250);
		expect(fake.instances[0].disposed).toBe(false);
		expect(fake.instances[1].disposed).toBe(true);
	});

	it('returns the last measured partial baseline on pause and starts the next comparison fresh', async () => {
		onPost = (message) => {
			if (message.id === 3 && message.event === 'metrics' && message.metrics?.step === 43)
				scope.onmessage!({ data: { id: 100, op: 'stop' } });
		};
		const paused = await request({ id: 3, op: 'compare' });
		expect(paused.ok).toBe(true);
		expect(paused.result).toMatchObject({
			status: 'paused',
			regularizedStep: 250,
			unregularizedStep: 43
		});
		expect(paused.result?.unregularized?.step).toBe(43);
		expect(fake.instances[1].disposed).toBe(true);
		onPost = null;
		const restarted = await request({ id: 4, op: 'compare' });
		const firstBaseline = messages.find(
			(m) => m.id === 4 && m.event === 'comparison' && m.comparison?.unregularized
		);
		expect(firstBaseline?.comparison?.unregularizedStep).toBe(0);
		expect(restarted.result?.status).toBe('complete');
		expect(fake.instances[0].step).toBe(250);
	});

	it('disposes a failed baseline while retaining earlier diagnostics and the main model', async () => {
		fake.failAt = 100;
		const failed = await request({ id: 3, op: 'compare' });
		expect(failed.ok).toBe(false);
		expect(failed.error).toBe('Diagnostic failed');
		expect(messages.some((m) => m.id === 3 && m.comparison?.unregularizedStep === 25)).toBe(true);
		expect(fake.instances[1].disposed).toBe(true);
		const main = await request({ id: 4, op: 'evaluate' });
		expect(main.result).toMatchObject({ step: 250 });
	});
});
