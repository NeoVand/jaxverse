// One lab per chapter: it owns the worker, the training loop and the numbers
// the plates read. Several plates share a lab, so booting is idempotent and
// only the plate that owns the transport disposes it.

import { base } from '$app/paths';
import { progress } from '$lib/data/progress.svelte';
import { DiffusionEngine, type CorpusInfo } from './worker-engine';
import { detectWebGPU, type Phase, type SampleOptions, type SampleResult } from './engine';
import type { Objective } from './model';

/** One house pace, tuned for reading: long enough to make visible progress
 *  between refreshes, short enough that Pause feels immediate. */
const CHUNK = 40;

/**
 * Wrap a redraw so that requests arriving mid-flight are not lost.
 *
 * Sliders fire a change per pixel of travel and a sample takes a few hundred
 * milliseconds, so a plain busy-flag drops every request but the first —
 * including the last one, which leaves the plate showing a picture that does
 * not match its own controls. This runs the job again, once, if anything asked
 * while it was busy.
 */
export function coalesce(job: () => Promise<void>): () => void {
	let running = false;
	let again = false;
	const go = async () => {
		if (running) {
			again = true;
			return;
		}
		running = true;
		try {
			await job();
		} finally {
			running = false;
		}
		if (again) {
			again = false;
			void go();
		}
	};
	return () => void go();
}

export interface LabOptions {
	objective: Objective;
	/** static/data filename of the shipped checkpoint, if the chapter has one. */
	checkpoint?: string;
	/** Start the live weights from the shipped ones rather than from noise. */
	warmStart?: boolean;
	batch?: number;
	lr?: number;
	/** Recorded once the reader's own weights pass `at` steps. */
	milestone?: { at: number; key: string };
}

export class DiffusionLab {
	phase = $state<Phase>('idle');
	error = $state('');
	step = $state(0);
	loss = $state(NaN);
	stepMs = $state(0);
	imagesPerSec = $state(0);
	training = $state(false);
	/** Downsampled for the sparkline; the full history is never needed. */
	lossHist = $state<number[]>([]);
	info = $state<CorpusInfo | null>(null);
	hasShipped = $state(false);
	/** Bumped after every training chunk so plates know to redraw. */
	tick = $state(0);

	/** Milestone to record the first time this lab trains past `at` steps. */
	private milestone: { at: number; key: string } | null = null;

	private engine: DiffusionEngine | null = null;
	private booting: Promise<void> | null = null;
	private stopFlag = false;
	private queue: Promise<unknown> = Promise.resolve();

	constructor(private opts: LabOptions) {
		this.milestone = opts.milestone ?? null;
	}

	boot(): Promise<void> {
		this.booting ??= this.doBoot();
		return this.booting;
	}

	private async doBoot() {
		this.phase = 'loading';
		try {
			if (!(await detectWebGPU())) {
				this.phase = 'no-webgpu';
				return;
			}
			let checkpoint: ArrayBuffer | undefined;
			if (this.opts.checkpoint) {
				const res = await fetch(`${base}/data/${this.opts.checkpoint}`);
				if (res.ok) checkpoint = await res.arrayBuffer();
			}
			const engine = new DiffusionEngine();
			this.info = await engine.init(base, {
				objective: this.opts.objective,
				checkpoint,
				batch: this.opts.batch ?? 32,
				lr: this.opts.lr ?? 3e-4
			});
			this.hasShipped = this.info.hasShipped;
			this.engine = engine;
			this.phase = 'ready';
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			this.phase = 'error';
		}
	}

	/** Serialize GPU work: the worker is one device and sampling mid-step
	 *  would interleave two jit caches over the same weights. */
	private run<T>(job: () => Promise<T>): Promise<T> {
		const next = this.queue.then(job, job);
		this.queue = next.catch(() => {});
		return next;
	}

	async toggleTrain(): Promise<void> {
		if (this.training) {
			this.stopFlag = true;
			await this.engine?.stop();
			return;
		}
		await this.boot();
		if (this.phase !== 'ready' || !this.engine) return;
		this.training = true;
		this.stopFlag = false;
		this.phase = 'training';
		try {
			while (!this.stopFlag) {
				await this.run(() =>
					this.engine!.train(CHUNK, (m) => {
						this.step = m.step;
						this.loss = Number.isNaN(this.loss) ? m.loss : this.loss * 0.95 + m.loss * 0.05;
						this.stepMs = m.stepMs;
						this.imagesPerSec = m.imagesPerSec;
					})
				);
				this.lossHist = [...this.lossHist, this.loss].slice(-240);
				this.tick++;
				if (this.milestone && this.step >= this.milestone.at) {
					progress.reach(this.milestone.key);
					this.milestone = null;
				}
			}
		} catch (e) {
			this.error = e instanceof Error ? e.message : String(e);
			this.phase = 'error';
		} finally {
			this.training = false;
			if (this.phase === 'training') this.phase = 'ready';
		}
	}

	async reset(seed = 20260905): Promise<void> {
		this.stopFlag = true;
		await this.engine?.stop();
		await this.run(async () => {
			await this.engine?.reset(seed);
		});
		this.step = 0;
		this.loss = NaN;
		this.lossHist = [];
		this.tick++;
	}

	sample(count: number, opts: SampleOptions): Promise<SampleResult> {
		return this.run(async () => {
			await this.boot();
			if (!this.engine) throw new Error('no engine');
			return this.engine.sample(count, opts);
		});
	}

	tiles(indices: number[], style: number): Promise<Float32Array> {
		return this.run(async () => {
			await this.boot();
			if (!this.engine) throw new Error('no engine');
			return this.engine.tiles(indices, style);
		});
	}

	async dispose(): Promise<void> {
		this.stopFlag = true;
		const e = this.engine;
		this.engine = null;
		this.booting = null;
		this.phase = 'idle';
		await e?.dispose();
	}
}
