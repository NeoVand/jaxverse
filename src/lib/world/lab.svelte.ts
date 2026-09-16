import { WorldEngine, type Comparison } from './worker-engine';
import type {
	WorldInfo,
	WorldMetrics,
	WorldEvaluation,
	ReadoutMetrics,
	WorldForecastRequest,
	WorldForecast,
	WorldPlanRequest,
	WorldPlan
} from './engine';

export type WorldPhase =
	'idle' | 'loading' | 'ready' | 'training' | 'forecasting' | 'planning' | 'comparing' | 'error';

/** One page owns one model. The plates share weights, never a global worker. */
export class WorldLab {
	phase = $state<WorldPhase>('idle');
	error = $state('');
	info = $state.raw<WorldInfo | null>(null);
	backend = $state('');
	metrics = $state.raw<WorldMetrics | null>(null);
	evaluation = $state.raw<WorldEvaluation | null>(null);
	readout = $state.raw<ReadoutMetrics | null>(null);
	history = $state.raw<WorldMetrics[]>([]);
	comparison = $state.raw<Comparison | null>(null);
	comparisonStep = $state(0);
	training = $state(false);
	notice = $state('');
	motionOwner = $state<string | null>(null);
	checkpoint = $state(0);
	private engine: WorldEngine | null = null;
	private booting: Promise<void> | null = null;
	private queue: Promise<unknown> = Promise.resolve();
	private stopFlag = false;
	private disposed = false;
	private trainingTask: Promise<void> | null = null;
	private revision = 0;
	get step() {
		return this.metrics?.step ?? 0;
	}
	get busy() {
		return !['idle', 'ready', 'error'].includes(this.phase);
	}

	boot(): Promise<void> {
		if (this.disposed) return Promise.resolve();
		this.booting ??= this.initialize();
		return this.booting;
	}
	private async initialize() {
		const revision = ++this.revision;
		this.checkpoint++;
		this.phase = 'loading';
		this.error = '';
		const engine = new WorldEngine();
		this.engine = engine;
		try {
			const result = await engine.init();
			if (this.disposed || revision !== this.revision) return;
			this.info = result;
			this.backend = result.backend;
			this.phase = 'ready';
		} catch (error) {
			if (revision === this.revision) this.fail(error);
		}
	}
	private fail(error: unknown) {
		if (this.disposed) return;
		this.error = error instanceof Error ? error.message : String(error);
		this.phase = 'error';
	}
	private run<T>(job: () => Promise<T>): Promise<T> {
		const next = this.queue.then(job);
		this.queue = next.catch(() => {});
		return next;
	}
	async toggleTrain(): Promise<void> {
		if (this.training) return this.stop();
		if (this.busy || this.motionOwner) return;
		await this.boot();
		if (this.disposed || !this.engine || this.phase !== 'ready') return;
		this.trainingTask = this.trainLoop();
		await this.trainingTask;
	}
	private async trainLoop() {
		const revision = this.revision;
		this.training = true;
		this.stopFlag = false;
		this.phase = 'training';
		this.notice = '';
		const target = this.step + 5000;
		try {
			while (!this.stopFlag && !this.disposed) {
				await this.run(() =>
					this.engine!.train(Math.min(20, target - this.step), (metrics) => {
						if (!this.disposed && revision === this.revision) this.metrics = metrics;
					})
				);
				if (this.disposed || revision !== this.revision) return;
				if (this.metrics) {
					const history = [...this.history, this.metrics];
					this.history =
						history.length > 240
							? history.filter((_, i) => i % 2 === 0 || i === history.length - 1)
							: history;
				}
				this.readout = null;
				// A finite first session leaves the reader free to try predictions.
				if (this.step >= target) this.stopFlag = true;
				// Pausing can end between chunk boundaries. Measure elapsed updates,
				// and always measure the final weights, rather than relying on step % 100.
				if (this.step - (this.evaluation?.step ?? 0) >= 100 || this.stopFlag)
					this.evaluation = await this.run(() => this.engine!.evaluate());
			}
		} catch (error) {
			if (revision === this.revision) this.fail(error);
		} finally {
			if (!this.disposed && revision === this.revision) {
				this.training = false;
				if (this.phase === 'training') this.phase = 'ready';
			}
		}
	}
	async stop() {
		this.stopFlag = true;
		try {
			await this.engine?.stop();
		} catch (error) {
			this.fail(error);
		}
	}
	private async freeze() {
		await this.stop();
		await this.trainingTask;
		await this.boot();
		if (!this.engine || this.disposed || this.phase === 'error')
			throw new Error(this.error || 'The experiment is unavailable.');
	}
	async forecast(request: WorldForecastRequest): Promise<WorldForecast | null> {
		if (this.busy && !this.training) return null;
		try {
			await this.freeze();
			this.phase = 'forecasting';
			this.error = '';
			return await this.run(async () => {
				this.readout = await this.engine!.fitReadout();
				return this.engine!.forecast(request);
			});
		} catch (error) {
			this.fail(error);
			return null;
		} finally {
			if (!this.disposed && this.phase === 'forecasting') this.phase = 'ready';
		}
	}
	async plan(request: WorldPlanRequest): Promise<WorldPlan | null> {
		if (this.busy && !this.training) return null;
		try {
			await this.freeze();
			this.phase = 'planning';
			this.error = '';
			return await this.run(async () => {
				this.readout = await this.engine!.fitReadout();
				return this.engine!.plan(request);
			});
		} catch (error) {
			this.fail(error);
			return null;
		} finally {
			if (!this.disposed && this.phase === 'planning') this.phase = 'ready';
		}
	}
	async compare() {
		if (this.busy && !this.training) return;
		let revision = this.revision;
		try {
			await this.freeze();
			revision = this.revision;
			this.phase = 'comparing';
			this.comparisonStep = 0;
			this.comparison = null;
			this.notice = '';
			const result = await this.run(() =>
				this.engine!.compare(
					(metrics) => {
						if (!this.disposed && revision === this.revision) this.comparisonStep = metrics.step;
					},
					(comparison) => {
						if (!this.disposed && revision === this.revision) this.comparison = comparison;
					}
				)
			);
			if (this.disposed || revision !== this.revision) return;
			this.comparison = result;
			this.comparisonStep = result.unregularizedStep;
			if (result.status === 'paused')
				this.notice = `Comparison paused at ${result.unregularizedStep} of ${result.steps} updates. These measurements are preserved; comparing again starts a fresh baseline.`;
		} catch (error) {
			if (revision === this.revision) this.fail(error);
		} finally {
			if (!this.disposed && revision === this.revision && this.phase === 'comparing')
				this.phase = 'ready';
		}
	}
	async reset() {
		await this.stop();
		await this.trainingTask;
		if (!this.engine || this.disposed) return;
		this.phase = 'loading';
		this.checkpoint++;
		try {
			this.info = await this.run(() => this.engine!.reset());
			this.metrics = null;
			this.history = [];
			this.evaluation = null;
			this.comparison = null;
			this.comparisonStep = 0;
			this.readout = null;
			this.error = '';
			this.notice = '';
			this.phase = 'ready';
		} catch (error) {
			this.fail(error);
		}
	}
	async retry() {
		this.revision++;
		this.stopFlag = true;
		await this.engine?.dispose();
		await this.trainingTask;
		this.engine = null;
		this.booting = null;
		this.trainingTask = null;
		this.info = null;
		this.metrics = null;
		this.history = [];
		this.evaluation = null;
		this.comparison = null;
		this.comparisonStep = 0;
		this.readout = null;
		this.training = false;
		this.motionOwner = null;
		this.error = '';
		this.notice = '';
		return this.boot();
	}
	async dispose() {
		this.disposed = true;
		this.stopFlag = true;
		const engine = this.engine;
		this.engine = null;
		await engine?.dispose();
	}
}

export const createWorldLab = () => new WorldLab();
