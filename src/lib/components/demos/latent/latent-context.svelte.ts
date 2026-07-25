// The chapter's one autoencoder, shared page-wide. Plate I ("the squeeze")
// owns Train/Reset and the bottleneck-width toggle; Plates II and III
// read the same weights live. Consumers re-derive their pixels from `tick`,
// which bumps once per trained chunk — the plates stay in sync without
// knowing about each other. Plate I disposes the lab when the page unmounts.

import { MlpEngine } from '$lib/nn/mlp-engine';
import { loadMnist, type MnistData } from '$lib/data/mnist';
import { progress } from '$lib/data/progress.svelte';
import { DIM } from './common';

/** The hourglass: 784 → 128 → 32 → D → 32 → 128 → 784, tanh between layers. */
export const layersFor = (d: number): number[] => [DIM, 128, 32, d, 32, 128, DIM];
/** activations() index of the bottleneck (post-tanh, so (−1,1)^D). */
export const LATENT_LAYER = 2;
/** forwardFrom() entry point for the decoder half: weight matrices 3, 4, 5. */
export const DECODER_FROM = 3;
/** Held-out mse below this reads as "digits, not fog" — the milestone. */
export const TRAINED_VAL = 0.045;

// 40-step chunks keep the contract's 25–50-step honest-eval cadence
const CHUNK = { steps: 40, sync: 4 };

function clamp01(a: Float32Array): Float32Array {
	// the read-out layer is linear; displays want honest 0..1 pixels
	for (let i = 0; i < a.length; i++) a[i] = a[i] <= 0 ? 0 : a[i] >= 1 ? 1 : a[i];
	return a;
}

class LatentLab {
	phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	training = $state(false);
	/** True while the engine is re-initing for a new bottleneck width. */
	rebuilding = $state(false);
	latentDim = $state<2 | 3>(2);
	step = $state(0);
	lossNow = $state(NaN);
	valLoss = $state(NaN);
	msPerStep = $state(0);
	device = $state('');
	paramCount = $state(0);
	errorMsg = $state('');
	lossHist = $state<number[]>([]);
	valHist = $state<number[]>([]);
	/** Bumped after every trained chunk, reset, and rebuild — the refresh cue. */
	tick = $state(0);
	trained = $state(false);

	/** Deliberately non-reactive: the worker handle and the raw pixels. */
	engine: MlpEngine | null = null;
	mnist: MnistData | null = null;
	private xAll: Float32Array | null = null;
	private nAll = 0;
	private gen = 0;
	private loopId = 0;

	private config(d: number) {
		return {
			layers: layersFor(d),
			activation: 'tanh' as const,
			loss: 'mse' as const,
			lr: 2e-3,
			batchSize: 128,
			seed: 7,
			valFraction: 0.2
		};
	}

	/** Auto-load target for use:inview — safe to call from every plate. */
	async boot(): Promise<void> {
		if (this.phase === 'loading' || this.phase === 'ready') return;
		const myGen = ++this.gen;
		this.phase = 'loading';
		this.errorMsg = '';
		try {
			const mnist = await loadMnist();
			if (myGen !== this.gen) return;
			this.mnist = mnist;
			const nTrain = mnist.trainY.length;
			const n = nTrain + mnist.testY.length;
			// The whole self-supervision trick is one line of data plumbing:
			// the target buffer IS the input buffer. valFraction 0.2 makes the
			// held-out tail exactly the 2000 test digits.
			const x = new Float32Array(n * DIM);
			x.set(mnist.trainX, 0);
			x.set(mnist.testX, nTrain * DIM);
			this.xAll = x;
			this.nAll = n;
			void this.engine?.dispose();
			this.engine = new MlpEngine();
			await this.engine.init(this.config(this.latentDim), { x, y: x, n });
			if (myGen !== this.gen) return;
			this.device = this.engine.device;
			this.paramCount = this.engine.paramCount;
			this.clearStats();
			this.phase = 'ready';
		} catch (err) {
			if (myGen !== this.gen) return;
			this.phase = 'error';
			this.errorMsg = err instanceof Error ? err.message : String(err);
		}
	}

	/** Re-init with a wider or narrower waist: fresh weights, phase preserved. */
	async setLatentDim(d: 2 | 3): Promise<void> {
		if (d === this.latentDim || this.rebuilding) return;
		this.latentDim = d;
		if (this.phase !== 'ready' || !this.engine || !this.xAll) return; // boot() will honor it
		const wasTraining = this.training;
		this.training = false;
		this.loopId++;
		const myGen = ++this.gen;
		this.rebuilding = true;
		try {
			await this.engine.stop();
			// init on the live worker resets params, optimizer and jit caches
			await this.engine.init(this.config(d), { x: this.xAll, y: this.xAll, n: this.nAll });
		} catch (err) {
			if (myGen === this.gen) {
				this.rebuilding = false;
				this.phase = 'error';
				this.errorMsg = err instanceof Error ? err.message : String(err);
			}
			return;
		}
		if (myGen !== this.gen) return;
		this.rebuilding = false;
		this.paramCount = this.engine.paramCount;
		this.clearStats();
		this.tick++;
		if (wasTraining) this.setTraining(true);
	}

	private clearStats(): void {
		this.step = 0;
		this.lossNow = NaN;
		this.valLoss = NaN;
		this.msPerStep = 0;
		this.lossHist = [];
		this.valHist = [];
		this.trained = false;
	}

	setTraining(on: boolean): void {
		if (on && !this.training && this.phase === 'ready' && !this.rebuilding) {
			this.training = true;
			void this.trainLoop(++this.loopId);
		} else if (!on) {
			this.training = false;
		}
	}

	private async trainLoop(id: number): Promise<void> {
		const myGen = this.gen;
		while (this.training && this.loopId === id && this.engine && myGen === this.gen) {
			try {
				const { steps, sync } = CHUNK;
				await this.engine.train(
					steps,
					(m) => {
						if (myGen !== this.gen) return;
						this.step = m.step;
						this.lossNow = m.loss;
						this.msPerStep = this.msPerStep ? this.msPerStep * 0.7 + m.stepMs * 0.3 : m.stepMs;
						this.lossHist = [...this.lossHist.slice(-239), m.loss];
					},
					sync
				);
				if (myGen !== this.gen || this.loopId !== id || !this.engine) return;
				const ev = await this.engine.eval();
				if (myGen !== this.gen || this.loopId !== id) return;
				this.valLoss = ev.loss;
				this.valHist = [...this.valHist.slice(-239), ev.loss];
				if (!this.trained && ev.loss < TRAINED_VAL) {
					this.trained = true;
					progress.reach('latent:trained');
				}
				this.tick++;
			} catch {
				return; // engine disposed mid-flight — the page is leaving
			}
		}
	}

	async resetWeights(): Promise<void> {
		if (!this.engine || this.phase !== 'ready' || this.rebuilding) return;
		const wasTraining = this.training;
		this.training = false;
		const myGen = this.gen;
		try {
			await this.engine.stop();
			await this.engine.reset(Math.floor(Math.random() * 1e9));
		} catch {
			return;
		}
		if (myGen !== this.gen) return;
		this.clearStats();
		this.tick++;
		if (wasTraining) this.setTraining(true);
	}

	/** Latent coordinates (n × d): the bottleneck layer of activations().
	 * The returned width is the truth — callers must not assume latentDim,
	 * because a rebuild can land between request and response. */
	async encode(x: Float32Array, n: number): Promise<{ z: Float32Array; d: number }> {
		if (!this.engine) throw new Error('engine is off');
		const acts = await this.engine.activations(x, n, 1024);
		return { z: acts.layers[LATENT_LAYER], d: acts.widths[LATENT_LAYER] };
	}

	/** Decoder alone: latent rows (n × d) → pixels (n × 784), display-clamped.
	 * chunk 16 keeps single-point cursor decodes cheap (jit pads per chunk);
	 * the manifold grid passes a bigger chunk for one dispatch. */
	async decode(z: Float32Array, n: number, chunk = 16): Promise<Float32Array> {
		if (!this.engine) throw new Error('engine is off');
		return clamp01(await this.engine.forwardFrom(DECODER_FROM, z, n, chunk));
	}

	/** Full round trip x → D(E(x)), display-clamped. */
	async reconstruct(x: Float32Array, n: number): Promise<Float32Array> {
		if (!this.engine) throw new Error('engine is off');
		return clamp01(await this.engine.predict(x, n, 16));
	}

	/** Page unmount: kill the worker, return to the unloaded state. */
	dispose(): void {
		this.gen++;
		this.loopId++;
		this.training = false;
		this.rebuilding = false;
		void this.engine?.dispose();
		this.engine = null;
		this.xAll = null;
		this.nAll = 0;
		this.phase = 'idle';
		this.clearStats();
		this.device = '';
		this.paramCount = 0;
		this.errorMsg = '';
		this.tick = 0;
	}
}

export const lab = new LatentLab();
