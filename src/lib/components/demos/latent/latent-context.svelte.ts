// The chapter's one autoencoder, shared page-wide. Every plate offers the same
// transport in its own header, and the bottleneck-width toggle lives on Plate I
// ("the squeeze"); all three read the same weights live. Consumers re-derive
// their pixels from `tick`, which bumps once per trained chunk — the plates stay
// in sync without knowing about each other. Plate I disposes the lab when the
// page unmounts.

import { MlpEngine } from '$lib/nn/mlp-engine';
import type { Activation } from '$lib/nn/engine';
import { loadMnist, type MnistData } from '$lib/data/mnist';
import { progress } from '$lib/data/progress.svelte';
import { DIM, liftFrom, pcaBasis, projectAll, type Basis } from './common';

export type Depth = 1 | 2 | 3;

/** Waist widths on offer. Two and three can be looked at directly; past that
 * the map is a shadow of the space, and the reconstruction is the reward. */
export const LATENT_DIMS = [2, 3, 8, 16];
/** Above this the plates plot a projection instead of the coordinates. */
const VIEW_MAX = 3;

/** The encoder ladder per depth; the decoder mirrors it, so the whole
 * hourglass is [784, …ladder, D, …ladder reversed, 784]. */
const LADDER: Record<Depth, number[]> = {
	1: [256],
	2: [256, 64],
	3: [384, 128, 32]
};

export const layersFor = (d: number, depth: Depth): number[] => {
	const enc = LADDER[depth];
	return [DIM, ...enc, d, ...[...enc].reverse(), DIM];
};

/** activations() index of the bottleneck — one entry per hidden layer of the
 * encoder, so the waist lands at `depth`. Only tanh bounds it to (−1,1)^D;
 * with the modern activations the cloud is unbounded, which is why every plate
 * frames itself from `zBound` instead of a fixed extent. */
export const latentLayerFor = (depth: Depth): number => depth;
/** forwardFrom() entry point for the decoder half. */
export const decoderFromFor = (depth: Depth): number => depth + 1;

/** relu is deliberately absent: at a waist two units wide its dead half is
 * fatal — a unit that clips to zero takes half the map with it, and the whole
 * encoder collapses to the mean digit. The smooth rectifiers keep a little
 * negative signal and behave. */
export const ACTIVATIONS: Activation[] = ['gelu', 'silu', 'tanh'];

/** Held-out mse below this reads as "digits, not fog" — the milestone. Set for
 * the hardest shape on offer (a two-number variational waist), which pays a KL
 * tax a plain autoencoder does not. */
export const TRAINED_VAL = 0.055;

// 40-step chunks keep the contract's 25–50-step honest-eval cadence
const CHUNK = { steps: 40, sync: 4 };

/** Boot steps get a deadline. A worker whose GPU device never arrives simply
 * stops answering, and without this the plates sit on "warming up…" forever
 * instead of offering the reader a retry. */
function guard<T>(work: Promise<T>, message: string, ms = 25000): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error(message)), ms);
		work.then(
			(v) => {
				clearTimeout(timer);
				resolve(v);
			},
			(e) => {
				clearTimeout(timer);
				reject(e);
			}
		);
	});
}

function clamp01(a: Float32Array): Float32Array {
	// the read-out layer is linear; displays want honest 0..1 pixels
	for (let i = 0; i < a.length; i++) a[i] = a[i] <= 0 ? 0 : a[i] >= 1 ? 1 : a[i];
	return a;
}

/** The frame every plate draws in: where the cloud sits, and how far it
 * reaches. The prior does most of this work for us. A variational waist is
 * pulled toward N(0, I), so the centre is the origin — and a projected waist is
 * mean-subtracted by construction, so that one is centred too. Only the reach
 * has to be measured, and it is measured generously: the 99.9th percentile of
 * |coordinate| over every axis at once, which is the second-farthest digit in
 * the set. One reach serves all axes so the scale stays uniform, and with the
 * plate's own padding on top, the whole cloud lands inside the plot. */
function frameOf(z: Float32Array, d: number): { center: number[]; span: number } {
	const center = new Array(d).fill(0);
	const rows = Math.floor(z.length / d);
	if (rows === 0) return { center, span: 2.6 };
	const mag = new Float32Array(rows * d);
	for (let i = 0; i < rows * d; i++) mag[i] = Math.abs(z[i]);
	mag.sort();
	const reach = mag[Math.max(0, Math.floor(mag.length * 0.999) - 1)];
	return { center, span: Math.max(0.4, reach) };
}

/** Typical nearest-neighbour distance among the encodings, from a strided
 * sample. This is the yardstick for "has any real digit ever been near this
 * address?", and measuring it beats assuming it: it survives a waist that
 * grows, drifts, or changes width. */
function nnScale(z: Float32Array, rows: number, d: number): number {
	const stride = Math.max(1, Math.floor(rows / 96));
	let sum = 0;
	let cnt = 0;
	for (let p = 0; p < rows; p += stride) {
		let best = Infinity;
		for (let i = 0; i < rows; i++) {
			if (i === p) continue;
			let dd = 0;
			for (let c = 0; c < d; c++) {
				const t = z[i * d + c] - z[p * d + c];
				dd += t * t;
				if (dd >= best) break;
			}
			if (dd < best) best = dd;
		}
		if (best < Infinity) {
			sum += Math.sqrt(best);
			cnt++;
		}
	}
	return cnt ? Math.max(1e-4, sum / cnt) : 0.05;
}

/** Ease the frame toward the cloud so an unbounded waist that keeps growing
 * pans smoothly instead of jumping every chunk; snap on the first reading. */
function ease(prev: number, next: number, first: boolean): number {
	return first || !Number.isFinite(prev) ? next : prev * 0.75 + next * 0.25;
}

class LatentLab {
	phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	training = $state(false);
	/** True while the engine is re-initing for a new architecture. */
	rebuilding = $state(false);
	latentDim = $state(2);
	depth = $state<Depth>(2);
	activation = $state<Activation>('gelu');
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

	/** Bumped whenever `testZ` is replaced — the cue to re-read the cache. */
	zVersion = $state(0);

	/** Deliberately non-reactive: the worker handle and the raw pixels. */
	engine: MlpEngine | null = null;
	mnist: MnistData | null = null;
	/** The 2000 held-out digits in latent coordinates, and the true width of
	 * that buffer (a rebuild can land between request and response). Two plates
	 * want this every chunk; the cache makes them share one encode pass. */
	testZ: Float32Array | null = null;
	testZd = 0;
	/** What the plates actually plot: the waist itself while it is narrow enough
	 * to look at, and its two or three strongest directions once it is not. */
	viewZ: Float32Array | null = null;
	viewDim = 2;
	basis: Basis | null = null;
	/** Where the latent cloud sits and how far it reaches, eased over time. The
	 * variational prior keeps the centre at the origin and the reach near three
	 * sigma, but it takes a few hundred steps to get there — so the map, the
	 * cage and the address grid all take their frame from here rather than from
	 * a constant, and pan smoothly while it settles. */
	zCenter = $state<number[]>([0, 0]);
	zSpan = $state(2.6);
	/** Typical distance between neighbouring digits in the real waist. */
	zNn = $state(0.05);
	private xAll: Float32Array | null = null;
	private nAll = 0;
	private gen = 0;
	private loopId = 0;
	private zBusy = false;
	private zWant = false;

	private config() {
		return {
			layers: layersFor(this.latentDim, this.depth),
			activation: this.activation,
			loss: 'mse' as const,
			// The waist is variational: the encoder emits a mean and a spread,
			// the decoder is fed a sample, and the KL fee pins the whole cloud
			// to the unit Gaussian. That is what makes the map a fixed, bounded
			// place instead of something that drifts and sprawls as it trains —
			// and it is why the bottleneck itself carries no non-linearity.
			// beta = 1/784 is the textbook β = 1 in this engine's mean-mse units.
			// A wide waist gets a lighter fee: there the cloud is only ever seen
			// through a projection, so the prior is buying less, and the codes are
			// worth more to Plate IV when they keep a little more of the digit.
			vae: { at: latentLayerFor(this.depth), beta: (this.latentDim <= 3 ? 1 : 0.3) / DIM },
			// the rectified family tolerates a brisker step than tanh ever did,
			// and this plate is watched rather than left running. The deep ladder
			// gets a gentler one: three squeezes down to two numbers will happily
			// collapse into the mean digit and never come back out.
			lr: this.depth === 3 ? 1.2e-3 : 3e-3,
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
			const mnist = await guard(loadMnist(), 'the digits never arrived');
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
			await guard(
				this.engine.init(this.config(), { x, y: x, n }),
				'the trainer never answered — reload the page to give it a fresh worker'
			);
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
	setLatentDim(d: number): Promise<void> {
		if (d === this.latentDim) return Promise.resolve();
		this.latentDim = d;
		return this.rebuild();
	}

	/** More hidden layers each side of the waist — a longer squeeze. */
	setDepth(depth: Depth): Promise<void> {
		if (depth === this.depth) return Promise.resolve();
		this.depth = depth;
		return this.rebuild();
	}

	setActivation(a: Activation): Promise<void> {
		if (a === this.activation) return Promise.resolve();
		this.activation = a;
		return this.rebuild();
	}

	/** Rebuild on the live worker: fresh weights, fresh optimizer, phase kept. */
	private async rebuild(): Promise<void> {
		if (this.phase !== 'ready' || !this.engine || !this.xAll) return; // boot() will honor it
		const wasTraining = this.training;
		this.training = false;
		this.loopId++;
		const myGen = ++this.gen;
		this.rebuilding = true;
		try {
			await this.engine.stop();
			// init on the live worker resets params, optimizer and jit caches
			await this.engine.init(this.config(), { x: this.xAll, y: this.xAll, n: this.nAll });
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
		// fresh weights mean a fresh latent scale: drop the cache and let the
		// next encode snap the frame instead of easing from the old one
		this.testZ = null;
		this.testZd = 0;
		this.viewZ = null;
		this.basis = null;
		this.zVersion = 0;
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
		const k = latentLayerFor(this.depth);
		const acts = await this.engine.activations(x, n, 1024);
		return { z: acts.layers[k], d: acts.widths[k] };
	}

	/** Re-encode the held-out set into `testZ`. Safe to call from several plates
	 * on the same tick: concurrent calls collapse into one trailing pass. */
	async refreshTestLatents(): Promise<void> {
		if (this.zBusy) {
			this.zWant = true;
			return;
		}
		const m = this.mnist;
		if (!m || !this.engine || this.phase !== 'ready') return;
		this.zBusy = true;
		const myGen = this.gen;
		try {
			const rows = m.testY.length;
			const { z, d } = await this.encode(m.testX, rows);
			if (myGen === this.gen) {
				const wide = d > VIEW_MAX;
				const basis = wide ? pcaBasis(z, rows, d, VIEW_MAX) : null;
				const view = basis ? projectAll(z, rows, basis) : z;
				const vd = basis ? basis.k : d;
				const f = frameOf(view, vd);
				const first = this.zVersion === 0 || this.zCenter.length !== vd;
				this.testZ = z;
				this.testZd = d;
				this.viewZ = view;
				this.viewDim = vd;
				this.basis = basis;
				this.zCenter = f.center.map((c, k) => ease(this.zCenter[k], c, first));
				this.zSpan = ease(this.zSpan, f.span, first);
				this.zNn = ease(this.zNn, nnScale(z, rows, d), first);
				this.zVersion++;
			}
		} catch {
			// engine disposed or rebuilding mid-flight
		}
		this.zBusy = false;
		if (this.zWant) {
			this.zWant = false;
			void this.refreshTestLatents();
		}
	}

	/** Display coordinates → a real latent address the decoder can answer for.
	 * The identity while the waist is narrow enough to plot directly. */
	lift(u: ArrayLike<number>): Float32Array {
		const b = this.basis;
		if (!b) return Float32Array.from(u as ArrayLike<number>);
		return liftFrom(u, b);
	}

	/** Decoder alone: latent rows (n × d) → pixels (n × 784), display-clamped.
	 * chunk 16 keeps single-point cursor decodes cheap (jit pads per chunk);
	 * the manifold grid passes a bigger chunk for one dispatch. */
	async decode(z: Float32Array, n: number, chunk = 16): Promise<Float32Array> {
		if (!this.engine) throw new Error('engine is off');
		return clamp01(await this.engine.forwardFrom(decoderFromFor(this.depth), z, n, chunk));
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
		this.zCenter = [0, 0];
		this.zSpan = 2.6;
		this.phase = 'idle';
		this.clearStats();
		this.device = '';
		this.paramCount = 0;
		this.errorMsg = '';
		this.tick = 0;
	}
}

export const lab = new LatentLab();
