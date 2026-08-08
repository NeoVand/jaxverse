// The shared bench for chapter 3: the classifier plate (the classifier) owns and trains
// one MlpEngine; Plates IV and V borrow the same weights through this module.
// Module-level on purpose — the plates live on one page and must see one model.

import type { MlpEngine } from '$lib/nn/mlp-engine';
import type { Activation } from '$lib/nn/engine';
import type { MnistData } from '$lib/data/mnist';

export type LabPhase = 'idle' | 'loading' | 'ready' | 'error';

class DigitsLab {
	/** Worker handle — a plain field on purpose (opaque, never templated). */
	engine: MlpEngine | null = null;
	/** Decoded dataset — plain field too; large, and read only on demand. */
	mnist: MnistData | null = null;

	/** Architecture, owned by the classifier plate and read by the plates downstream. */
	depth = $state(2);
	width = $state(128);
	activation = $state<Activation>('relu');

	phase = $state<LabPhase>('idle');
	step = $state(0);
	loss = $state(NaN);
	/** Held-out accuracy — with the test rows as the engine's validation tail,
	 * this IS test-set accuracy. */
	testAcc = $state(NaN);
	/** Accuracy on a fixed sample of the rows it does train on; the shortfall
	 * against testAcc is the memorization it got away with. */
	trainAcc = $state(NaN);
	msPerStep = $state(0);
	device = $state('');
	paramCount = $state(0);
	/** Bumped whenever the weights meaningfully changed (after each eval'd
	 * train chunk, after reset) — dependent plates re-read the model on it. */
	version = $state(0);

	/** Back to the powered-off state; the owner disposes the engine itself. */
	clear() {
		this.engine = null;
		this.mnist = null;
		this.phase = 'idle';
		this.step = 0;
		this.loss = NaN;
		this.testAcc = NaN;
		this.trainAcc = NaN;
		this.msPerStep = 0;
		this.device = '';
		this.paramCount = 0;
		this.version = 0;
	}
}

export const lab = new DigitsLab();

// Canvas colors are read from CSS at draw time; bumping `tick` on theme flips
// makes draw effects re-run, so static tiles never keep stale ink.
export { themePulse, watchTheme } from '$lib/viz/tokens.svelte';
