// Chapter 9's lab: one worker, the noise-predicting model, shared by the
// training plate and the trajectory plate. The plate that owns the transport
// is the one that disposes it.

import { DiffusionLab } from '$lib/diffusion/lab.svelte';

export const lab = new DiffusionLab({
	objective: 'eps',
	checkpoint: 'fashion-eps.bin',
	batch: 32,
	lr: 3e-4,
	// far short of a finished model, but far enough that the top row of the
	// training plate has stopped being static and started being garment-shaped
	milestone: { at: 1500, key: 'noise:drew' }
});

/** Eight pictures is what fits across a plate without going illegibly small. */
export const SHOWN = 8;
