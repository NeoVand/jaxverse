// Chapter 10's labs.
//
// Two of them, because two of this chapter's plates are arguments rather than
// demonstrations: they put the noise-predicting model of chapter 9 beside the
// velocity model of this one and let the reader read the difference off the
// page. The second worker is only booted by the plates that need it.

import { DiffusionLab } from '$lib/diffusion/lab.svelte';

export const lab = new DiffusionLab({
	objective: 'flow',
	checkpoint: 'fashion-flow.bin',
	batch: 32,
	lr: 3e-4
});

/** The chapter-9 model, for the two comparison plates. */
export const rival = new DiffusionLab({
	objective: 'eps',
	checkpoint: 'fashion-eps.bin',
	batch: 32,
	lr: 3e-4
});

export const SHOWN = 8;

/**
 * Pairs for the morph plate, by label index.
 *
 * Chosen so that both ends are legible at 28 pixels and the halfway point is
 * genuinely ambiguous rather than simply one of the two: a sneaker and a boot
 * share a sole, a shirt and a dress share a torso, a bag and a sandal share
 * almost nothing, which is the interesting one.
 */
export const MORPHS: [number, number][] = [
	[7, 9], // sneaker → ankle boot
	[0, 3], // t-shirt → dress
	[2, 4], // pullover → coat
	[8, 5], // bag → sandal
	[1, 3], // trouser → dress
	[6, 2] // shirt → pullover
];
