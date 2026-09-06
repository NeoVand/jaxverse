// Chapter 10's labs.
//
// Two of them, because two of this chapter's plates are arguments rather than
// demonstrations: they put the noise-predicting model of chapter 9 beside the
// velocity model of this one and let the reader read the difference off the
// page. The second worker is only booted by the plates that need it.

import { DiffusionLab } from '$lib/diffusion/lab.svelte';

export const lab = new DiffusionLab({
	objective: 'flow',
	checkpoint: 'emoji-flow.bin',
	batch: 32,
	lr: 3e-4
});

/** The chapter-9 model, for the two comparison plates. */
export const rival = new DiffusionLab({
	objective: 'eps',
	checkpoint: 'emoji-eps.bin',
	batch: 32,
	lr: 3e-4
});

export const SHOWN = 8;

/**
 * Prompts for the studio's opening state and its "surprise me" button.
 *
 * Every word here is in the 310-tag vocabulary — checked, because a suggested
 * prompt that the model cannot read would be the worst possible first
 * impression. "ghost", "rocket" and "rainbow" were all tried and dropped for
 * exactly that reason.
 */
export const SUGGESTIONS = [
	'smiling cat face',
	'red heart',
	'yellow star',
	'angry face',
	'fire',
	'moon face',
	'purple flower',
	'crying face',
	'green tree',
	'bird',
	'blue water',
	'dog face'
];

/** Pairs chosen because neither picture exists and both halves stay legible. */
export const COMBOS: [string, string][] = [
	// first because it is the cleanest fusion the model manages: a plain disc
	// from one prompt, a set of features from the other, and a face on the moon
	['moon', 'smiling face'],
	['cat face', 'heart'],
	['star', 'clock'],
	['bird', 'fire'],
	['dog face', 'flower'],
	['sun', 'angry face']
];
