// The plate registry — one ordered sequence of figures per chapter.
//
// Every figure in the book is a plate, whether it trains something or merely
// explains it, and its numeral comes from its position here rather than from a
// number typed at the call site. Prose refers to plates by id through
// <PlateRef>, so inserting a figure renumbers the chapter and every sentence
// that points into it, at once, correctly.

import { getContext, setContext } from 'svelte';

export const plateOrder: Record<string, readonly string[]> = {
	home: ['field'],
	descent: ['race', 'stepsize', 'flat'],
	neuron: ['neuron', 'atlas', 'workshop'],
	space: ['layer', 'rings', 'tangles'],
	digits: ['dataset', 'pipeline', 'softmax', 'classifier', 'drawpad', 'inside'],
	latent: ['autoencoder', 'squeeze', 'map', 'manifold', 'neighbors'],
	language: [
		'game',
		'skipgram',
		'vectors',
		'tokentree',
		'tokenizer',
		'scribe',
		'inspector',
		'transformer',
		'attention',
		'walkthrough'
	],
	reward: ['loop', 'chart', 'pendulum'],
	taste: ['balance', 'pairs', 'judge', 'goodhart', 'leash', 'fence'],
	rook: ['vocab', 'pretrain', 'probe', 'play', 'sft', 'rlvr', 'arena']
};

/** 1-based position of a plate in its chapter, or undefined if unregistered. */
export function plateNumber(chapter: string, id: string): number | undefined {
	const i = plateOrder[chapter]?.indexOf(id) ?? -1;
	return i < 0 ? undefined : i + 1;
}

export function roman(x: number): string {
	const table: Array<[number, string]> = [
		[10, 'X'],
		[9, 'IX'],
		[5, 'V'],
		[4, 'IV'],
		[1, 'I']
	];
	let out = '';
	let v = x;
	for (const [k, s] of table)
		while (v >= k) {
			out += s;
			v -= k;
		}
	return out;
}

/** "Plate VII" — the phrase prose and demo copy should use, never a literal. */
export function plateLabel(chapter: string, id: string): string {
	const n = plateNumber(chapter, id);
	return n === undefined ? 'the plate' : `Plate ${roman(n)}`;
}

/** The in-page anchor a plate publishes, so PlateRef can jump to it. */
export function plateAnchor(id: string): string {
	return `plate-${id}`;
}

// Which chapter we are reading. Set once by the page shell; every plate and
// every reference to one reads it, so no demo has to be told where it lives.
// Held as a getter so plates stay reactive when the shell swaps chapters.
const CHAPTER = Symbol('jaxverse:chapter');

export function setChapter(get: () => string): void {
	setContext(CHAPTER, get);
}

export function getChapter(): () => string {
	return getContext<() => string>(CHAPTER) ?? (() => '');
}
