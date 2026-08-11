// One glyph per chapter. Purely presentational, but shared: the contents rail
// and the header bar have to agree, or the same chapter wears two faces.

import {
	BookOpen,
	Castle,
	Gamepad2,
	Grid3x3,
	Map,
	Mountain,
	PenLine,
	Scale,
	Spline,
	Type
} from 'lucide-svelte';
import type { ChapterSlug } from './chapters';

export type Glyph = typeof Mountain;

export const chapterGlyphs: Record<ChapterSlug, Glyph> = {
	descent: Mountain,
	neuron: Spline,
	space: Grid3x3,
	digits: PenLine,
	latent: Map,
	language: Type,
	reward: Gamepad2,
	taste: Scale,
	rook: Castle
};

/** The epilogue closes the book, so it gets the book. */
export const epilogueGlyph: Glyph = BookOpen;
