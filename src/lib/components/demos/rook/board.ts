// One chessboard for the whole chapter, configured once.
//
// The board itself is cm-chessboard (SVG, Staunty pieces, markers and arrows as
// extensions); everything below is the book's side of the contract: where the
// sprites live, and the semantic marker/arrow vocabulary the plates speak in.
// The tones map to CSS classes rather than colors, because the palette lives in
// layout.css and follows the theme — see the `.cm-chessboard.jaxverse` block.

import { base } from '$app/paths';
import type { ArrowType, MarkerType } from 'cm-chessboard/src/Chessboard.js';

/** The sprite folder, copied from the cm-chessboard package into static/. */
export const BOARD_ASSETS = `${base}/chessboard/`;

/**
 * cm-chessboard touches `document` as soon as a board is built, so it is loaded
 * on demand rather than imported at the top of a component — which also keeps
 * it out of the bundle of every chapter that never draws a board. One shared
 * promise: a page with eight boards still loads the library once.
 */
type BoardModules = typeof import('cm-chessboard/src/Chessboard.js') &
	typeof import('cm-chessboard/src/extensions/markers/Markers.js') &
	typeof import('cm-chessboard/src/extensions/arrows/Arrows.js') &
	typeof import('cm-chessboard/src/extensions/accessibility/Accessibility.js');

let pending: Promise<BoardModules> | null = null;

export function loadBoardLib(): Promise<BoardModules> {
	pending ??= Promise.all([
		import('cm-chessboard/src/Chessboard.js'),
		import('cm-chessboard/src/extensions/markers/Markers.js'),
		import('cm-chessboard/src/extensions/arrows/Arrows.js'),
		import('cm-chessboard/src/extensions/accessibility/Accessibility.js')
	]).then(([core, markers, arrows, a11y]) => ({ ...core, ...markers, ...arrows, ...a11y }));
	return pending;
}

/** How the book paints a square. */
export type BoardTone = 'accent' | 'warm' | 'bad' | 'good' | 'violet';

/**
 * Marker types are compared by identity inside cm-chessboard, so these must be
 * module-level constants — never rebuilt per render.
 */
export const MARK: {
	wash: Record<BoardTone, MarkerType>;
	selected: MarkerType;
	check: MarkerType;
} = {
	wash: {
		accent: { class: 'jx-wash jx-accent', slice: 'markerSquare' },
		warm: { class: 'jx-wash jx-warm', slice: 'markerSquare' },
		bad: { class: 'jx-wash jx-bad', slice: 'markerSquare' },
		good: { class: 'jx-wash jx-good', slice: 'markerSquare' },
		violet: { class: 'jx-wash jx-violet', slice: 'markerSquare' }
	},
	selected: { class: 'jx-selected', slice: 'markerFrame' },
	check: { class: 'jx-check', slice: 'markerSquare' }
};

export const ARROW: Record<BoardTone, ArrowType> = {
	accent: { class: 'jx-arrow jx-accent' },
	warm: { class: 'jx-arrow jx-warm' },
	bad: { class: 'jx-arrow jx-bad' },
	good: { class: 'jx-arrow jx-good' },
	violet: { class: 'jx-arrow jx-violet' }
};

/** Every declarative marker, for a clean sweep before repainting. (The
 * legal-move dots are the Markers extension's own, cleared with its call.) */
export const ALL_MARKS: MarkerType[] = [...Object.values(MARK.wash), MARK.selected, MARK.check];

/** A square name, or null when the string isn't one ('e2' from 'e2e4'). */
export function square(uci: string, end: 'from' | 'to'): string | null {
	const sq = end === 'from' ? uci.slice(0, 2) : uci.slice(2, 4);
	if (sq.length !== 2) return null;
	const file = sq.charCodeAt(0) - 97;
	const rank = Number(sq[1]);
	return file >= 0 && file <= 7 && rank >= 1 && rank <= 8 ? sq : null;
}
