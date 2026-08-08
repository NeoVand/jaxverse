/**
 * The one place canvas and WebGL code learns what color anything is.
 *
 * Everything drawn to a pixel buffer has to resolve the design tokens itself —
 * CSS variables mean nothing to a 2-D context — and four chapters had each
 * grown their own copy of this, with fallbacks that had drifted out of date.
 * There is one copy now, and the fallbacks below are the real values from
 * layout.css, so a token that fails to resolve still lands on the right hue.
 *
 * Colors are read at paint time, never cached: a theme switch bumps
 * `themePulse` and every painter re-reads through here.
 */

export type Rgb = [number, number, number];

/** Light-mode values from layout.css — the fallback if a token won't resolve. */
const FALLBACK: Record<string, string> = {
	'--paper': '#faf9f5',
	'--band': '#f6f4ec',
	'--surface': '#ffffff',
	'--surface-2': '#f3f1ea',
	'--line': '#e5e2d8',
	'--line-soft': '#efede4',
	'--ink': '#1d1c18',
	'--ink-2': '#605d54',
	'--ink-3': '#a3a094',
	'--accent': '#2b45d8',
	'--accent-strong': '#4a569b',
	'--warm': '#d3541f',
	'--good': '#22774d',
	'--bad': '#bb3a2b',
	'--cat-0': '#4457d4',
	'--cat-1': '#b0651c',
	'--cat-2': '#268577',
	'--cat-3': '#c14b6e',
	'--cat-4': '#6a6ae0',
	'--cat-5': '#8b7413',
	'--cat-6': '#23789f',
	'--cat-7': '#b8542f',
	'--cat-8': '#8a55c0',
	'--cat-9': '#4d8a37'
};

/** One design token, resolved against the element's live theme. */
export function token(el: Element, name: string): string {
	return getComputedStyle(el).getPropertyValue(name).trim() || FALLBACK[name] || '#888888';
}

export interface Tokens {
	paper: string;
	band: string;
	surface: string;
	surface2: string;
	ink: string;
	ink2: string;
	ink3: string;
	line: string;
	lineSoft: string;
	accent: string;
	accentStrong: string;
	warm: string;
	good: string;
	bad: string;
	/** The ten category hues, indexed by class. */
	cats: string[];
	/** The book's signed-value convention: ultramarine up, vermilion down. */
	pos: string;
	neg: string;
}

/** Every token a painter could want, resolved once per frame. */
export function readTokens(el: Element): Tokens {
	const s = getComputedStyle(el);
	const v = (name: string) => s.getPropertyValue(name).trim() || FALLBACK[name] || '#888888';
	return {
		paper: v('--paper'),
		band: v('--band'),
		surface: v('--surface'),
		surface2: v('--surface-2'),
		ink: v('--ink'),
		ink2: v('--ink-2'),
		ink3: v('--ink-3'),
		line: v('--line'),
		lineSoft: v('--line-soft'),
		accent: v('--accent'),
		accentStrong: v('--accent-strong'),
		warm: v('--warm'),
		good: v('--good'),
		bad: v('--bad'),
		cats: Array.from({ length: 10 }, (_, d) => v(`--cat-${d}`)),
		pos: v('--accent'),
		neg: v('--warm')
	};
}

/** Parse #rrggbb → [r, g, b]; tolerates whitespace. */
export function hexRgb(hex: string): Rgb {
	const h = hex.replace('#', '').trim();
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// ── theme reactivity ───────────────────────────────────────────────────────
// Canvases can't inherit a color change the way the DOM can, so they need to
// be told. Anything that paints reads `themePulse.tick` inside its effect and
// repaints when the class on <html> or the system preference changes.

class ThemePulse {
	tick = $state(0);
}

export const themePulse = new ThemePulse();

let watching = false;

/** Idempotent; safe to call from every plate (a no-op during prerender). */
export function watchTheme(): void {
	if (watching || typeof window === 'undefined') return;
	watching = true;
	const bump = () => {
		themePulse.tick += 1;
	};
	matchMedia('(prefers-color-scheme: dark)').addEventListener('change', bump);
	new MutationObserver(bump).observe(document.documentElement, {
		attributes: true,
		attributeFilter: ['class']
	});
}
