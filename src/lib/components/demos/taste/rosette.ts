// The thing you have an opinion about.
//
// Six numbers in, one printer's rosette out — the kind of engraved device a
// book like this one would strike on a title page. Pure geometry, no state:
// the same gene always draws the same figure, which is what lets the chapter
// show you a whole field of them and then claim the field is a map of your
// taste.
//
// Two decisions carry the design. Every ornament is struck inside the same
// hairline medallion, so a comparison is about the figure and never about its
// size — the way a real specimen sheet is set. And the space is built so both
// of its ends are honest: somewhere in the middle are ornaments a person might
// actually choose, and out in one corner — every arm, the fullest bloom, the
// heaviest stroke — the petals overlap into a solid disc of ink. That corner
// is not a bug we failed to design out. It is the chapter's punchline, and the
// optimizer of Plate IV goes looking for it.

import { N_GENES, type Gene } from '$lib/optim-rl/preference';

export const VIEW = 100;
const CX = VIEW / 2;
const CY = VIEW / 2;
/** The medallion every ornament is struck inside. */
export const R_MAX = 44;

export interface Corolla {
	/** One closed outline per arm. */
	petals: string[];
	/** 0…1 — the inner ring is drawn lighter than the outer one. */
	alpha: number;
}

export interface RosetteShape {
	outer: Corolla;
	inner: Corolla | null;
	/** Concentric hairlines set inside the flower. */
	rings: number[];
	stroke: number;
	/** How solidly petals are filled — heavy strokes come with heavy fill. */
	fill: number;
	/** 0 = ultramarine, 1 = vermilion. */
	mix: number;
	hub: number;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** The gene, decoded into the quantities the drawing actually uses. */
export function decode(gene: Gene) {
	const g = (i: number) => clamp01(gene[i] ?? 0.5);
	return {
		arms: Math.round(lerp(4, 18, g(0))),
		curl: lerp(-1.25, 1.25, g(1)),
		reach: lerp(0.42, 0.96, g(2)),
		weight: lerp(0.35, 1, g(3)),
		bloom: g(4),
		mix: g(5)
	};
}

/** One ring of petals, swept along a spiral spine. */
function corolla(
	arms: number,
	inner: number,
	tip: number,
	half: number,
	curl: number,
	phase: number
): string[] {
	const span = Math.max(1e-3, tip - inner);
	// The spine bends more the further out it goes, so a curled petal is a
	// comma rather than a straight ray pointed sideways.
	const spine = (r: number) => phase + curl * Math.pow((r - inner) / span, 1.15) * 0.95;
	const pt = (r: number, a: number) =>
		`${(CX + r * Math.cos(a)).toFixed(2)},${(CY + r * Math.sin(a)).toFixed(2)}`;

	const r1 = inner + span * 0.3;
	const r2 = inner + span * 0.74;
	const out: string[] = [];
	for (let a = 0; a < arms; a++) {
		// −π/2 puts the first arm at twelve o'clock, so every rosette is upright
		const base = (a / arms) * Math.PI * 2 - Math.PI / 2;
		const s0 = base + spine(inner);
		const s1 = base + spine(r1);
		const s2 = base + spine(r2);
		const s3 = base + spine(tip);
		out.push(
			[
				`M ${pt(inner, s0)}`,
				`C ${pt(r1, s1 + half)} ${pt(r2, s2 + half * 0.52)} ${pt(tip, s3)}`,
				`C ${pt(r2, s2 - half * 0.52)} ${pt(r1, s1 - half)} ${pt(inner, s0)}`,
				'Z'
			].join(' ')
		);
	}
	return out;
}

export function rosette(gene: Gene): RosetteShape {
	if (gene.length !== N_GENES) throw new Error(`rosette: expected ${N_GENES} genes`);
	const d = decode(gene);

	const hub = lerp(2.4, 5.6, d.weight);
	const inner = hub + 1.6;
	const tip = R_MAX * d.reach;
	// Petals widen with the stroke, so heavy ink and many arms conspire into a
	// solid disc rather than merely a bolder outline.
	const half = (Math.PI / d.arms) * lerp(0.5, 1.32, d.weight);

	const outer: Corolla = {
		petals: corolla(d.arms, inner, tip, half, d.curl, 0),
		alpha: 1
	};

	// The second corolla: a rosette's characteristic inner crown, offset by
	// half a step so it reads between the outer petals rather than behind them.
	const inner2 =
		d.bloom > 0.14
			? {
					petals: corolla(
						d.arms,
						hub * 0.8,
						lerp(inner + 2, tip * 0.66, clamp01((d.bloom - 0.14) / 0.86)),
						half * 0.82,
						d.curl * 0.6,
						Math.PI / d.arms
					),
					alpha: lerp(0.35, 0.95, clamp01((d.bloom - 0.14) / 0.86))
				}
			: null;

	// A hairline set just outside the petals, when the bloom is full enough to
	// want a frame of its own.
	const rings: number[] = [];
	if (d.bloom > 0.62 && tip + 5 < R_MAX - 1) rings.push(tip + 4.5);
	if (d.bloom > 0.88 && tip + 9 < R_MAX - 1) rings.push(tip + 8.5);

	return {
		outer,
		inner: inner2,
		rings,
		stroke: lerp(0.55, 6.4, Math.pow(d.weight, 1.25)),
		fill: lerp(0.05, 0.72, Math.pow(d.weight, 1.5)),
		mix: d.mix,
		hub
	};
}

/**
 * The ornament's colour, in the book's own inks, so it re-tunes itself in dark
 * mode along with everything else.
 *
 * The gene runs ultramarine → ink → vermilion, and the route matters. Mixing
 * the two accents *directly* takes the short way round the hue wheel, through
 * violet and hot rose, and hands you a page of ornaments in colours this book
 * does not own. Routing through plain ink instead keeps every specimen inside
 * the house palette: most are struck in ink, as a real engraving would be,
 * and the two accents are what the extremes cost you.
 */
export function inkOf(mix: number, alpha = 1): string {
	const t = clamp01(mix) * 2 - 1; // −1 ultramarine · 0 ink · +1 vermilion
	// ^0.65 widens the tinted band, so the accents are not a rarity
	const strength = (Math.pow(Math.abs(t), 0.65) * 100).toFixed(1);
	const accent = t < 0 ? 'var(--accent)' : 'var(--warm)';
	const base = `color-mix(in oklab, ${accent} ${strength}%, var(--ink))`;
	return alpha >= 1
		? base
		: `color-mix(in srgb, ${base} ${(alpha * 100).toFixed(1)}%, transparent)`;
}

/**
 * How much ink a rosette spends, in [0,1] — used only to tell the reader, on
 * the over-optimization plate, how much of what the judge loves is blot.
 */
export function inkLoad(gene: Gene): number {
	const d = decode(gene);
	const petals = (d.arms / 18) * lerp(0.4, 1, d.weight) * d.reach;
	return clamp01((petals + d.weight + d.bloom * 0.7) / 2.7);
}
