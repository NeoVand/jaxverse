// The book's one sparkline. Every slim telemetry strip — SpaceLab, the digits
// classifier, the latent squeeze, and any trainer that grows one later — draws
// its curves through here, so the log floor, the 2px breathing room, and the
// path style stay locked across chapters.

export interface SparkOpts {
	/** Compress through log space — the right scale for losses. */
	log?: boolean;
	/** Log floor: values are clamped up to this before the log. */
	floor?: number;
	/** Shared y-range (from sparkSpan) when several series must ride one axis. */
	lo?: number;
	hi?: number;
}

/** SVG path for a slim sparkline: values laid out evenly across `w`, drawn
 * with a 2px vertical inset so the stroke never kisses the strip's edges. */
export function sparkPath(vals: number[], w: number, h: number, opts: SparkOpts = {}): string {
	if (vals.length < 2) return '';
	const { log = false, floor = 1e-4 } = opts;
	const t = (v: number) => (log ? Math.log(Math.max(v, floor)) : v);
	let lo = opts.lo ?? Infinity;
	let hi = opts.hi ?? -Infinity;
	if (opts.lo === undefined || opts.hi === undefined) {
		lo = Infinity;
		hi = -Infinity;
		for (const v of vals) {
			const y = t(v);
			if (y < lo) lo = y;
			if (y > hi) hi = y;
		}
	}
	if (hi - lo < 1e-9) hi = lo + 1e-9;
	return vals
		.map((v, i) => {
			const x = (i / (vals.length - 1)) * w;
			const y = h - ((t(v) - lo) / (hi - lo)) * (h - 4) - 2;
			return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
		})
		.join(' ');
}

/** Shared log-space range across several series — one honest y-axis. */
export function sparkSpan(series: number[][], floor = 1e-6): [number, number] {
	let lo = Infinity;
	let hi = -Infinity;
	for (const vals of series)
		for (const v of vals) {
			const y = Math.log(Math.max(v, floor));
			if (y < lo) lo = y;
			if (y > hi) hi = y;
		}
	if (!Number.isFinite(lo)) return [0, 1];
	if (hi - lo < 1e-9) hi = lo + 1e-9;
	return [lo, hi];
}
