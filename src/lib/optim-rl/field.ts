// Where a practice hall stands, and why that is a claim rather than a layout.
//
// The swarm plate draws every hall as a live pendulum standing somewhere in a
// field of two skills. That only means anything if the position is the data —
// which is harder than it sounds, because the halls converge. Six trained
// halls differ by a couple of percent, and a couple of percent is smaller
// than the pendulums being drawn. Something has to give, and the whole
// question is WHAT.
//
// The first version gave up the data: it ran a few rounds of ordinary
// collision repulsion, pushing overlapping rigs apart in whichever direction
// separated them. Once the pool converged that repulsion was larger than the
// differences it was drawing, so the picture became noise wearing the costume
// of a measurement — a hall delivering 0.98 could sit to the LEFT of one
// delivering 0.94 because the solver happened to resolve it that way. That is
// the worst kind of chart bug: it does not look broken.
//
// What gives here instead is the one degree of freedom the plot makes no
// claim about. Fitness is a straight line through this field —
// deliver + ½·catch — so there is a direction along which every point has
// exactly the same fitness. Overlapping halls are slid along THAT, and only
// that. Their fitness ordering is preserved to the last bit, because sliding
// along a level set cannot change the level. What the fan says is true too:
// halls spread out along it are equally good and differently skilled.

/** How much the catch half is worth next to the swing half, in the fitness
 * the pool actually races on. Must match the worker's `score`. */
export const CATCH_WEIGHT = 0.5;

/** The number every hall is ranked by: can it get the stack up, and can it
 * keep it there. Both halves in one line, and the line is drawable. */
export function fitness(deliver: number, catchRate: number): number {
	return deliver + CATCH_WEIGHT * catchRate;
}

export interface Camera {
	/** Data → screen, for both axes. */
	x: (v: number) => number;
	y: (v: number) => number;
	/** The screen-space unit vector along which fitness does not change. */
	iso: readonly [number, number];
}

/**
 * Build the mapping and, with it, the one safe direction to move in.
 *
 * The level set of `deliver + w·catch` runs along data-space (−w, 1). Screen
 * y is flipped, so that becomes (−w·sx, −sy) on the canvas — which is what
 * `separate` slides along, and the reason it can slide at all.
 */
export function camera(
	view: readonly number[],
	box: { left: number; right: number; top: number; bottom: number },
	w = CATCH_WEIGHT
): Camera {
	const [x0, x1, y0, y1] = view;
	const sx = (box.right - box.left) / Math.max(1e-6, x1 - x0);
	const sy = (box.bottom - box.top) / Math.max(1e-6, y1 - y0);
	const ix = -w * sx;
	const iy = -sy;
	const len = Math.hypot(ix, iy) || 1;
	return {
		x: (v: number) => box.left + (v - x0) * sx,
		y: (v: number) => box.bottom - (v - y0) * sy,
		iso: [ix / len, iy / len]
	};
}

/**
 * Fan overlapping points apart along `iso`, in place.
 *
 * Only the component along `iso` is ever written, so every point's fitness
 * survives exactly. `minD` is the separation asked for; the return value is
 * the separation actually used, which is smaller when the pool is too
 * crowded for the room available — reporting it rather than silently
 * overflowing lets the caller shrink what it is drawing instead.
 */
export function separate(
	px: Float64Array,
	py: Float64Array,
	n: number,
	minD: number,
	iso: readonly [number, number],
	window: readonly [number, number]
): number {
	if (n < 2) return minD;
	const [tLo, tHi] = window;
	// n points fanned single-file need (n−1)·d of rung. Asking for more than
	// exists is how points end up clamped, and a clamp is a lie.
	const d = Math.min(minD, (tHi - tLo) / Math.max(1, n - 1));
	if (!(d > 0)) return 0;
	// Position along iso is the only coordinate in play, so the whole thing
	// is one-dimensional: project, spread, write back. One sorted sweep
	// solves it exactly — iterative relaxation was tried first and needs
	// unbounded rounds to pull apart points that start coincident, which is
	// precisely the converged-pool case this exists for.
	const t = new Float64Array(n);
	const bx = new Float64Array(n);
	const by = new Float64Array(n);
	let mid = 0;
	for (let i = 0; i < n; i++) {
		t[i] = px[i] * iso[0] + py[i] * iso[1];
		bx[i] = px[i] - t[i] * iso[0];
		by[i] = py[i] - t[i] * iso[1];
		mid += t[i] / n;
	}
	const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => t[a] - t[b]);
	// `max` and not `set`: halls already far enough apart are left exactly
	// where the data put them, and only the crowd is opened up.
	for (let k = 1; k < n; k++) {
		const a = order[k - 1];
		const b = order[k];
		if (t[b] < t[a] + d) t[b] = t[a] + d;
	}
	// Slide the finished fan back onto the pool, then into the window.
	// Re-centring on the pool alone was not enough: a pool sitting in a
	// corner — which is where every run STARTS, all six halls at (0, 0) —
	// has no room on one side, and a fan centred there put half of itself
	// outside the plate or, once that was clamped, collapsed to nothing.
	let after = 0;
	for (let i = 0; i < n; i++) after += t[i] / n;
	let shift = mid - after;
	const lo = t[order[0]] + shift;
	const hi = t[order[n - 1]] + shift;
	if (lo < tLo) shift += tLo - lo;
	else if (hi > tHi) shift += tHi - hi;
	for (let i = 0; i < n; i++) {
		const ti = t[i] + shift;
		px[i] = bx[i] + ti * iso[0];
		py[i] = by[i] + ti * iso[1];
	}
	return d;
}

/**
 * The endpoints of the line of equal fitness `f`, clipped to the view.
 *
 * These are the rungs of the ladder the pool is climbing: every hall standing
 * on one is worth the same to the race, whichever half of the skill it got
 * there with. Returns null when the line misses the frame entirely.
 */
export function isoLine(
	f: number,
	view: readonly number[],
	w = CATCH_WEIGHT
): [number, number, number, number] | null {
	const [x0, x1, y0, y1] = view;
	// deliver + w·catch = f, walked across the frame and clipped to it.
	const pts: [number, number][] = [];
	const push = (x: number, y: number) => {
		if (x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]);
	};
	push(x0, (f - x0) / w);
	push(x1, (f - x1) / w);
	push(f - w * y0, y0);
	push(f - w * y1, y1);
	if (pts.length < 2) return null;
	// The clip can report a corner twice; take the two furthest apart.
	let best: [number, number, number, number] | null = null;
	let far = -1;
	for (let i = 0; i < pts.length; i++)
		for (let j = i + 1; j < pts.length; j++) {
			const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
			if (d > far) {
				far = d;
				best = [pts[i][0], pts[i][1], pts[j][0], pts[j][1]];
			}
		}
	return far > 1e-9 ? best : null;
}

/** Round `v` down to a sensible tick spacing near `target`. */
export function niceStep(target: number): number {
	if (!(target > 0)) return 1;
	const mag = Math.pow(10, Math.floor(Math.log10(target)));
	const norm = target / mag;
	return (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
}

/**
 * How far a drawn rig reaches from the point it stands on.
 *
 * The pendulum is 0.85 m of rod at 1.25 slot-widths to the metre, the cart
 * slides up to 0.7 of a slot along its own little rail, and the attention
 * halo swells to 1.4. Rounding all of that up to 1.8 is what keeps a hall
 * that has mastered both halves from hanging over the edge of its own field
 * — which it did, because the plot reserved one slot and the rig needs
 * nearly two.
 */
export const RIG_REACH = 1.8;

/**
 * The drawable rectangle: the canvas, less room for the rigs and the two axis
 * labels — plus `slack`, the margin a crowded fan may borrow without pushing
 * a rig off the canvas. It is the smallest of the four fixed pads, because
 * the fan can run in any direction and the tightest side is the one that
 * decides.
 */
export function fieldBox(w: number, h: number, slot: number) {
	const reach = slot * RIG_REACH;
	const pad = { left: 30, right: 22, top: 22, bottom: 24 };
	return {
		left: pad.left + reach,
		right: w - pad.right - reach,
		top: pad.top + reach,
		bottom: h - pad.bottom - reach,
		// Deliberately small. The fan borrows this to open a degenerate rung —
		// six untrained halls all at (0, 0) sit in a CORNER, where the line of
		// equal fitness through them has no length inside the box at all. But
		// every pixel borrowed is a pixel a rig is drawn outside the frame, so
		// this buys just enough to separate a stack and no more; the frame is
		// drawn wide enough to contain even a fully borrowed fan.
		slack: Math.min(10, pad.left, pad.right, pad.top, pad.bottom)
	};
}

/**
 * The stretch of rung that is actually usable, as an interval in the same
 * projection `separate` works in.
 *
 * Not the box's width along `iso` — that is an upper bound the fan can still
 * overrun, because the fan lives where the halls are and the halls are
 * usually up in one corner. This clips the line through the pool's centre
 * against the box and hands back both ends, so a pool with room on one side
 * only can still fan out into the side it has.
 *
 * `slack` lets the fan borrow a little of the margin the rigs are drawn in.
 * Without it the opening seconds of every run look wrong: six untrained halls
 * are all at exactly (0, 0), which is a CORNER of the box, where the rung has
 * no length inside the frame at all and the pool draws as one pendulum. A
 * fraction of a slot is enough to open it and stays well inside the canvas.
 */
export function isoWindow(
	cx: number,
	cy: number,
	iso: readonly [number, number],
	box: { left: number; right: number; top: number; bottom: number },
	slack = 0
): [number, number] {
	let back = Infinity;
	let fwd = Infinity;
	const slab = (p: number, dir: number, lo: number, hi: number) => {
		if (Math.abs(dir) < 1e-9) {
			if (p < lo || p > hi) {
				fwd = 0;
				back = 0;
			}
			return;
		}
		const t1 = (lo - p) / dir;
		const t2 = (hi - p) / dir;
		fwd = Math.min(fwd, Math.max(t1, t2));
		back = Math.min(back, -Math.min(t1, t2));
	};
	slab(cx, iso[0], box.left - slack, box.right + slack);
	slab(cy, iso[1], box.top - slack, box.bottom + slack);
	const t = cx * iso[0] + cy * iso[1];
	return [t - Math.max(0, back), t + Math.max(0, fwd)];
}
