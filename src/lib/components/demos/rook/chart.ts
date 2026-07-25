// Tiny SVG chart arithmetic shared by the rook plates. String-built paths,
// fixed viewBoxes — the house convention for loss-curve-style figures.

/** Linear map from [d0, d1] to [r0, r1]; guards a degenerate domain. */
export function scale(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
	const span = d1 - d0 || 1;
	const k = (r1 - r0) / span;
	return (v) => r0 + (v - d0) * k;
}

/** M/L path through the given pixel points. */
export function polyline(pts: Array<[number, number]>): string {
	return pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
}
