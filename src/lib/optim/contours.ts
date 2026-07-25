/**
 * Contour rings over the loss grid — d3-contour on the extended grid with
 * log-spaced thresholds, projected straight to SVG path strings (no d3-geo:
 * the projection is a plain affine map, so we walk the MultiPolygon rings
 * ourselves). The strings feed `new Path2D(d)` for canvas or `<path d>`.
 *
 * d3-contour convention: coordinates live in grid space with samples at
 * cell centers, so gx / res maps back to the extended parameter fraction
 * exactly — heatmap pixels and contour lines align.
 */

import { contours } from 'd3-contour';
import { contourThresholds, type LossGrid } from './landscape';

export function contourPaths(
	grid: LossGrid,
	project: (x: number, y: number) => [number, number],
	levels?: number
): string[] {
	const { res, extXMin, extXMax, extYMin, extYMax, values } = grid;
	const spanX = extXMax - extXMin;
	const spanY = extYMax - extYMin;

	const gen = contours().size([res, res]).smooth(true).thresholds(contourThresholds(grid, levels));

	const out: string[] = [];
	for (const multi of gen(Array.from(values))) {
		let d = '';
		for (const polygon of multi.coordinates) {
			for (const ring of polygon) {
				for (let k = 0; k < ring.length; k++) {
					const x = extXMin + (ring[k][0] / res) * spanX;
					const y = extYMin + (ring[k][1] / res) * spanY;
					const [px, py] = project(x, y);
					d += `${k === 0 ? 'M' : 'L'}${px.toFixed(1)} ${py.toFixed(1)}`;
				}
				d += 'Z';
			}
		}
		if (d) out.push(d);
	}
	return out;
}
