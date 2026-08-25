// Shared geometry for the Bending Space views: the reference grid that gets
// warped, tiny PCA for peeking at wide hidden layers, 3-D projection, and
// plane–cube clipping for the separating plane.

export interface GridLines {
	/** All polyline vertices, row-major [x, y] — one activations() call maps everything. */
	verts: Float32Array;
	/** Polyline extents into verts: [start, count] per line. */
	lines: Array<[number, number]>;
	n: number;
}

/** A square grid of horizontal + vertical polylines over [-e, e]². */
export function makeGridLines(cells = 12, samples = 25, extent = 1.1): GridLines {
	const lines: Array<[number, number]> = [];
	const pts: number[] = [];
	const coord = (i: number, n: number) => -extent + (2 * extent * i) / n;
	for (let r = 0; r <= cells; r++) {
		const y = coord(r, cells);
		lines.push([pts.length / 2, samples]);
		for (let s = 0; s < samples; s++) pts.push(coord(s, samples - 1), y);
	}
	for (let c = 0; c <= cells; c++) {
		const x = coord(c, cells);
		lines.push([pts.length / 2, samples]);
		for (let s = 0; s < samples; s++) pts.push(x, coord(s, samples - 1));
	}
	return { verts: new Float32Array(pts), lines, n: pts.length / 2 };
}

/** The probe lattice the decision map is painted on. Extents may differ per
 * axis so the wash can cover a non-square canvas without distortion. */
export function makeProbeGrid(res: number, extentX = 1.15, extentY = extentX): Float32Array {
	const g = new Float32Array(res * res * 2);
	for (let j = 0; j < res; j++) {
		for (let i = 0; i < res; i++) {
			const k = j * res + i;
			g[2 * k] = -extentX + (2 * extentX * (i + 0.5)) / res;
			g[2 * k + 1] = -extentY + (2 * extentY * (j + 0.5)) / res;
		}
	}
	return g;
}

/**
 * The shadow: the top-k principal directions of a hidden layer too wide to
 * draw, kept STABLE from one frame to the next.
 *
 * Power iteration finds a direction, not an arrow — flip the sign of an
 * eigenvector and it is still an eigenvector — and it has no opinion about
 * the order of two components with nearly equal eigenvalues. Refitting from
 * scratch every refresh therefore produced a basis that silently negated or
 * swapped its axes several times a second, and since the view lerps between
 * consecutive snapshots, each flip was animated: the whole cloud swept
 * through the origin and reassembled mirrored. That is what a reader saw as
 * the wide-layer plate "not behaving".
 *
 * So the fit is handed the previous basis and does two things with it: starts
 * the iteration there, which is already nearly converged and cuts the work by
 * an order of magnitude, and then matches each new component back to the old
 * one it most resembles — reordering, and negating where the sign turned
 * over. The shadow drifts as the representation drifts, and does nothing
 * else.
 */
export interface Pca {
	/** k orthonormal directions in the hidden space, each of length d. */
	basis: Float64Array[];
	/** The centre they are measured from. */
	mean: Float64Array;
	d: number;
}

export function pcaFit(data: Float32Array, n: number, d: number, k: number, prev?: Pca): Pca {
	const mean = new Float64Array(d);
	for (let i = 0; i < n; i++) for (let q = 0; q < d; q++) mean[q] += data[i * d + q];
	for (let q = 0; q < d; q++) mean[q] /= n;

	const matVec = (v: Float64Array, out: Float64Array) => {
		out.fill(0);
		for (let i = 0; i < n; i++) {
			let dot = 0;
			for (let q = 0; q < d; q++) dot += (data[i * d + q] - mean[q]) * v[q];
			for (let q = 0; q < d; q++) out[q] += dot * (data[i * d + q] - mean[q]);
		}
	};
	const normalize = (v: Float64Array) => {
		let s = 0;
		for (let q = 0; q < d; q++) s += v[q] * v[q];
		s = Math.sqrt(s) || 1;
		for (let q = 0; q < d; q++) v[q] /= s;
		return s;
	};

	const comps = Math.min(k, d);
	const warm = prev && prev.d === d && prev.basis.length >= comps;
	// A warm basis is already the answer to within a drift; three sweeps hold
	// it there. Cold, it is a guess and needs the full climb.
	const iters = warm ? 3 : 24;
	const basis: Float64Array[] = [];
	for (let c = 0; c < comps; c++) {
		let v = new Float64Array(d);
		if (warm) v.set(prev.basis[c]);
		else for (let q = 0; q < d; q++) v[q] = Math.sin(q * 2.3 + c * 1.7) + 0.5;
		normalize(v);
		const tmp = new Float64Array(d);
		for (let it = 0; it < iters; it++) {
			matVec(v, tmp);
			for (const b of basis) {
				let dot = 0;
				for (let q = 0; q < d; q++) dot += tmp[q] * b[q];
				for (let q = 0; q < d; q++) tmp[q] -= dot * b[q];
			}
			if (normalize(tmp) < 1e-12) break;
			v = tmp.slice();
		}
		basis.push(v.slice());
	}
	return prev ? alignTo(prev, { basis, mean, d }) : { basis, mean, d };
}

/**
 * Reorder and sign-flip `next` so each of its axes continues the axis of
 * `prev` it most resembles. Greedy on |cosine|, which is enough: the basis
 * moves a little per refresh, so the correct match is nearly always the
 * obvious one, and when two components genuinely swap this follows them
 * across rather than cutting to the new order.
 */
function alignTo(prev: Pca, next: Pca): Pca {
	if (prev.d !== next.d) return next;
	const k = Math.min(prev.basis.length, next.basis.length);
	const taken = new Array(next.basis.length).fill(false);
	const out: Float64Array[] = [];
	for (let c = 0; c < k; c++) {
		let best = -1;
		let bestDot = 0;
		for (let j = 0; j < next.basis.length; j++) {
			if (taken[j]) continue;
			let dot = 0;
			for (let q = 0; q < next.d; q++) dot += prev.basis[c][q] * next.basis[j][q];
			if (Math.abs(dot) > Math.abs(bestDot) || best < 0) {
				best = j;
				bestDot = dot;
			}
		}
		if (best < 0) break;
		taken[best] = true;
		const v = next.basis[best].slice();
		if (bestDot < 0) for (let q = 0; q < next.d; q++) v[q] = -v[q];
		out.push(v);
	}
	for (let j = 0; j < next.basis.length; j++) if (!taken[j]) out.push(next.basis[j]);
	return { basis: out.slice(0, next.basis.length), mean: next.mean, d: next.d };
}

/** Rows of `data` in shadow coordinates: n × k, centred on the fit's mean. */
export function pcaApply(pca: Pca, data: Float32Array, n: number, k: number): Float32Array {
	const d = pca.d;
	const comps = Math.min(k, pca.basis.length);
	const out = new Float32Array(n * k);
	for (let i = 0; i < n; i++)
		for (let c = 0; c < comps; c++) {
			let dot = 0;
			const b = pca.basis[c];
			for (let q = 0; q < d; q++) dot += (data[i * d + q] - pca.mean[q]) * b[q];
			out[i * k + c] = dot;
		}
	return out;
}

/**
 * Where the classifier's hyperplane cuts the shadow.
 *
 * A plane w·h + c = 0 lives in the full hidden space. A point drawn at shadow
 * coordinates t stands for mean + Σ tᵢ·basisᵢ, so along that slice the plane
 * is w·mean + c + Σ tᵢ (w·basisᵢ) = 0 — still a plane, with normal (w·basisᵢ)
 * and offset (w·mean + c). It is the plane's TRACE in the three directions
 * being shown and not the plane itself: whatever the classifier does in the
 * directions the shadow drops is, by construction, not on screen.
 */
export function pcaPlane(
	pca: Pca,
	w: readonly number[],
	c: number,
	k = 3
): { n: number[]; c: number } {
	const comps = Math.min(k, pca.basis.length);
	const n: number[] = [];
	for (let i = 0; i < comps; i++) {
		let dot = 0;
		for (let q = 0; q < pca.d; q++) dot += w[q] * pca.basis[i][q];
		n.push(dot);
	}
	while (n.length < k) n.push(0);
	let off = c;
	for (let q = 0; q < pca.d; q++) off += w[q] * pca.mean[q];
	return { n, c: off };
}

/** Rotate-project a 3-D point to view coordinates (y up). */
export function project3(
	x: number,
	y: number,
	z: number,
	yaw: number,
	pitch: number
): [number, number, number] {
	const cy = Math.cos(yaw);
	const sy = Math.sin(yaw);
	const x1 = cy * x + sy * z;
	const z1 = -sy * x + cy * z;
	const cp = Math.cos(pitch);
	const sp = Math.sin(pitch);
	const y2 = cp * y - sp * z1;
	const z2 = sp * y + cp * z1;
	return [x1, y2, z2]; // z2 = depth for shading
}

/** Clip the plane n·h + c = 0 against the cube [-e, e]³ → ordered polygon. */
export function planeCubePolygon(
	n0: number,
	n1: number,
	n2: number,
	c: number,
	e = 1
): Array<[number, number, number]> {
	const pts: Array<[number, number, number]> = [];
	const val = (x: number, y: number, z: number) => n0 * x + n1 * y + n2 * z + c;
	const corners: Array<[number, number, number]> = [];
	for (const x of [-e, e])
		for (const y of [-e, e]) for (const z of [-e, e]) corners.push([x, y, z]);
	const edges: Array<[number, number]> = [
		[0, 1],
		[0, 2],
		[0, 4],
		[1, 3],
		[1, 5],
		[2, 3],
		[2, 6],
		[3, 7],
		[4, 5],
		[4, 6],
		[5, 7],
		[6, 7]
	];
	for (const [a, b] of edges) {
		const va = val(...corners[a]);
		const vb = val(...corners[b]);
		if ((va < 0 && vb >= 0) || (vb < 0 && va >= 0)) {
			const t = va / (va - vb);
			pts.push([
				corners[a][0] + t * (corners[b][0] - corners[a][0]),
				corners[a][1] + t * (corners[b][1] - corners[a][1]),
				corners[a][2] + t * (corners[b][2] - corners[a][2])
			]);
		}
	}
	if (pts.length < 3) return pts;
	// order around centroid in the plane's own basis
	const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
	const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
	const cz = pts.reduce((s, p) => s + p[2], 0) / pts.length;
	const nl = Math.hypot(n0, n1, n2) || 1;
	const nx = n0 / nl;
	const ny = n1 / nl;
	const nz = n2 / nl;
	let ux = 1;
	let uy = 0;
	let uz = 0;
	if (Math.abs(nx) > 0.9) {
		ux = 0;
		uy = 1;
	}
	// u = normalize(u - (u·n)n), v = n × u
	const und = ux * nx + uy * ny + uz * nz;
	ux -= und * nx;
	uy -= und * ny;
	uz -= und * nz;
	const ul = Math.hypot(ux, uy, uz) || 1;
	ux /= ul;
	uy /= ul;
	uz /= ul;
	const vx = ny * uz - nz * uy;
	const vy = nz * ux - nx * uz;
	const vz = nx * uy - ny * ux;
	return pts
		.map((p) => {
			const dx = p[0] - cx;
			const dy = p[1] - cy;
			const dz = p[2] - cz;
			return {
				p,
				a: Math.atan2(dx * vx + dy * vy + dz * vz, dx * ux + dy * uy + dz * uz)
			};
		})
		.sort((m, o) => m.a - o.a)
		.map((m) => m.p);
}

/** Design tokens resolved at draw time so canvases follow theme flips. */
export { readTokens, hexRgb, type Rgb, type Tokens } from '$lib/viz/tokens.svelte';
