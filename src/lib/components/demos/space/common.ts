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

/** The probe lattice the decision map is painted on. */
export function makeProbeGrid(res: number, extent = 1.15): Float32Array {
	const g = new Float32Array(res * res * 2);
	for (let j = 0; j < res; j++) {
		for (let i = 0; i < res; i++) {
			const k = j * res + i;
			g[2 * k] = -extent + (2 * extent * (i + 0.5)) / res;
			g[2 * k + 1] = -extent + (2 * extent * (j + 0.5)) / res;
		}
	}
	return g;
}

/** Top-2 principal directions by power iteration — the “shadow” view for
 * hidden layers wider than 3. Returns n×2 projections, centered. */
export function pca2(data: Float32Array, n: number, d: number): Float32Array {
	const mean = new Float64Array(d);
	for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += data[i * d + k];
	for (let k = 0; k < d; k++) mean[k] /= n;

	const matVec = (v: Float64Array, out: Float64Array) => {
		out.fill(0);
		for (let i = 0; i < n; i++) {
			let dot = 0;
			for (let k = 0; k < d; k++) dot += (data[i * d + k] - mean[k]) * v[k];
			for (let k = 0; k < d; k++) out[k] += dot * (data[i * d + k] - mean[k]);
		}
	};
	const normalize = (v: Float64Array) => {
		let s = 0;
		for (let k = 0; k < d; k++) s += v[k] * v[k];
		s = Math.sqrt(s) || 1;
		for (let k = 0; k < d; k++) v[k] /= s;
	};

	const basis: Float64Array[] = [];
	for (let c = 0; c < 2; c++) {
		let v = new Float64Array(d);
		for (let k = 0; k < d; k++) v[k] = Math.sin(k * 2.3 + c * 1.7) + 0.5;
		const tmp = new Float64Array(d);
		for (let it = 0; it < 24; it++) {
			matVec(v, tmp);
			// deflate against earlier components
			for (const b of basis) {
				let dot = 0;
				for (let k = 0; k < d; k++) dot += tmp[k] * b[k];
				for (let k = 0; k < d; k++) tmp[k] -= dot * b[k];
			}
			normalize(tmp);
			v = tmp.slice();
		}
		basis.push(v.slice());
	}

	const out = new Float32Array(n * 2);
	for (let i = 0; i < n; i++) {
		for (let c = 0; c < 2; c++) {
			let dot = 0;
			for (let k = 0; k < d; k++) dot += (data[i * d + k] - mean[k]) * basis[c][k];
			out[i * 2 + c] = dot;
		}
	}
	return out;
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
export function readTokens(el: Element) {
	const s = getComputedStyle(el);
	const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback;
	return {
		paper: v('--paper', '#faf9f5'),
		surface: v('--surface', '#ffffff'),
		ink: v('--ink', '#1d1c18'),
		ink2: v('--ink-2', '#605d54'),
		ink3: v('--ink-3', '#a3a094'),
		line: v('--line', '#e5e2d8'),
		lineSoft: v('--line-soft', '#efede4'),
		accent: v('--accent', '#2b45d8'),
		warm: v('--warm', '#d3541f'),
		good: v('--good', '#22774d'),
		bad: v('--bad', '#bb3a2b')
	};
}

export type Tokens = ReturnType<typeof readTokens>;

/** Parse #rrggbb → [r,g,b]; tolerates whitespace. */
export function hexRgb(hex: string): [number, number, number] {
	const h = hex.replace('#', '').trim();
	return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
