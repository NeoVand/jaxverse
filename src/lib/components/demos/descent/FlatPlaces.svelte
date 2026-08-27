<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	import { parseColor } from '$lib/optim/colormap';

	// The two kinds of flat place, drawn as the two surfaces themselves.
	//
	// An earlier version of this figure drew the loss along a bundle of
	// straight lines through the flat point — defensible, since a real
	// landscape has a million axes and nobody can look at it, but the picture
	// came out as two thickets of parabolas that read the same at a glance.
	// A reader should not have to count strokes to learn the difference
	// between a bowl and a saddle. So: two patches of actual ground, one
	// camera, one light, turning slowly.
	//
	// Both are quadratic, with the same curvature in magnitude — z = k(x² + y²)
	// against z = k(x² − y²) — so the only difference between the panels is the
	// sign on one axis, which is exactly the thing the chapter is about.
	//
	// The rim carries the argument. It is the circle of unit radius around the
	// flat point, so its height at angle θ IS the loss one step out in
	// direction θ. On the floor that circle is level and entirely above the
	// point: every direction climbs. On the saddle it is a sine wave through
	// the point's own height, drawn in ink where it rises and vermilion where
	// it falls. The dashed ring is that height, held level, to give the eye a
	// datum to judge "above" and "below" against.

	// ── mesh ──
	const RINGS = 32; // radial divisions of the disc
	const SPOKES = 84; // angular divisions
	const NV = (RINGS + 1) * SPOKES;
	const NQ = RINGS * SPOKES;
	const DASHES = SPOKES / 2; // every other datum segment, for a dashed ring
	const MARK_N = 29; // samples round the small ring at the flat point, closed
	const MARK_SEG = MARK_N - 1;
	const N_ITEMS = NQ + SPOKES + DASHES + MARK_SEG;
	const TAU = Math.PI * 2;

	const K = 0.55; // curvature amplitude, world units
	const LIFT = 0.02; // how far the rim floats off the surface it traces
	const LEVELS = 256; // steps in the lit→shadowed shading ramp
	const GROW = 0.5; // px pushed outward from a quad's centroid

	/** Light, fixed in the world rather than to the camera: the patch is a lit
	 *  object on a turntable, so a face keeps its brightness as you walk round. */
	const L = ((): [number, number, number] => {
		const v: [number, number, number] = [-0.42, 0.86, 0.3];
		const n = Math.hypot(...v);
		return [v[0] / n, v[1] / n, v[2] / n];
	})();

	interface Surf {
		key: string;
		name: string;
		sub: string;
		/** +1 for the floor, −1 for the saddle: the whole difference. */
		sign: number;
		/** world lattice, y up */
		vx: Float32Array;
		vy: Float32Array;
		vz: Float32Array;
		lam: Float32Array;
		shade: Uint8Array;
		/** the rim, floated just clear of the surface */
		rx: Float32Array;
		ry: Float32Array;
		rz: Float32Array;
		/** per rim segment: does the ground rise this way, or fall? */
		rise: Uint8Array;
		/** the level ring at the flat point's own height */
		dx: Float32Array;
		dy: Float32Array;
		dz: Float32Array;
		/** the small ring painted on the ground round the flat point */
		mx: Float32Array;
		my: Float32Array;
		mz: Float32Array;
	}

	function build(key: string, name: string, sub: string, sign: number): Surf {
		// centre the patch vertically in its panel: the floor lives entirely
		// above its flat point, the saddle straddles it.
		const mid = (K + sign * K) / 4;
		const h = (x: number, z: number) => K * (x * x + sign * z * z);

		const vx = new Float32Array(NV);
		const vy = new Float32Array(NV);
		const vz = new Float32Array(NV);
		for (let r = 0; r <= RINGS; r++) {
			const rad = r / RINGS;
			for (let s = 0; s < SPOKES; s++) {
				const a = (s / SPOKES) * TAU;
				const x = rad * Math.cos(a);
				const z = rad * Math.sin(a);
				const k = r * SPOKES + s;
				vx[k] = x;
				vz[k] = z;
				vy[k] = h(x, z) - mid;
			}
		}

		// Lambert, from the analytic normal at each quad's centre. ∇h is exact
		// here, so the shading owes nothing to the mesh resolution. Kept raw
		// for now: the two patches are normalised together, further down, so
		// that equal slopes come out equally bright in both panels.
		const lam = new Float32Array(NQ);
		for (let r = 0; r < RINGS; r++) {
			for (let s = 0; s < SPOKES; s++) {
				const rad = (r + 0.5) / RINGS;
				const a = ((s + 0.5) / SPOKES) * TAU;
				const x = rad * Math.cos(a);
				const z = rad * Math.sin(a);
				const nx = -2 * K * x;
				const nz = -2 * K * sign * z;
				const nl = Math.hypot(nx, 1, nz);
				lam[r * SPOKES + s] = Math.max(0, (nx * L[0] + L[1] + nz * L[2]) / nl);
			}
		}

		const rx = new Float32Array(SPOKES);
		const ry = new Float32Array(SPOKES);
		const rz = new Float32Array(SPOKES);
		const dx = new Float32Array(SPOKES);
		const dy = new Float32Array(SPOKES);
		const dz = new Float32Array(SPOKES);
		const rise = new Uint8Array(SPOKES);
		// The flat point, marked by a small ring painted on the ground round it
		// rather than by a dot sitting on top. It lies in the surface, so it
		// tilts and foreshortens with the patch instead of floating; and being
		// a circle it reads the same from every angle, where a tick or a cross
		// turns into a scratch at half the yaws it is seen from.
		const mx = new Float32Array(MARK_N);
		const my = new Float32Array(MARK_N);
		const mz = new Float32Array(MARK_N);
		for (let t = 0; t < MARK_N; t++) {
			const ang = (t / (MARK_N - 1)) * TAU;
			const x = 0.135 * Math.cos(ang);
			const z = 0.135 * Math.sin(ang);
			const nx = -2 * K * x;
			const nz = -2 * K * sign * z;
			const nl = Math.hypot(nx, 1, nz);
			mx[t] = x + (LIFT * nx) / nl;
			my[t] = h(x, z) - mid + LIFT / nl;
			mz[t] = z + (LIFT * nz) / nl;
		}
		for (let s = 0; s < SPOKES; s++) {
			const a = (s / SPOKES) * TAU;
			const x = Math.cos(a);
			const z = Math.sin(a);
			const nx = -2 * K * x;
			const nz = -2 * K * sign * z;
			const nl = Math.hypot(nx, 1, nz);
			rx[s] = x + (LIFT * nx) / nl;
			ry[s] = h(x, z) - mid + LIFT / nl;
			rz[s] = z + (LIFT * nz) / nl;
			dx[s] = x;
			dy[s] = -mid;
			dz[s] = z;
			// the sign of the segment, read at its midpoint, against the height
			// of the flat point itself — which is 0 before the panel is centred
			const am = ((s + 0.5) / SPOKES) * TAU;
			rise[s] = h(Math.cos(am), Math.sin(am)) >= 0 ? 1 : 0;
		}

		return {
			key,
			name,
			sub,
			sign,
			vx,
			vy,
			vz,
			lam,
			shade: new Uint8Array(NQ),
			rx,
			ry,
			rz,
			rise,
			dx,
			dy,
			dz,
			mx,
			my,
			mz
		};
	}

	const SURFACES: Surf[] = [
		build('floor', 'a floor', 'every direction curves up', 1),
		build('saddle', 'a saddle', 'one of them curves down', -1)
	];

	// One exposure for both patches: the darkest face anywhere sets the far end
	// of the ramp and the brightest sets the near end, so a slope that catches
	// the light on the floor reads exactly as bright on the saddle.
	{
		let lo = Infinity;
		let hi = -Infinity;
		for (const su of SURFACES)
			for (const v of su.lam) {
				if (v < lo) lo = v;
				if (v > hi) hi = v;
			}
		const span = Math.max(hi - lo, 1e-6);
		for (const su of SURFACES)
			for (let q = 0; q < NQ; q++)
				su.shade[q] = Math.round((1 - (su.lam[q] - lo) / span) * (LEVELS - 1));
	}

	// ── camera ──
	let yaw = $state(0.62);
	let pitch = $state(0.58);
	let dragging = $state(false);
	let dragLast: [number, number] | null = null;
	let canvas: HTMLCanvasElement;
	let dirty = true;
	let reducedMotion = false;

	function onPointerDown(e: PointerEvent) {
		dragging = true;
		dragLast = [e.clientX, e.clientY];
		(e.target as Element).setPointerCapture(e.pointerId);
	}
	function onPointerMove(e: PointerEvent) {
		if (!dragging || !dragLast) return;
		yaw += (e.clientX - dragLast[0]) * 0.008;
		pitch = Math.max(0.12, Math.min(1.15, pitch + (e.clientY - dragLast[1]) * 0.006));
		dragLast = [e.clientX, e.clientY];
		dirty = true;
	}
	function onPointerUp() {
		dragging = false;
		dragLast = null;
	}

	/** Arrow keys steer too, so the shape is reachable without a mouse. */
	function onKeyDown(e: KeyboardEvent) {
		const step = 0.12;
		if (e.key === 'ArrowLeft') yaw -= step;
		else if (e.key === 'ArrowRight') yaw += step;
		else if (e.key === 'ArrowUp') pitch = Math.min(1.15, pitch + step * 0.6);
		else if (e.key === 'ArrowDown') pitch = Math.max(0.12, pitch - step * 0.6);
		else return;
		e.preventDefault();
		dirty = true;
	}

	/** Rotate-project a 3-D point (y up); returns [x, y, depth], depth growing
	 *  toward the viewer. The descent chapter's own convention, kept local so
	 *  the plates stay decoupled from one another. */
	function project3(
		x: number,
		y: number,
		z: number,
		cy: number,
		sy: number,
		cp: number,
		sp: number
	): [number, number, number] {
		const x1 = cy * x + sy * z;
		const z1 = -sy * x + cy * z;
		return [x1, cp * y - sp * z1, sp * y + cp * z1];
	}

	// scratch, allocated once
	const sx = new Float32Array(NV);
	const sy = new Float32Array(NV);
	const rsx = new Float32Array(SPOKES);
	const rsy = new Float32Array(SPOKES);
	const rsd = new Float32Array(SPOKES);
	const dsx = new Float32Array(SPOKES);
	const dsy = new Float32Array(SPOKES);
	const dsd = new Float32Array(SPOKES);
	const msx = new Float32Array(MARK_N);
	const msy = new Float32Array(MARK_N);
	const msd = new Float32Array(MARK_N);
	const sd = new Float32Array(NV);
	const depth = new Float32Array(N_ITEMS);
	const order = new Uint16Array(N_ITEMS);

	$effect(() => {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

		let disposed = false;
		let raf = 0;
		let last = performance.now();
		let lutKey = '';
		let lut: string[] = [];

		function draw() {
			if (!ctx) return;
			const dpr = Math.min(devicePixelRatio || 1, 2);
			const W = canvas.clientWidth;
			const H = canvas.clientHeight;
			if (W === 0 || H === 0) return;
			if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
				canvas.width = Math.round(W * dpr);
				canvas.height = Math.round(H * dpr);
				dirty = true;
			}

			const st = getComputedStyle(canvas);
			const tk = {
				paper: st.getPropertyValue('--surface').trim(),
				ink: st.getPropertyValue('--ink').trim(),
				ink3: st.getPropertyValue('--ink-3').trim(),
				accent: st.getPropertyValue('--accent').trim(),
				warm: st.getPropertyValue('--warm').trim()
			};

			// Lit→shadowed fill ramp. Both ends are mixed from the page's own
			// two colours, but which of those is the BRIGHT one flips with the
			// theme, so the ramp is anchored on luminance rather than on the
			// token names: a face turned toward the light is always the lighter
			// of the two, in either theme. Neither end reaches the page colour —
			// a patch that dissolved into the paper would have no silhouette.
			const lk = `${tk.paper}|${tk.ink}|${tk.accent}`;
			if (lk !== lutKey) {
				lutKey = lk;
				const p = parseColor(tk.paper);
				const i = parseColor(tk.ink);
				const a = parseColor(tk.accent);
				// the ink of the page, cooled toward the accent, so the ground
				// reads as material rather than as grey
				const ink: [number, number, number] = [
					i[0] * 0.78 + a[0] * 0.22,
					i[1] * 0.78 + a[1] * 0.22,
					i[2] * 0.78 + a[2] * 0.22
				];
				const lum = (c: [number, number, number]) => 0.21 * c[0] + 0.72 * c[1] + 0.07 * c[2];
				const dark = lum(p) < lum(ink);
				const mix = (t: number): [number, number, number] => [
					p[0] + (ink[0] - p[0]) * t,
					p[1] + (ink[1] - p[1]) * t,
					p[2] + (ink[2] - p[2]) * t
				];
				const bright = dark ? mix(0.6) : mix(0.04);
				const shadow = dark ? mix(0.05) : mix(0.8);
				lut = Array.from({ length: LEVELS }, (_, n) => {
					const w = n / (LEVELS - 1);
					const c = (k: number) => Math.round(bright[k] + (shadow[k] - bright[k]) * w);
					return `rgb(${c(0)},${c(1)},${c(2)})`;
				});
			}

			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, W, H);
			ctx.lineJoin = 'round';
			ctx.lineCap = 'round';

			const cy = Math.cos(yaw);
			const syw = Math.sin(yaw);
			const cp = Math.cos(pitch);
			const sp = Math.sin(pitch);
			const panelW = W / 2;
			const S = Math.min(panelW * 0.35, H * 0.4);

			for (let pi = 0; pi < SURFACES.length; pi++) {
				const su = SURFACES[pi];
				const ox = panelW * (pi + 0.5);
				const oyPx = H * 0.5;

				for (let k = 0; k < NV; k++) {
					const [px, py, pd] = project3(su.vx[k], su.vy[k], su.vz[k], cy, syw, cp, sp);
					sx[k] = ox + px * S;
					sy[k] = oyPx - py * S;
					sd[k] = pd;
				}
				for (let s = 0; s < SPOKES; s++) {
					const [px, py, pd] = project3(su.rx[s], su.ry[s], su.rz[s], cy, syw, cp, sp);
					rsx[s] = ox + px * S;
					rsy[s] = oyPx - py * S;
					rsd[s] = pd;
					const [qx, qy, qd] = project3(su.dx[s], su.dy[s], su.dz[s], cy, syw, cp, sp);
					dsx[s] = ox + qx * S;
					dsy[s] = oyPx - qy * S;
					dsd[s] = qd;
				}
				for (let k = 0; k < MARK_N; k++) {
					const [px, py, pd] = project3(su.mx[k], su.my[k], su.mz[k], cy, syw, cp, sp);
					msx[k] = ox + px * S;
					msy[k] = oyPx - py * S;
					msd[k] = pd;
				}

				// A saddle folds back on itself in projection, so the exact
				// axis sweep the race uses for a heightfield does not apply
				// here. Every piece is small and convex, so ordering them by
				// the depth of their centre is exact for these surfaces.
				for (let q = 0; q < NQ; q++) {
					const r = (q / SPOKES) | 0;
					const s = q % SPOKES;
					const s1 = (s + 1) % SPOKES;
					const k00 = r * SPOKES + s;
					const k01 = r * SPOKES + s1;
					const k10 = (r + 1) * SPOKES + s;
					const k11 = (r + 1) * SPOKES + s1;
					depth[q] = (sd[k00] + sd[k01] + sd[k10] + sd[k11]) / 4;
				}
				for (let s = 0; s < SPOKES; s++)
					depth[NQ + s] = (rsd[s] + rsd[(s + 1) % SPOKES]) / 2 + 1e-3;
				for (let d = 0; d < DASHES; d++) {
					const s = d * 2;
					depth[NQ + SPOKES + d] = (dsd[s] + dsd[(s + 1) % SPOKES]) / 2;
				}
				// per segment, not per ring: a single depth for the whole mark
				// puts half of it behind the ground it is painted on
				for (let t = 0; t < MARK_SEG; t++)
					depth[NQ + SPOKES + DASHES + t] = (msd[t] + msd[t + 1]) / 2 + 2e-3;

				for (let n = 0; n < N_ITEMS; n++) order[n] = n;
				order.sort((a, b) => depth[a] - depth[b]);

				let fill = -1;
				for (let n = 0; n < N_ITEMS; n++) {
					const it = order[n];
					if (it < NQ) {
						const r = (it / SPOKES) | 0;
						const s = it % SPOKES;
						const s1 = (s + 1) % SPOKES;
						const ks = [
							r * SPOKES + s,
							r * SPOKES + s1,
							(r + 1) * SPOKES + s1,
							(r + 1) * SPOKES + s
						];
						const ci = su.shade[it];
						if (ci !== fill) {
							fill = ci;
							ctx.fillStyle = lut[ci];
						}
						let mx = 0;
						let my = 0;
						for (const k of ks) {
							mx += sx[k] / 4;
							my += sy[k] / 4;
						}
						// each quad inflated half a pixel from its centroid, so
						// antialiasing never opens paper-coloured seams between
						// neighbours
						ctx.beginPath();
						for (let c = 0; c < 4; c++) {
							const k = ks[c];
							const ex = sx[k] - mx;
							const ey = sy[k] - my;
							const g = 1 + GROW / (Math.hypot(ex, ey) || 1);
							if (c === 0) ctx.moveTo(mx + ex * g, my + ey * g);
							else ctx.lineTo(mx + ex * g, my + ey * g);
						}
						ctx.closePath();
						ctx.fill();
					} else if (it < NQ + SPOKES) {
						const s = it - NQ;
						const s1 = (s + 1) % SPOKES;
						const up = su.rise[s] === 1;
						// the book's own pair for a signed quantity: ultramarine
						// where the ground climbs, vermilion where it drops
						ctx.strokeStyle = up ? tk.accent : tk.warm;
						ctx.lineWidth = 2.6;
						ctx.beginPath();
						ctx.moveTo(rsx[s], rsy[s]);
						ctx.lineTo(rsx[s1], rsy[s1]);
						ctx.stroke();
						fill = -1;
					} else if (it < NQ + SPOKES + DASHES) {
						const s = (it - NQ - SPOKES) * 2;
						const s1 = (s + 1) % SPOKES;
						ctx.strokeStyle = tk.ink3;
						ctx.lineWidth = 1;
						ctx.globalAlpha = 0.7;
						ctx.beginPath();
						ctx.moveTo(dsx[s], dsy[s]);
						ctx.lineTo(dsx[s1], dsy[s1]);
						ctx.stroke();
						ctx.globalAlpha = 1;
						fill = -1;
					} else {
						const t = it - NQ - SPOKES - DASHES;
						ctx.strokeStyle = tk.ink;
						ctx.lineWidth = 1.3;
						ctx.globalAlpha = 0.75;
						ctx.beginPath();
						ctx.moveTo(msx[t], msy[t]);
						ctx.lineTo(msx[t + 1], msy[t + 1]);
						ctx.stroke();
						ctx.globalAlpha = 1;
						fill = -1;
					}
				}
			}
		}

		function frame(now: number) {
			if (disposed) return;
			const dt = Math.min(now - last, 64);
			last = now;
			if (!reducedMotion && !dragging) {
				yaw += dt * 0.00011; // idle orbit, the pace the space chapter turns at
				dirty = true;
			}
			if (dirty) {
				dirty = false;
				draw();
			}
			raf = requestAnimationFrame(frame);
		}

		raf = requestAnimationFrame(frame);
		const ro = new ResizeObserver(() => (dirty = true));
		ro.observe(canvas);
		return () => {
			disposed = true;
			cancelAnimationFrame(raf);
			ro.disconnect();
		};
	});
</script>

<Plate
	id="flat"
	title="The two kinds of flat"
	caption="Two patches of ground, each perfectly flat inside the small ring painted at its centre, lit the same way and seen from the same angle. They differ by one sign. Left: the ground curves up whichever way you leave — a floor, and a walker that arrives there is staying. Right: it curves up along one axis and down along the other — a saddle, and a walker carrying any momentum or any noise will eventually find the way off. The heavy circle drawn on each patch is the loss one step out in every direction at once: level and entirely above the point on the floor, a wave that crosses the point's own height four times on the saddle, in ultramarine where the ground rises and vermilion where it falls. The dashed ring holds that height level, to judge the wave against. Now count directions. With two of them to agree, floors are easy to come by; with a million, a floor is an extraordinary coincidence, and almost every flat place a large network stalls at is the picture on the right."
>
	<div class="p-4 sm:p-5">
		<canvas
			bind:this={canvas}
			class="block w-full select-none"
			class:cursor-grab={!dragging}
			class:cursor-grabbing={dragging}
			style="aspect-ratio: 76 / 31; touch-action: none;"
			role="button"
			tabindex="0"
			aria-label="Two three-dimensional patches of a loss surface side by side, turning slowly, each flat inside a small ring painted at its centre. The left patch is a shallow round bowl: the ground rises away from the point in every direction, and the circle traced around the point sits level, entirely above it. The right patch is a saddle: the ground rises along one axis and falls along the perpendicular one, and the circle traced around the point rises on two opposite arcs, drawn in ultramarine, and falls on the two arcs between them, drawn in vermilion, crossing the height of the point four times. A dashed ring in each panel marks that height. Both patches turn on their own; drag or press the arrow keys to steer them."
			onkeydown={onKeyDown}
			onpointerdown={onPointerDown}
			onpointermove={onPointerMove}
			onpointerup={onPointerUp}
			onpointercancel={onPointerUp}
		></canvas>

		<div class="mt-1 grid grid-cols-2 gap-4">
			{#each SURFACES as su (su.key)}
				<div class="text-center">
					<div class="label">{su.name}</div>
					<div class="num text-[10.5px] text-ink-3">{su.sub}</div>
				</div>
			{/each}
		</div>

		<p class="num mt-3 text-center text-[10.5px] text-ink-3">
			both patches turn on their own; drag to steer them
		</p>
	</div>
</Plate>

<style>
	.label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 13px;
		color: var(--ink-2);
	}
</style>
