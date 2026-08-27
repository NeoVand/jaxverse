<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';

	// One layer, held still, on the sheet it acts on.
	//
	// The chapter's argument is that h = σ(Wx + b) deforms the plane rather
	// than drawing a line on it, and the two plates that follow are live labs
	// where that happens too fast, and in too many dimensions, to study. This
	// is the still frame in between: the same sheet three times, before, after
	// the affine part, and after the bend.
	//
	// Nothing here is drawn by hand. Every grid line and every datum is the
	// real map applied to real coordinates, so the picture cannot drift away
	// from what it claims — in particular the middle panel's lines are straight
	// because an affine map keeps them straight, not because a ruler was used,
	// and the last panel's lines crowd against its edges because that is what
	// tanh does to a plane, not because they were drawn that way.

	/** A turn, a shear and a stretch at once, so all three are legible: the
	 *  sheet visibly rotates, and its cells visibly stop being square. Nothing
	 *  else is special about these numbers. */
	const W: readonly [number, number, number, number] = [1.15, -0.55, 0.62, 1.0];
	const B: readonly [number, number] = [0.25, -0.15];

	type Pt = readonly [number, number];

	const affine = ([x, y]: Pt): Pt => [W[0] * x + W[1] * y + B[0], W[2] * x + W[3] * y + B[1]];
	const bend = ([x, y]: Pt): Pt => [Math.tanh(x), Math.tanh(y)];

	/** Through the first `stage` operations: 0 the plane, 1 affine, 2 bent. */
	function through(p: Pt, stage: number): Pt {
		let q = p;
		if (stage >= 1) q = affine(q);
		if (stage >= 2) q = bend(q);
		return q;
	}

	// ── geometry, in viewBox units ──
	const PANEL = 140;
	const TOP = 30;
	const XS = [8, 190, 372] as const;

	/**
	 * How wide a window each panel frames.
	 *
	 * The first two share one, because they are the same sheet at the same
	 * size and the comparison between them is the whole point. The third is
	 * framed to exactly (−1, 1)² — twice the magnification, and the caption
	 * says so. Drawn at the shared scale it is honest and illegible: tanh
	 * crushes the entire infinite plane into a box a quarter of the frame's
	 * area, and the curvature that is the reason for the panel disappears
	 * inside it. Framed this way the panel's own border IS the boundary the
	 * bend maps into, which is also what makes the grid crowd against it.
	 */
	const SPAN = [2.0, 2.0, 1.0] as const;

	const sx = (p: number, x: number) => XS[p] + ((x + SPAN[p]) / (2 * SPAN[p])) * PANEL;
	const sy = (p: number, y: number) => TOP + ((SPAN[p] - y) / (2 * SPAN[p])) * PANEL;

	/** Sampled rather than ruled: a curved stage comes out smooth, a straight
	 *  one comes out exactly straight. Extent runs well past every frame — the
	 *  plane does not stop at the border, and each panel clips its own. */
	const LINES = 13;
	const EXTENT = 1.9;
	const SAMPLES = 41;

	interface GridLine {
		d: string;
		/** The two lines through the origin, drawn a shade stronger so a reader
		 *  can follow where the origin went. */
		axis: boolean;
	}

	function grid(stage: number): GridLine[] {
		const out: GridLine[] = [];
		const at = (i: number, n: number) => -EXTENT + (2 * EXTENT * i) / (n - 1);
		for (let dir = 0; dir < 2; dir++)
			for (let i = 0; i < LINES; i++) {
				const fixed = at(i, LINES);
				let d = '';
				for (let k = 0; k < SAMPLES; k++) {
					const t = at(k, SAMPLES);
					const p: Pt = dir === 0 ? [fixed, t] : [t, fixed];
					const [ux, uy] = through(p, stage);
					d += `${k === 0 ? 'M' : 'L'} ${sx(stage, ux).toFixed(2)} ${sy(stage, uy).toFixed(2)} `;
				}
				out.push({ d: d.trim(), axis: Math.abs(fixed) < 1e-9 });
			}
		return out;
	}

	const GRIDS = [grid(0), grid(1), grid(2)];

	/** A ring around a disk — the tangle the next plate hands to a network.
	 *  Angles are evenly spaced rather than sampled: a still figure should come
	 *  out identical every time it is drawn. */
	function circle(r: number, n: number, turn = 0): Pt[] {
		return Array.from({ length: n }, (_, i): Pt => {
			const th = (2 * Math.PI * i) / n + turn;
			return [r * Math.cos(th), r * Math.sin(th)];
		});
	}
	const DISK: Pt[] = [...circle(0.3, 6, 0.5), ...circle(0.58, 12)];
	const RING: Pt[] = circle(1.0, 28);

	const STAGES = [
		{ key: 'x', head: 'x', name: 'the plane', sub: 'a ring around a disk' },
		{ key: 'a', head: 'Wx + b', name: 'the affine part', sub: 'turned, stretched, shifted' },
		{ key: 's', head: 'σ(Wx + b) · h', name: 'the bend', sub: 'all of it, inside (−1, 1)²' }
	] as const;

	const ARROWS = [
		{ key: 'a', from: 0, to: 1, eq: 'Wx + b', tone: 'accent' },
		{ key: 's', from: 1, to: 2, eq: 'σ', tone: 'teal' }
	] as const;
</script>

<Plate
	id="layer"
	title="What one layer does"
	caption="One layer, drawn on the sheet it acts on rather than on the data sitting upon it. Left: the plane, with a ring around a disk. Middle: the affine part has turned, stretched and shifted the whole sheet, and every line of the grid is still straight — an affine map cannot make a straight line bend, which is the entire reason a stack of them alone would collapse into one. Right: the bend, applied to each coordinate on its own. The lines curve now, and the whole infinite plane has been drawn inside a square two units across: that square is this panel's teal border, which is why the grid crowds against it and why the panel is magnified twice over the other two. Nothing was cut and nothing was pulled apart, and the ring still surrounds the disk — which is exactly the trouble the next plate walks into."
>
	<svg
		viewBox="0 0 520 210"
		class="mx-auto block w-full max-w-[825px]"
		role="img"
		aria-label="One layer of a neural network drawn as a deformation of the plane, in three panels. Left: a square grid carrying a ring of points around a small disk of points at the centre. Middle: after the affine part W x plus b the grid is rotated, sheared and shifted, and every line in it is still perfectly straight. Right: after the tanh bend, applied to each coordinate separately, the grid lines curve and the whole plane is compressed inside a square two units across, with the lines crowding against its edges. In all three panels the ring still encircles the disk."
	>
		<defs>
			{#each STAGES as stage, p (stage.key)}
				<clipPath id="ld-clip-{p}">
					<rect x={XS[p]} y={TOP} width={PANEL} height={PANEL} rx="3" />
				</clipPath>
			{/each}
			<marker
				id="ld-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="6"
				markerHeight="6"
				orient="auto"
			>
				<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
			</marker>
		</defs>

		{#each STAGES as stage, p (stage.key)}
			<rect
				x={XS[p]}
				y={TOP}
				width={PANEL}
				height={PANEL}
				rx="3"
				fill="var(--surface)"
				stroke={p === 2 ? 'var(--cat-2)' : 'var(--line)'}
				stroke-width={p === 2 ? 1.3 : 1}
			/>

			<g clip-path="url(#ld-clip-{p})">
				{#each GRIDS[p] as line, i (i)}
					<path
						d={line.d}
						fill="none"
						stroke="var(--ink-3)"
						stroke-width={line.axis ? 0.9 : 0.7}
						opacity={line.axis ? 0.5 : 0.26}
					/>
				{/each}

				{#each RING as pt, i (i)}
					{@const [ux, uy] = through(pt, p)}
					<circle cx={sx(p, ux)} cy={sy(p, uy)} r="2.1" fill="var(--warm)" />
				{/each}
				{#each DISK as pt, i (i)}
					{@const [ux, uy] = through(pt, p)}
					<circle cx={sx(p, ux)} cy={sy(p, uy)} r="2.1" fill="var(--accent)" />
				{/each}
			</g>

			<text x={XS[p] + PANEL / 2} y="18" text-anchor="middle" class="cap {p === 2 ? 'teal' : ''}">
				{stage.head}
			</text>
			<text x={XS[p] + PANEL / 2} y={TOP + PANEL + 21} text-anchor="middle" class="label">
				{stage.name}
			</text>
			<text x={XS[p] + PANEL / 2} y={TOP + PANEL + 34} text-anchor="middle" class="cap dim">
				{stage.sub}
			</text>
		{/each}

		<!-- each operation, written on the arrow that performs it -->
		{#each ARROWS as a (a.key)}
			<line
				x1={XS[a.from] + PANEL + 8}
				y1={TOP + PANEL / 2}
				x2={XS[a.to] - 8}
				y2={TOP + PANEL / 2}
				stroke="var(--ink-3)"
				stroke-width="1"
				marker-end="url(#ld-arrow)"
			/>
			<text
				x={(XS[a.from] + PANEL + XS[a.to]) / 2}
				y={TOP + PANEL / 2 - 9}
				text-anchor="middle"
				class="cap {a.tone}"
			>
				{a.eq}
			</text>
		{/each}
	</svg>
</Plate>

<style>
	.label {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 12.5px;
		fill: var(--ink-2);
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-2);
	}
	.dim {
		fill: var(--ink-3);
	}
	.accent {
		fill: var(--accent);
	}
	.teal {
		fill: var(--cat-2);
	}
</style>
