<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';

	// The difference between the two kinds of flat place, drawn the only way it
	// can be drawn once a landscape has more than three axes: not as a surface,
	// but as the loss along a bundle of straight lines through one point.
	//
	// This is the picture behind the paragraph about saddles, and it is also
	// how the question is actually asked of a real network — you cannot see a
	// million-dimensional surface, but you can walk a line across it and write
	// down what happens, as often as you like.
	//
	// Two things this figure has to get right, and an earlier version of it got
	// both wrong. The first is that a direction going DOWN must look as
	// emphatic as a direction going up: the loss window is symmetric about the
	// flat point and the falling curvatures match the rising ones in
	// magnitude, so the horizon runs through the middle of the panel and each
	// half belongs to one sign. The second is that the panels differ by three
	// curves in twenty-four, which is far too little ink to read out of a
	// tangle — hence the row of marks beneath each one, one per direction and
	// in the same order, where the count can simply be counted.

	const SLICES = 24;

	/**
	 * The three that fall, at index 3, 11 and 19 — spread through the bundle
	 * rather than adjacent to it. Given by hand rather than drawn from the same
	 * spread as the rest, so that three domes through one point land at clearly
	 * different depths instead of on top of one another.
	 *
	 * The lean stays well under |2a/3| in each case, which is where a cubic
	 * turns its own dome back over: these have to fall on BOTH sides of the
	 * point, or they are drawing a slope rather than a direction that descends.
	 */
	const DOWN: Record<number, { a: number; b: number }> = {
		3: { a: -0.42, b: 0.1 },
		11: { a: -0.66, b: -0.08 },
		19: { a: -0.92, b: 0.12 }
	};

	/**
	 * A deterministic stand-in for randomness. Two golden-ratio strides give
	 * numbers that spread evenly and look unstructured, and — unlike a seeded
	 * generator — a figure that never trains is identical on every render,
	 * every reload and every machine.
	 */
	const spread = (i: number, k: number) => ((i + 1) * 0.6180339887 + k * 0.3247179572) % 1;

	interface Slice {
		d: string;
		down: boolean;
	}

	/** The loss along one line: curvature, plus a little third-order lean so
	 *  the bundle does not come out as a set of perfectly nested mirrors. */
	function slice(i: number, saddle: boolean): Slice {
		const down = saddle && i in DOWN;
		const a = down ? DOWN[i].a : 0.3 + 0.65 * spread(i, 0);
		const b = down ? DOWN[i].b : (spread(i, 1) - 0.5) * 0.28;
		let d = '';
		for (let k = 0; k <= 48; k++) {
			const t = -1 + (2 * k) / 48;
			const L = a * t * t + b * t * t * t;
			d += `${k === 0 ? 'M' : 'L'} ${px(saddle ? 1 : 0, t).toFixed(2)} ${py(L).toFixed(2)} `;
		}
		return { d: d.trim(), down };
	}

	// ── geometry, in viewBox units ──
	const W = 212;
	const H = 128;
	const TOP = 24;
	const XS = [30, 278] as const;
	/** Symmetric on purpose: the flat point sits at the middle of the panel, so
	 *  the half above the horizon belongs to the directions that rise and the
	 *  half below to the directions that fall. An asymmetric window quietly
	 *  argues that one of the two matters more. */
	const SPAN = 1.15;

	const px = (p: number, t: number) => XS[p] + ((t + 1) / 2) * W;
	const py = (L: number) => TOP + ((SPAN - L) / (2 * SPAN)) * H;

	// ── the tally: one mark per direction, in the bundle's own order ──
	const TALLY_Y = TOP + H + 17;
	const PITCH = W / SLICES;
	/** Half-width and half-height of one mark. */
	const MW = 3.5;
	const MH = 2.3;

	/** A quadratic whose apex sits MH past its ends: ∪ for a direction that
	 *  rises away from the point, ∩ for one that falls. */
	function mark(p: number, i: number, down: boolean): string {
		const cx = XS[p] + (i + 0.5) * PITCH;
		const end = down ? TALLY_Y + MH : TALLY_Y - MH;
		const ctrl = down ? TALLY_Y - 3 * MH : TALLY_Y + 3 * MH;
		return `M ${(cx - MW).toFixed(2)} ${end} Q ${cx.toFixed(2)} ${ctrl} ${(cx + MW).toFixed(2)} ${end}`;
	}

	const PANELS = [
		{ key: 'floor', saddle: false, name: 'a floor', sub: 'every direction curves up' },
		{ key: 'saddle', saddle: true, name: 'a saddle', sub: 'three of the twenty-four do not' }
	] as const;

	const BUNDLES = PANELS.map((p) => Array.from({ length: SLICES }, (_, i) => slice(i, p.saddle)));
</script>

<Plate
	id="flat"
	title="The two kinds of flat"
	caption="Twenty-four straight lines drawn through one flat point of a loss surface, and the loss along each of them — which is how the question gets asked of a real model, since nobody can look at the surface itself. The dashed horizon is the height of the point; above it the ground rises, below it the ground falls. Left: every line curves upward. The point is the bottom of a bowl, and a walker that arrives is staying. Right: the same point, except that three of the twenty-four turn over instead, and a walker carrying any momentum or any noise will eventually stumble into one of the three. The row of marks under each panel is one per direction in the same order, so the count is there to be checked rather than taken on trust. With two directions to agree on, floors are easy to come by. With a million, a floor is an extraordinary coincidence — and almost every flat place a large network stalls at is the picture on the right."
>
	<svg
		viewBox="0 0 520 214"
		class="mx-auto block w-full max-w-[760px]"
		role="img"
		aria-label="Two panels, each showing the loss along twenty-four straight lines drawn through a single flat point of a loss surface. A dashed horizon marks the height of the point, halfway up each panel. In the left panel, labelled a floor, all twenty-four curves rise away from the point into the upper half. In the right panel, labelled a saddle, twenty-one rise and three turn over and fall into the lower half, drawn in vermilion. Beneath each panel is a row of twenty-four small marks, one per direction and in the same order: a cup for a direction that rises, a cap for one that falls. The left row is twenty-four cups; the right row has caps in the fourth, twelfth and twentieth places."
	>
		<defs>
			{#each PANELS as panel, p (panel.key)}
				<clipPath id="fp-clip-{p}">
					<rect x={XS[p]} y={TOP} width={W} height={H} rx="3" />
				</clipPath>
			{/each}
		</defs>

		{#each PANELS as panel, p (panel.key)}
			<rect
				x={XS[p]}
				y={TOP}
				width={W}
				height={H}
				rx="3"
				fill="var(--surface)"
				stroke="var(--line)"
				stroke-width="1"
			/>

			<g clip-path="url(#fp-clip-{p})">
				<!-- the height of the flat point itself: the line each direction
				     either leaves upward or leaves downward -->
				<line
					x1={XS[p]}
					y1={py(0)}
					x2={XS[p] + W}
					y2={py(0)}
					stroke="var(--ink-3)"
					stroke-width="0.8"
					stroke-dasharray="3 3"
					opacity="0.6"
				/>

				<!-- rising lines first, so the ones that fall sit on top of them -->
				{#each BUNDLES[p].filter((s) => !s.down) as s, i (i)}
					<path d={s.d} fill="none" stroke="var(--ink-3)" stroke-width="1" opacity="0.4" />
				{/each}
				{#each BUNDLES[p].filter((s) => s.down) as s, i (i)}
					<path
						d={s.d}
						fill="none"
						stroke="var(--warm)"
						stroke-width="1.7"
						opacity="0.95"
						stroke-linecap="round"
					/>
				{/each}
			</g>

			<circle cx={px(p, 0)} cy={py(0)} r="3.1" fill="var(--accent)" />

			<!-- one mark per direction, in the bundle's own order -->
			{#each BUNDLES[p] as s, i (i)}
				<path
					d={mark(p, i, s.down)}
					fill="none"
					stroke={s.down ? 'var(--warm)' : 'var(--ink-3)'}
					stroke-width={s.down ? 1.7 : 1.1}
					opacity={s.down ? 0.95 : 0.6}
					stroke-linecap="round"
				/>
			{/each}

			<text x={XS[p] + W / 2} y={TOP + H + 42} text-anchor="middle" class="label">{panel.name}</text
			>
			<text x={XS[p] + W / 2} y={TOP + H + 55} text-anchor="middle" class="cap dim"
				>{panel.sub}</text
			>
		{/each}

		<text x={XS[0] + W / 2} y="14" text-anchor="middle" class="cap dim">
			the loss along each direction
		</text>
		<text x={XS[1] + W / 2} y="14" text-anchor="middle" class="cap dim">
			the same point, three ways out
		</text>
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
</style>
