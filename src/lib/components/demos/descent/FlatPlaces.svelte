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
	// Both panels are the same twenty-four directions and the same arithmetic.
	// Only the sign of the curvature along three of them differs, which is the
	// entire distinction and the reason it is so hard to come by in the plural.

	const SLICES = 24;

	/**
	 * The three that fall, at index 3, 11 and 19 — spread through the bundle
	 * rather than adjacent to it. Their curvatures and leans are given by hand
	 * rather than drawn from the same spread as the rest, because three
	 * downward parabolas through one point at similar depths land on top of one
	 * another and read as a single thick line, which is the opposite of the
	 * thing this panel is counting.
	 */
	const DOWN: Record<number, { a: number; b: number }> = {
		3: { a: -0.2, b: 0.11 },
		11: { a: -0.34, b: -0.15 },
		19: { a: -0.5, b: 0.19 }
	};

	/**
	 * A deterministic stand-in for randomness. Two golden-ratio strides give
	 * numbers that spread evenly and look unstructured, and — unlike a seeded
	 * generator — a figure that never trains is identical on every render,
	 * every reload and every machine.
	 */
	const spread = (i: number, k: number) => ((i + 1) * 0.6180339887 + k * 0.3247179572) % 1;

	/** The loss along one line: curvature, plus a little third-order lean so
	 *  the bundle reads as a landscape rather than a bouquet of parabolas. */
	function slice(i: number, saddle: boolean): { d: string; down: boolean } {
		const down = saddle && i in DOWN;
		const a = down ? DOWN[i].a : 0.2 + 0.7 * spread(i, 0);
		const b = down ? DOWN[i].b : (spread(i, 1) - 0.5) * 0.34;
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
	const H = 132;
	const TOP = 26;
	const XS = [30, 278] as const;
	/** The loss window, chosen so the deepest descent and the steepest rise
	 *  both stay inside the frame with a little air. */
	const LO = -0.78;
	const HI = 1.14;

	const px = (p: number, t: number) => XS[p] + ((t + 1) / 2) * W;
	const py = (L: number) => TOP + ((HI - L) / (HI - LO)) * H;

	const PANELS = [
		{ key: 'floor', saddle: false, name: 'a floor', sub: 'every direction curves up' },
		{ key: 'saddle', saddle: true, name: 'a saddle', sub: 'three of the twenty-four do not' }
	] as const;

	const BUNDLES = PANELS.map((p) => Array.from({ length: SLICES }, (_, i) => slice(i, p.saddle)));
</script>

<Plate
	id="flat"
	title="The two kinds of flat"
	caption="Twenty-four straight lines drawn through one flat point of a loss surface, and the loss along each of them — which is how the question gets asked of a real model, since nobody can look at the surface itself. Left: every line curves upward. The point is the bottom of a bowl, and a walker that arrives is staying. Right: the same point, except that three of the twenty-four curve down instead, and a walker carrying any momentum or any noise will eventually stumble into one of the three. With two directions to agree on, floors are easy to come by. With a million, a floor is an extraordinary coincidence — and almost every flat place a large network stalls at is the picture on the right."
>
	<svg
		viewBox="0 0 520 198"
		class="mx-auto block w-full max-w-[760px]"
		role="img"
		aria-label="Two panels, each showing the loss along twenty-four straight lines drawn through a single flat point of a loss surface. In the left panel, labelled a floor, every one of the twenty-four curves upward away from the point. In the right panel, labelled a saddle, twenty-one curve upward and three curve downward, drawn in vermilion. The point itself is marked in both."
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
				<!-- the height of the flat point itself, so up and down are readable -->
				<line
					x1={XS[p]}
					y1={py(0)}
					x2={XS[p] + W}
					y2={py(0)}
					stroke="var(--ink-3)"
					stroke-width="0.8"
					stroke-dasharray="3 3"
					opacity="0.55"
				/>

				<!-- rising lines first, so the three that fall sit on top of them -->
				{#each BUNDLES[p].filter((s) => !s.down) as s, i (i)}
					<path d={s.d} fill="none" stroke="var(--ink-3)" stroke-width="1" opacity="0.38" />
				{/each}
				{#each BUNDLES[p].filter((s) => s.down) as s, i (i)}
					<path
						d={s.d}
						fill="none"
						stroke="var(--warm)"
						stroke-width="1.5"
						opacity="0.9"
						stroke-linecap="round"
					/>
				{/each}
			</g>

			<circle cx={px(p, 0)} cy={py(0)} r="3.1" fill="var(--accent)" />

			<text x={XS[p] + W / 2} y={TOP + H + 22} text-anchor="middle" class="label">{panel.name}</text
			>
			<text x={XS[p] + W / 2} y={TOP + H + 35} text-anchor="middle" class="cap dim"
				>{panel.sub}</text
			>
		{/each}

		<text x={XS[0] + W / 2} y="16" text-anchor="middle" class="cap dim">
			the loss along each direction
		</text>
		<text x={XS[1] + W / 2} y="16" text-anchor="middle" class="cap dim">
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
