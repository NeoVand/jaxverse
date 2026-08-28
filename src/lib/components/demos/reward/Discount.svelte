<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';

	// What γ actually does, drawn once.
	//
	// The chapter states the discounted return as an equation and then asks the
	// reader to carry it — but the equation says "near consequences weigh more"
	// without saying how much more, and that is the whole content of the knob.
	// Forty decisions and one score at the end is the chapter's own example, so
	// the picture uses it: forty ticks, each as tall as the credit the final
	// reward hands that decision, at three discounts. The eye reads how far
	// back the reward reaches without doing any arithmetic, and the numbers
	// beside the first tick say it exactly.
	//
	// γ = 0.97 is not a round number chosen to sit between the other two. It is
	// what the sailing chart on the next plate actually runs at, which is why
	// it is the one drawn in ink.

	const N = 40; // the chapter's forty decisions
	// No per-row gloss. Three sentences of interpretation down the left margin
	// is the caption's job done twice, and the strings are wider than the
	// margin can hold — check:labels caught all three hanging off the left of
	// the frame, by as much as 62 units. The row carries its discount and the
	// share its first decision keeps; the caption says what that means.
	const ROWS = [0.9, 0.97, 0.99] as const;

	// ── geometry, in viewBox units ──
	const X0 = 84;
	const X1 = 470;
	const PITCH = (X1 - X0) / (N - 1);
	const BAR = 5.2;
	const H = 44; // a bar at full credit
	const BASE = [80, 152, 224] as const;

	/** Credit the terminal reward hands decision `i`, counting from the first.
	 *  The last decision is undelayed and keeps all of it; decision i is
	 *  N−1−i steps earlier, so it keeps γ to that power. */
	const credit = (g: number, i: number) => Math.pow(g, N - 1 - i);
	const pct = (v: number) => `${v < 0.1 ? (v * 100).toFixed(1) : Math.round(v * 100)}%`;
</script>

<Plate
	id="discount"
	title="How far back a reward reaches"
	caption="One passage, forty decisions, and a single score at the end — the chapter's own example, drawn. Each tick is one decision, and its height is the share of that final score the discounted return hands it: γ raised to the number of steps it happened before the end. The knob is doing more than it looks. At γ = 0.90 the opening moves of the passage are credited with almost nothing, so a learner cannot tell a good start from a bad one and only ever tunes its endgame. At γ = 0.99 the first decision still carries two-thirds of the blame, and every bad decision anywhere in the passage is charged to every other — which is a fair rule and a hopelessly noisy one. Between them sits 0.97, what the chart on the next plate actually sails at. This is the whole of credit assignment in one line: nothing tells you which of the forty was the mistake, so you spread the answer over all of them, and γ is how you spread it."
>
	<svg
		viewBox="0 0 520 258"
		class="mx-auto block w-full max-w-[760px]"
		role="img"
		aria-label="Three rows of forty ticks, one row per discount value. Each tick is a decision in a forty-step passage and its height is the share of the final reward credited to it, which is the discount raised to the number of steps before the end. At a discount of 0.90 the row is flat until the last handful of decisions, and the first decision receives 1.6 per cent. At 0.97, drawn in ultramarine because it is the value this chapter's sailing demo uses, the row slopes gently and the first decision receives 30 per cent. At 0.99 the row is almost level and the first decision receives 68 per cent. A vermilion mark at the right of every row is where the single reward arrives."
	>
		{#each ROWS as g, r (g)}
			{@const accent = g === 0.97}
			<!-- the row's own baseline: the decisions of one passage, in order -->
			<line
				x1={X0 - 6}
				y1={BASE[r]}
				x2={X1 + 30}
				y2={BASE[r]}
				stroke="var(--line)"
				stroke-width="1"
			/>

			{#each Array.from({ length: N }, (_, i) => i) as i (i)}
				{@const c = credit(g, i)}
				<rect
					x={X0 + i * PITCH - BAR / 2}
					y={BASE[r] - H * c}
					width={BAR}
					height={Math.max(0.6, H * c)}
					rx="1"
					fill={accent ? 'var(--accent)' : 'var(--ink-3)'}
					opacity={accent ? 0.9 : 0.62}
				/>
			{/each}

			<!-- where the score arrives, once, after the last decision -->
			<circle cx={X1 + 20} cy={BASE[r] - H} r="3.4" fill="var(--warm)" />

			<text x={X0 - 14} y={BASE[r] + 4} text-anchor="end" class="knob">γ = {g.toFixed(2)}</text>

			<!-- the first decision's share, said exactly, beside the tick that shows it -->
			<text
				x={X0 - 2}
				y={BASE[r] - Math.max(12, H * credit(g, 0)) - 6}
				class="num-lead"
				fill={accent ? 'var(--accent)' : 'var(--ink-2)'}>{pct(credit(g, 0))}</text
			>
		{/each}

		<text x={X0} y={BASE[2] + 22} class="cap">first decision</text>
		<text x={X1 + 24} y={BASE[2] + 22} text-anchor="end" class="cap">fortieth</text>
		<text x={X1 + 20} y={BASE[0] - H - 12} text-anchor="middle" class="cap warm">reward</text>
	</svg>
</Plate>

<style>
	.knob {
		font-family: var(--font-mono);
		font-size: 10.5px;
		fill: var(--cat-1);
	}
	.cap {
		font-family: var(--font-mono);
		font-size: 8.5px;
		fill: var(--ink-3);
	}
	.warm {
		fill: var(--warm);
	}
	.num-lead {
		font-family: var(--font-mono);
		font-size: 10px;
	}
</style>
