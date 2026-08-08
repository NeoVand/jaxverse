<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// The classifier's shape, drawn once: pixels in, ten scores out, softmax
	// turning those scores into a belief. Static by design — it is the map the
	// reader keeps in mind while the live plates below do the work.
	import { softmax } from './common';

	const LAYERS = [784, 128, 128, 10];
	// bar height ∝ √width, so 784 and 10 can share one 56px glyph
	const GLYPH = LAYERS.map((w, i) => {
		const h = Math.round(Math.sqrt(w) * 1.86 * 10) / 10;
		return { x: 126 + i * 26, y: 58 - h / 2, h };
	});

	// one plausible verdict, so the two bar groups are honest about each other
	const LOGITS = [-1.2, 0.4, -0.9, 1.1, -1.7, 0.3, -0.5, 3.1, -1.0, 0.7];
	const PROBS = softmax(LOGITS, 0, 10);
	const zLim = Math.max(...LOGITS.map(Math.abs));
	const pLim = Math.max(...PROBS);

	const Z_X = 278; // left edge of the score bars
	const P_X = 490; // left edge of the probability bars
	const SLOT = 8;
	const BAR = 5;
	const Z_MID = 58; // the zero rule for signed scores
	const P_BASE = 84; // the floor the probabilities stand on
</script>

<Plate
	id="pipeline"
	title="The shape of the classifier"
	caption="The whole machine in one line. Everything left of the softmax is the network you have already built twice; everything right of it is a probability distribution over the ten answers, and the tallest bar is what the model says."
>
	<svg
		viewBox="0 0 578 124"
		class="mx-auto block w-full max-w-[900px]"
		role="img"
		aria-label="The classifier as a pipeline: a 28 by 28 image becomes 784 numbers, the network turns them into ten scores of any size, and the softmax turns those into ten probabilities that sum to one. The largest probability is the verdict."
	>
		<defs>
			<marker
				id="pipe-arrow"
				viewBox="0 0 8 8"
				refX="6.5"
				refY="4"
				markerWidth="6"
				markerHeight="6"
				orient="auto-start-reverse"
			>
				<path d="M 0 0.8 L 7 4 L 0 7.2 Z" fill="var(--ink-3)" />
			</marker>
		</defs>

		<!-- the image, as the machine receives it -->
		<rect x="6" y="30" width="56" height="56" rx="5" fill="var(--surface)" stroke="var(--line)" />
		{#each [1, 2, 3, 4, 5, 6] as k (k)}
			<line
				x1={6 + k * 8}
				y1="30"
				x2={6 + k * 8}
				y2="86"
				stroke="var(--line-soft)"
				stroke-width="0.6"
			/>
			<line
				x1="6"
				y1={30 + k * 8}
				x2="62"
				y2={30 + k * 8}
				stroke="var(--line-soft)"
				stroke-width="0.6"
			/>
		{/each}
		<path
			d="M 20 44 L 48 44 L 32 76"
			fill="none"
			stroke="var(--ink)"
			stroke-width="6"
			stroke-linecap="round"
			stroke-linejoin="round"
			opacity="0.82"
		/>
		<text x="34" y="103" text-anchor="middle" class="cap">28 × 28</text>
		<text x="34" y="115" text-anchor="middle" class="cap dim">784 numbers</text>

		<line
			x1="70"
			y1="58"
			x2="106"
			y2="58"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#pipe-arrow)"
		/>

		<!-- the network: the same stack every chapter has been building -->
		<rect
			x="112"
			y="24"
			width="112"
			height="68"
			rx="7"
			fill="var(--surface)"
			stroke="var(--line)"
		/>
		{#each GLYPH as g (g.x)}
			<rect
				x={g.x}
				y={g.y}
				width="10"
				height={g.h}
				rx="1.5"
				fill="none"
				stroke="var(--ink-3)"
				stroke-width="1"
			/>
		{/each}
		<text x="168" y="16" text-anchor="middle" class="label">the network</text>
		<text x="168" y="107" text-anchor="middle" class="cap dim">layers of weights</text>

		<line
			x1="232"
			y1="58"
			x2="272"
			y2="58"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#pipe-arrow)"
		/>

		<!-- ten scores, free to be any size, either sign -->
		<line
			x1={Z_X - 4}
			y1={Z_MID}
			x2={Z_X + 10 * SLOT}
			y2={Z_MID}
			stroke="var(--line)"
			stroke-width="1"
		/>
		{#each LOGITS as z, i (i)}
			{@const h = (Math.abs(z) / zLim) * 28}
			<rect
				x={Z_X + i * SLOT + (SLOT - BAR) / 2}
				y={z >= 0 ? Z_MID - h : Z_MID}
				width={BAR}
				height={Math.max(1, h)}
				rx="1"
				fill={z >= 0 ? 'var(--accent)' : 'var(--warm)'}
				opacity="0.85"
			/>
		{/each}
		<text x={Z_X + 40} y="16" text-anchor="middle" class="label">the scores · z</text>
		<text x={Z_X + 40} y="103" text-anchor="middle" class="cap">ten numbers</text>
		<text x={Z_X + 40} y="115" text-anchor="middle" class="cap dim">any size, either sign</text>

		<!-- the squashing step -->
		<line
			x1={Z_X + 10 * SLOT + 6}
			y1={Z_MID}
			x2="390"
			y2={Z_MID}
			stroke="var(--ink-3)"
			stroke-width="1"
		/>
		<rect
			x="388"
			y={Z_MID - 13}
			width="68"
			height="26"
			rx="13"
			fill="var(--surface)"
			stroke="var(--cat-2)"
		/>
		<text x="422" y={Z_MID + 4.5} text-anchor="middle" class="pill">softmax</text>
		<line
			x1="456"
			y1={Z_MID}
			x2={P_X - 6}
			y2={Z_MID}
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#pipe-arrow)"
		/>

		<!-- ten probabilities, a budget of exactly one -->
		<line
			x1={P_X - 4}
			y1={P_BASE}
			x2={P_X + 10 * SLOT}
			y2={P_BASE}
			stroke="var(--line)"
			stroke-width="1"
		/>
		{#each PROBS as p, i (i)}
			{@const h = (p / pLim) * 46}
			<rect
				x={P_X + i * SLOT + (SLOT - BAR) / 2}
				y={P_BASE - Math.max(1, h)}
				width={BAR}
				height={Math.max(1, h)}
				rx="1"
				fill="var(--cat-2)"
				opacity={p === pLim ? 0.95 : 0.5}
			/>
		{/each}
		<text x={P_X + 40} y="16" text-anchor="middle" class="label">the belief · p</text>
		<text x={P_X + 40} y="103" text-anchor="middle" class="cap">ten probabilities</text>
		<text x={P_X + 40} y="115" text-anchor="middle" class="cap dim">they sum to one</text>
		<text x={P_X + 7 * SLOT + BAR / 2} y={P_BASE - 44} text-anchor="middle" class="verdict">7</text>
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
	.pill {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 13px;
		fill: var(--cat-2);
	}
	.verdict {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		fill: var(--cat-2);
	}
</style>
