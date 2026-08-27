<script lang="ts">
	import Plate from '$lib/components/ui/Plate.svelte';
	// The chapter's game, drawn once before any machinery: cover the next word
	// and guess. Two prompts, two very different beliefs — one nearly certain,
	// one spread over a dictionary. Static by design; the plates below play
	// this game for real.

	// one plausible belief per prompt (illustrative, hand-set)
	const EASY = [
		{ w: 'tail', p: 0.78 },
		{ w: 'head', p: 0.07 },
		{ w: 'paw', p: 0.05 },
		{ w: 'nose', p: 0.03 },
		{ w: 'bone', p: 0.02 }
	];
	// a flat comb: many candidates, none favoured
	const HARD_N = 14;
	const HARD = Array.from({ length: HARD_N }, (_, i) => 0.09 - 0.004 * i);
	const HARD_WORDS = ['cat', 'blue', 'pizza', 'the', 'seven'];

	const BASE = 108; // the comb's floor
	const TALL = 52; // bar height at p = 1 would be off-scale; scale to the max
	const easyMax = Math.max(...EASY.map((d) => d.p));
	const hardMax = Math.max(...HARD);
</script>

<Plate
	id="game"
	title="Guess the covered word"
	caption="The game, before any machinery: guess the covered word. A guess is a whole belief — a probability for every word you know — and how sharply it peaks is how easy the puzzle was."
>
	<svg
		viewBox="0 0 620 160"
		class="mx-auto block w-full max-w-[900px]"
		role="img"
		aria-label="The next-token game: two sentences each missing their last word. For 'the little dog wagged his', the belief over candidates is nearly all on 'tail' — low surprise. For 'my favorite word is', the belief is spread thin over many words — high surprise."
	>
		<!-- ── left: barely a puzzle ── -->
		<text x="148" y="22" text-anchor="middle" class="sentence">
			the little dog wagged his
			<tspan class="mask-txt">&#8202;</tspan>
		</text>
		<rect
			x="238"
			y="9"
			width="26"
			height="17"
			rx="3.5"
			fill="var(--warm-soft)"
			stroke="var(--warm)"
		/>
		<text x="251" y="21.5" text-anchor="middle" class="mask-q">?</text>

		<line
			x1="148"
			y1="34"
			x2="148"
			y2="48"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#game-arrow)"
		/>

		{#each EASY as d, i (d.w)}
			{@const h = (d.p / easyMax) * TALL}
			{@const x = 56 + i * 40}
			<rect
				x={x - 7}
				y={BASE - h}
				width="14"
				height={h}
				rx="1.5"
				fill="var(--cat-2)"
				opacity={i === 0 ? 0.95 : 0.45}
			/>
			<text {x} y={BASE + 13} text-anchor="middle" class="cand" class:cand-top={i === 0}>{d.w}</text
			>
			{#if i === 0}
				<text {x} y={BASE - h - 6} text-anchor="middle" class="p-top">{d.p.toFixed(2)}</text>
			{/if}
		{/each}
		<text x="246" y={BASE + 1} class="cap dim">…</text>
		<line x1="40" y1={BASE} x2="256" y2={BASE} stroke="var(--line)" stroke-width="1" />
		<text x="148" y="145" text-anchor="middle" class="cap">barely a puzzle</text>
		<text x="148" y="156" text-anchor="middle" class="cap dim">one word soaks up the belief</text>

		<!-- the fold between the two prompts -->
		<line x1="310" y1="8" x2="310" y2="150" stroke="var(--line-soft)" stroke-width="1" />

		<!-- ── right: no favourite anywhere in the dictionary ── -->
		<text x="452" y="22" text-anchor="middle" class="sentence"> my favorite word is </text>
		<rect
			x="524"
			y="9"
			width="26"
			height="17"
			rx="3.5"
			fill="var(--warm-soft)"
			stroke="var(--warm)"
		/>
		<text x="537" y="21.5" text-anchor="middle" class="mask-q">?</text>

		<line
			x1="452"
			y1="34"
			x2="452"
			y2="48"
			stroke="var(--ink-3)"
			stroke-width="1"
			marker-end="url(#game-arrow)"
		/>

		{#each HARD as p, i (i)}
			{@const h = (p / hardMax) * TALL * 0.32}
			{@const x = 356 + i * 14.5}
			<rect
				x={x - 4.5}
				y={BASE - h}
				width="9"
				height={h}
				rx="1.5"
				fill="var(--cat-2)"
				opacity="0.45"
			/>
			{#if i % 3 === 0 && i / 3 < HARD_WORDS.length}
				<text {x} y={BASE + 13} text-anchor="middle" class="cand">{HARD_WORDS[i / 3]}</text>
			{/if}
		{/each}
		<text x="566" y={BASE + 1} class="cap dim">…</text>
		<line x1="344" y1={BASE} x2="576" y2={BASE} stroke="var(--line)" stroke-width="1" />
		<text x="452" y="145" text-anchor="middle" class="cap">genuinely hard</text>
		<text x="452" y="156" text-anchor="middle" class="cap dim">the belief spread thin</text>

		<defs>
			<marker
				id="game-arrow"
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
	</svg>
</Plate>

<style>
	.sentence {
		font-family: var(--font-serif);
		font-style: italic;
		font-size: 14px;
		fill: var(--ink);
	}
	.mask-q {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		fill: var(--warm);
	}
	.cand {
		font-family: var(--font-mono);
		font-size: 9px;
		fill: var(--ink-2);
	}
	.cand-top {
		fill: var(--cat-2);
		font-weight: 600;
	}
	.p-top {
		font-family: var(--font-mono);
		font-size: 9px;
		font-weight: 600;
		fill: var(--cat-2);
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
