<script lang="ts">
	// Plate II — the comparator. The reader is the annotator, and this plate is
	// the entire data-collection stage of RLHF: two candidates, one verdict, no
	// scores anywhere. Everything downstream on this page is fitted to what
	// happens here, which is the point — a reward model is its annotators'
	// taste in a box, and on this page the annotator is you.
	import { ChevronLeft, ChevronRight, RotateCcw } from 'lucide-svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import { progress } from '$lib/data/progress.svelte';
	import type { Gene } from '$lib/optim-rl/preference';
	import Rosette from './Rosette.svelte';
	import { taste, TARGET_PAIRS, MIN_PAIRS } from './taste-context.svelte';

	let pair = $state<[Gene, Gene]>(taste.proposal());
	/** The card the reader last chose, held briefly so the choice registers. */
	let flash = $state<0 | 1 | null>(null);

	function choose(side: 0 | 1) {
		flash = side;
		taste.record(pair[side], pair[1 - side]);
		if (taste.count >= MIN_PAIRS) progress.reach('taste:judged');
		setTimeout(() => {
			flash = null;
			pair = taste.proposal();
		}, 140);
	}

	function skip() {
		taste.skip();
		pair = taste.proposal();
	}

	function reset() {
		taste.reset();
		pair = taste.proposal();
	}

	function onkey(ev: KeyboardEvent) {
		if (ev.key === 'ArrowLeft') choose(0);
		else if (ev.key === 'ArrowRight') choose(1);
		else if (ev.key === '0' || ev.key === 'Escape') skip();
		else return;
		ev.preventDefault();
	}

	// The last few verdicts, newest first — winner in ink, loser faded.
	// `taste.pairs` is a plain field (Float64Arrays have no business inside a
	// deep proxy), so the count is what makes this recompute.
	const recent = $derived.by(() => {
		void taste.count;
		return taste.pairs.slice(-9).reverse();
	});
	const done = $derived(taste.count >= TARGET_PAIRS);
</script>

<Plate
	id="pairs"
	title="The comparator"
	live
	caption="Two ornaments, one question, no scores anywhere. This is the whole data-collection stage of RLHF, and every model further down this page is fitted to what you do here. Twenty-four judgments is plenty; keep going if you are enjoying yourself. Left and right arrow keys work, and so does 0 for “no preference” — which real annotation interfaces also offer, because forcing a verdict out of a genuine tie is how you teach a model noise."
>
	{#snippet status()}
		<span>{taste.count} judged</span>
		{#if taste.skipped > 0}<span class="text-ink-3">· {taste.skipped} passed</span>{/if}
	{/snippet}
	{#snippet actions()}
		<Btn onclick={reset} title="Forget every judgment and start the chapter over">
			<RotateCcw size={12} /> Reset
		</Btn>
	{/snippet}

	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		class="cmp px-4 pb-1"
		role="group"
		aria-label="Choose the ornament you prefer"
		tabindex="-1"
		onkeydown={onkey}
	>
		<p class="ask font-serif">Which of these would you set on a title page?</p>

		<div class="arena">
			{#each [0, 1] as const as side (side)}
				<button
					class="card"
					class:flash={flash === side}
					onclick={() => choose(side)}
					aria-label={side === 0 ? 'Choose the left ornament' : 'Choose the right ornament'}
				>
					<Rosette gene={pair[side]} size={192} />
					<span class="pick">
						{#if side === 0}<ChevronLeft size={11} />{/if}
						this one
						{#if side === 1}<ChevronRight size={11} />{/if}
					</span>
				</button>
			{/each}
		</div>

		<div class="mt-4 flex flex-wrap items-center justify-center gap-3">
			<Btn onclick={skip} title="Record no preference and draw a new pair">no preference</Btn>
			<span class="num text-[11px] text-ink-3">
				{#if done}
					enough to fit a judge — and enough to fool one
				{:else}
					{TARGET_PAIRS - taste.count} more for a comfortable fit
				{/if}
			</span>
		</div>

		{#if recent.length}
			<div class="film" aria-label="Your recent verdicts">
				{#each recent as p, i (taste.count - i)}
					<span class="verdict" title="you preferred the upper ornament">
						<span class="won"><Rosette gene={p.winner} size={42} /></span>
						<span class="lost"><Rosette gene={p.loser} size={30} /></span>
					</span>
				{/each}
			</div>
		{/if}
	</div>
</Plate>

<style>
	.cmp:focus {
		outline: none;
	}
	.ask {
		text-align: center;
		font-size: 17px;
		font-style: italic;
		color: var(--ink-2);
		font-variation-settings: 'opsz' 18;
		margin: 0.35rem 0 1.1rem;
	}
	.arena {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: clamp(0.75rem, 4vw, 3rem);
		max-width: 34rem;
		margin: 0 auto;
	}
	.card {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.4rem;
		padding: 1rem 0.5rem 0.7rem;
		border: 1px solid var(--line-soft);
		border-radius: var(--r-3);
		background: var(--surface);
		transition:
			border-color 120ms,
			transform 120ms,
			background 120ms;
	}
	.card:hover {
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
		transform: translateY(-2px);
	}
	.card:focus-visible {
		outline: none;
		box-shadow: var(--focus-ring);
	}
	.card.flash {
		background: var(--accent-soft);
		border-color: var(--accent);
		transform: translateY(0);
	}
	.pick {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-family: var(--font-sans);
		font-size: 10.5px;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		color: var(--ink-3);
	}
	.card:hover .pick {
		color: var(--accent);
	}
	.film {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 0.55rem;
		margin-top: 1.35rem;
		padding-top: 0.9rem;
		border-top: 1px solid var(--line-soft);
	}
	.verdict {
		display: flex;
		align-items: center;
		gap: 1px;
		padding-right: 0.45rem;
		border-right: 1px solid var(--line-soft);
	}
	.verdict:last-child {
		border-right: none;
	}
	.lost {
		opacity: 0.3;
	}
	@media (max-width: 480px) {
		.arena {
			gap: 0.6rem;
		}
	}
</style>
