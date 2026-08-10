<script lang="ts">
	// The model's own writing, as a row of pointable tokens: colour carries the
	// verdict (warm = capture, bad = refused, dim = unverifiable), and pointing
	// at one plays it on whichever board the owning plate put beside this.
	import type { FilmView } from './film-view.svelte';

	interface Props {
		view: FilmView;
		/** Shown above the row, right-aligned, when there is something to point at. */
		hint?: boolean;
	}

	let { view, hint = true }: Props = $props();
</script>

<div class="flex min-w-0 flex-col gap-2">
	<div class="flex flex-wrap items-baseline justify-between gap-x-4">
		<span class="eyebrow">the sentence it wrote</span>
		{#if hint && view.playable.length > 0}
			<span class="num text-[10px] text-ink-3">point at a move to see it played</span>
		{/if}
	</div>

	<!-- the row and the verdict are floored at the height they usually take, so a
	     shorter game than the last one cannot pull the whole plate up between two
	     measurements. Floored at two lines, not the worst case: a rare third line
	     nudges the plate once, which reads better than a permanent hole. -->
	<div class="flex min-h-[3.2rem] flex-wrap content-start gap-1">
		<span class="tok tok-marker">⟨game⟩</span>
		{#each view.sentence as p (p.ply)}
			{#if p.state === 'unchecked'}
				<span class="tok tok-dim" title="unverifiable — the board was lost earlier">{p.uci}</span>
			{:else}
				<button
					type="button"
					class="tok"
					class:tok-bad={p.state === 'illegal'}
					class:tok-warm={p.capture}
					class:tok-on={view.shown?.ply === p.ply}
					onmouseenter={() => view.point(p.ply)}
					onmouseleave={() => view.point(null)}
					onfocus={() => view.point(p.ply)}
					onblur={() => view.point(null)}
				>
					{p.uci}{#if p.state === 'illegal'}✕{:else if p.capture}×{/if}
				</button>
			{/if}
		{/each}
		{#if view.hiddenTail > 0}
			<span class="tok tok-dim">+{view.hiddenTail} more</span>
		{:else if view.ended}
			<span class="tok tok-marker">⟨game⟩</span>
		{/if}
	</div>

	<p class="min-h-[2.6rem] text-[12px] leading-relaxed text-ink-3">{view.verdict}</p>
	<p class="num flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-ink-3">
		<span><span style="color: var(--warm);">×</span> capture</span>
		<span><span style="color: var(--bad);">✕</span> refused by the judge</span>
		<span class="opacity-65">grey · unverifiable</span>
	</p>
</div>

<style>
	/* Colour carries the verdict; hover and selection only tint the ground, so a
	   token never changes its meaning by being looked at. */
	.tok {
		font-family: var(--font-mono);
		font-size: 11px;
		line-height: 1.45;
		padding: 1px 4px;
		border: 1px solid transparent;
		border-radius: 3px;
		color: var(--ink-2);
		cursor: default;
	}
	button.tok:hover,
	button.tok:focus-visible {
		background: color-mix(in srgb, var(--accent) 12%, transparent);
	}
	.tok-on {
		border-color: color-mix(in srgb, var(--accent) 40%, transparent);
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.tok-warm {
		color: var(--warm);
	}
	.tok-bad {
		color: var(--bad);
	}
	.tok-dim {
		color: var(--ink-3);
		opacity: 0.65;
	}
	.tok-marker {
		color: var(--ink-3);
	}
</style>
