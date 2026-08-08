<script lang="ts">
	import type { Snippet } from 'svelte';
	import { getChapter, plateAnchor, plateNumber, roman } from '$lib/data/plates';

	// Every figure in this book is a plate, and every plate is a band: the page
	// steps onto slightly different paper, edge to edge between two hairlines,
	// and the figure sits on the plate rail inside it. No frame, no corners —
	// the change of material is the frame.
	//
	// The numeral is never typed at the call site. It comes from the plate's
	// position in $lib/data/plates, so the sequence and every sentence that
	// points into it stay honest when a figure is added or moved.
	interface Props {
		/** Registry id — its position decides the numeral. */
		id: string;
		title: string;
		caption?: string;
		/** Marks a plate that computes: it earns a live dot in the header. */
		live?: boolean;
		/** Right side of the header — status text, live indicators. */
		status?: Snippet;
		/** Far right of the header — the demo's transport (Train/Pause/Reset…). */
		actions?: Snippet;
		children: Snippet;
	}

	let { id, title, caption, live = false, status, actions, children }: Props = $props();

	const chapter = getChapter();
	const n = $derived(plateNumber(chapter(), id));
</script>

<div class="band" id={plateAnchor(id)}>
	<figure class="rail">
		<div class="plate-head">
			<span class="flex items-baseline gap-2.5">
				{#if n !== undefined}
					<span class="eyebrow plate-label">Plate {roman(n)}</span>
				{/if}
				<span class="plate-title">{title}</span>
				{#if live}
					<span class="plate-live" title="trains in your browser" aria-label="live"></span>
				{/if}
			</span>
			{#if status || actions}
				<span class="flex flex-wrap items-center gap-x-3 gap-y-1">
					{#if status}
						<span class="num flex items-center gap-2 text-[11px] text-ink-3"
							>{@render status()}</span
						>
					{/if}
					{#if actions}
						<span class="flex items-center gap-1.5">{@render actions()}</span>
					{/if}
				</span>
			{/if}
		</div>

		{@render children()}

		{#if caption}
			<figcaption class="plate-caption">{caption}</figcaption>
		{/if}
	</figure>
</div>
