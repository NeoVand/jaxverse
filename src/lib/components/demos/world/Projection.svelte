<script lang="ts">
	interface Props {
		values: number[];
		label: string;
		warm?: boolean;
		emptyLabel?: string;
	}
	let { values, label, warm = false, emptyLabel = 'Waiting for a measurement' }: Props = $props();
	const bins = $derived.by(() => {
		const counts = Array.from({ length: 24 }, (_, i) => ({ center: -3 + (i + 0.5) / 4, count: 0 }));
		for (const value of values)
			counts[Math.max(0, Math.min(23, Math.floor((value + 3) * 4)))].count++;
		return counts;
	});
</script>

<svg
	viewBox="0 0 340 150"
	role="img"
	aria-label={`${label}: distribution of a fixed embedding coordinate. Horizontal range minus three to three; outer bins include the tails.`}
>
	{#if !values.length}<text x="170" y="72" text-anchor="middle" class="empty">{emptyLabel}</text
		>{/if}
	<line x1="20" y1="120" x2="320" y2="120" class="axis" />
	<line x1="170" y1="18" x2="170" y2="120" class="zero" />
	{#each bins as bin (bin.center)}
		<rect
			x={20 + (bin.center + 3) * 50 - 5.5}
			y={120 - (100 * bin.count) / Math.max(1, values.length)}
			width="11"
			height={(100 * bin.count) / Math.max(1, values.length)}
			fill={warm ? 'var(--warm)' : 'var(--accent)'}
			opacity="0.75"
		/>
	{/each}
	<text x="20" y="141" text-anchor="start">≤ −3</text><text x="170" y="141" text-anchor="middle"
		>0</text
	><text x="320" y="141" text-anchor="end">≥ 3</text>
	<text x="20" y="14">fraction of observations · fixed scale 0–1</text>
</svg>

<style>
	svg {
		width: 100%;
		display: block;
	}
	rect {
		transition:
			y 180ms ease,
			height 180ms ease;
	}
	@media (prefers-reduced-motion: reduce) {
		rect {
			transition: none;
		}
	}
	.axis {
		stroke: var(--line);
		stroke-width: 1;
	}
	.zero {
		stroke: var(--line-soft);
		stroke-width: 1;
	}
	.empty {
		font: 12px var(--font-sans);
		fill: var(--ink-2);
	}
	text {
		fill: var(--ink-3);
		font: 10px var(--font-mono);
	}
</style>
