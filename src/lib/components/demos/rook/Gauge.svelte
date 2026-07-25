<script lang="ts">
	// A quiet horizontal meter: eyebrow label, hairline track, tonal fill.
	interface Props {
		label: string;
		/** 0..1, or null before the first measurement. */
		value: number | null;
		tone?: 'accent' | 'warm' | 'good';
		/** Optional dimmed annotation after the percentage. */
		detail?: string;
	}

	let { label, value, tone = 'accent', detail }: Props = $props();

	const toneColor = $derived(
		tone === 'warm' ? 'var(--warm)' : tone === 'good' ? 'var(--good)' : 'var(--accent)'
	);
	const pct = $derived(value === null ? 0 : Math.max(0, Math.min(1, value)) * 100);
</script>

<div>
	<div class="mb-1 flex items-baseline justify-between gap-3">
		<span class="eyebrow">{label}</span>
		<span class="num text-[12px] text-ink">
			{value === null ? '—' : `${(value * 100).toFixed(0)}%`}{#if detail}<span class="text-ink-3">
					· {detail}</span
				>{/if}
		</span>
	</div>
	<div class="h-1.5 overflow-hidden rounded-full bg-surface-2">
		<div
			class="h-full rounded-full"
			style="width: {pct.toFixed(1)}%; background: {toneColor}; transition: width 400ms ease;"
		></div>
	</div>
</div>
