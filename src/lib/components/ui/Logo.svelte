<script lang="ts">
	// The mark: a loss curve descending to its minimum, where the ball rests.
	let { size = 20 }: { size?: number } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true" class="shrink-0">
	<path
		d="M4 6 Q 13 30 17 22 Q 20 16 28 18"
		stroke="var(--accent)"
		stroke-width="3.4"
		stroke-linecap="round"
	/>
	<circle cx="15.2" cy="24.4" r="3.4" fill="var(--warm)" />
</svg>
