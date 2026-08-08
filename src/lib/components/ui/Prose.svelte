<script lang="ts">
	import type { Snippet } from 'svelte';

	// The reading rail: 42rem of Newsreader, centered. The book has exactly two
	// rails — this one and the plate rail — and every element picks one.
	let { children }: { children: Snippet } = $props();
</script>

<div class="chapter-prose rail-prose">
	{@render children()}
</div>
