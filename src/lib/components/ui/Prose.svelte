<script lang="ts">
	import type { Snippet } from 'svelte';

	// The reading column: 42rem of Newsreader, centered.
	let { children }: { children: Snippet } = $props();
</script>

<div class="chapter-prose mx-auto max-w-2xl px-5">
	{@render children()}
</div>
