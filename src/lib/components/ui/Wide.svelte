<script lang="ts">
	import type { Snippet } from 'svelte';

	// The demo column: wider than prose, still framed by the page.
	let { children }: { children: Snippet } = $props();
</script>

<div class="mx-auto max-w-5xl px-5">
	{@render children()}
</div>
