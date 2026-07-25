<script lang="ts">
	// The quiet not-ready states every rook plate shares. Plates auto-load via
	// use:inview — this row is what shows while the shared engine wakes (and
	// the graceful fallback when the browser has no WebGPU).
	import Btn from '$lib/components/ui/Btn.svelte';
	import { lab } from './rook-context.svelte';
</script>

<div class="flex min-h-28 flex-col items-center justify-center gap-3 px-6 py-8 text-center">
	{#if lab.phase === 'no-webgpu'}
		<p class="max-w-md font-serif text-[15px] text-ink-2 italic">
			This chapter trains and samples a real transformer in your browser, which needs WebGPU. A
			current Chrome or Edge will run everything; the prose reads fine without it.
		</p>
	{:else if lab.phase === 'error'}
		<span class="text-[12.5px] text-bad">{lab.error}</span>
		<Btn onclick={() => void lab.power()}>Retry</Btn>
	{:else}
		<span class="eyebrow">waking Rook — corpus + trained checkpoint, ≈2 MB…</span>
		<p class="max-w-md text-[12px] text-ink-3">
			one 1.3M-parameter transformer on your GPU, shared by every plate in this chapter — nothing
			leaves this page
		</p>
	{/if}
</div>
