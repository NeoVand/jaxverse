<script lang="ts">
	// The pre-ready face the scribe's plates share. Booting is automatic (each
	// plate calls scribe.boot() from use:inview), so this is a status panel, not
	// a gate — the only button here is Retry. No WebGPU → prose, not a dead demo.
	import Btn from '$lib/components/ui/Btn.svelte';
	import { scribe } from './lab.svelte';

	interface Props {
		/** Shown while idle or booting — what is being fetched and built. */
		note: string;
		tall?: boolean;
	}

	let { note, tall = false }: Props = $props();
</script>

{#if scribe.phase === 'no-webgpu'}
	<div class="px-6 py-10">
		<p
			class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
			style="font-variation-settings: 'opsz' 14;"
		>
			This chapter's model needs WebGPU, and this browser doesn't offer it — Chrome or Edge on a
			desktop do. The text still reads fine without it: the captions describe what each plate shows.
		</p>
	</div>
{:else}
	<div
		class="flex flex-col items-center justify-center gap-4 px-6 py-8"
		class:min-h-80={tall}
		class:min-h-52={!tall}
	>
		{#if scribe.phase === 'error'}
			<p class="num max-w-md text-center text-[12px]" style="color: var(--bad);">
				{scribe.errorMsg || 'the training worker failed to start'}
			</p>
			<Btn onclick={() => void scribe.boot()}>Retry</Btn>
		{:else}
			<span class="num text-center text-[12px] text-ink-3">{scribe.loadNote || note}</span>
		{/if}
	</div>
{/if}
