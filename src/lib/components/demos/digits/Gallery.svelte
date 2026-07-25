<script lang="ts">
	// Plate I — the dataset itself, before any model touches it. Auto-loads on
	// scroll; data only, no engine. loadMnist caches the in-flight promise
	// module-wide, so the classifier below reuses this same download.
	import { Shuffle } from 'lucide-svelte';
	import { loadMnist, type MnistData } from '$lib/data/mnist';
	import Btn from '$lib/components/ui/Btn.svelte';
	import { inview } from '$lib/components/ui/inview';
	import { themePulse, watchTheme } from './digits-context.svelte';
	import { DIM, blit, hexRgb, inkImage, readTokens } from './common';

	const COLS = 16;
	const COUNT = COLS * 8; // 128 digits — a full page of the collection

	interface Pick {
		t: number; // row in the training set
		y: number; // its label, captured so the template never reads `data`
	}

	let phase = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let errorMsg = $state('');
	let ids = $state<Pick[]>([]);
	let data: MnistData | null = null; // plain — large, read on demand in painters

	watchTheme();

	function pick(): Pick[] {
		if (!data) return [];
		const n = data.trainY.length;
		const out: Pick[] = [];
		while (out.length < Math.min(COUNT, n)) {
			const t = Math.floor(Math.random() * n);
			if (!out.some((p) => p.t === t)) out.push({ t, y: data.trainY[t] });
		}
		return out;
	}

	async function boot() {
		if (phase !== 'idle') return;
		phase = 'loading';
		errorMsg = '';
		try {
			data = await loadMnist();
			ids = pick();
			phase = 'ready';
		} catch (err) {
			phase = 'error';
			errorMsg = err instanceof Error ? err.message : String(err);
		}
	}

	function retry() {
		phase = 'idle';
		boot();
	}

	function tile(t: number) {
		return (el: HTMLCanvasElement) => {
			void themePulse.tick; // repaint on theme flips
			if (!data) return;
			blit(el, [inkImage(data.trainX, t * DIM, hexRgb(readTokens(el).ink))], 56);
		};
	}
</script>

<div class="flex flex-col" use:inview={boot}>
	<div
		class="flex flex-wrap items-center justify-between gap-2 border-b border-line-soft px-4 py-2.5"
	>
		<span class="eyebrow">{COUNT} of 8,000 training digits</span>
		{#if phase === 'ready'}
			<Btn onclick={() => (ids = pick())}>
				<Shuffle size={12} aria-hidden="true" /> Reshuffle
			</Btn>
		{/if}
	</div>

	{#if phase === 'ready'}
		<div
			class="grid gap-px bg-line-soft p-px"
			style="grid-template-columns: repeat({COLS}, minmax(0, 1fr));"
		>
			{#each ids as d (d.t)}
				<canvas
					class="block aspect-square w-full bg-surface"
					aria-label="a handwritten {d.y}"
					{@attach tile(d.t)}
				></canvas>
			{/each}
		</div>
	{:else if phase === 'error'}
		<div class="flex h-40 items-center justify-center gap-3">
			<span class="text-[12.5px] text-bad">{errorMsg}</span>
			<Btn onclick={retry}>Retry</Btn>
		</div>
	{:else}
		<div class="flex h-40 items-center justify-center">
			<span class="eyebrow">fetching ten thousand digits (≈1.7 MB, cached)…</span>
		</div>
	{/if}
</div>
