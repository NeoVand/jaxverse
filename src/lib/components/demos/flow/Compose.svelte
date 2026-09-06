<script lang="ts">
	// Two prompts at once.
	//
	// The three rows are the argument: prompt A alone, prompt B alone, and the
	// sum of the two pushes. Nothing about the model was trained to combine —
	// it never saw a cat with a heart for a face. What it saw is each idea
	// separately, and what the sampler does is add the two directions.
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { Shuffle, Sparkles } from 'lucide-svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/emoji/Tiles.svelte';
	import { makeVocab, parsePrompt } from '$lib/diffusion/corpus';
	import { coalesce } from '$lib/diffusion/lab.svelte';
	import { COMBOS, lab } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	const SHOWN = 6;

	let promptA = $state(COMBOS[0][0]);
	let promptB = $state(COMBOS[0][1]);
	// Two pushes stack, so each is set below the single-prompt sweet spot of
	// two — at three apiece the sum saturates and both ideas turn to poster paint.
	let wA = $state(2);
	let wB = $state(2);
	let style = $state<number | null>(null);
	let seed = $state(515);
	let onlyA = $state<Float32Array | null>(null);
	let onlyB = $state<Float32Array | null>(null);
	let both = $state<Float32Array | null>(null);
	let busy = $state(false);
	let combo = 0;

	const vocab = $derived(makeVocab(lab.info?.tags ?? []));

	const draw = coalesce(async () => {
		if (lab.phase !== 'ready') return;
		busy = true;
		const a = { tags: parsePrompt(promptA, vocab).tags, style };
		const b = { tags: parsePrompt(promptB, vocab).tags, style };
		const common = { steps: 20, seed, fromShipped: lab.hasShipped };
		try {
			onlyA = (await lab.sample(SHOWN, { ...common, a, guidanceA: wA })).pixels;
			onlyB = (await lab.sample(SHOWN, { ...common, a: b, guidanceA: wB })).pixels;
			both = (await lab.sample(SHOWN, { ...common, a, b, guidanceA: wA, guidanceB: wB })).pixels;
		} catch {
			// disposed mid-flight
		}
		busy = false;
	});

	// Three rows per redraw and two sliders driving it: coalesce, so the last
	// weight the reader chose is the one that ends up on the page.
	$effect(() => {
		void wA;
		void wB;
		void style;
		void seed;
		if (lab.phase === 'ready') draw();
	});

	function shuffle() {
		combo = (combo + 1) % COMBOS.length;
		[promptA, promptB] = COMBOS[combo];
		seed = Math.floor(Math.random() * 100000);
	}
</script>

<Plate id="compose" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing three rows…</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn kind="primary" disabled={busy || lab.phase !== 'ready'} onclick={draw}>
			<Sparkles size={12} aria-hidden="true" /> Draw
		</Btn>
		<Btn disabled={busy || lab.phase !== 'ready'} onclick={shuffle}>
			<Shuffle size={12} aria-hidden="true" /> Another pair
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					Composing two prompts needs WebGPU. What the plate shows: a row for each prompt on its
					own, and a third row where both pushes are applied at once and the model draws something
					that is in neither the corpus nor any of the eight styles it learned from.
				</p>
			</div>
		{:else}
			<div class="flex flex-col gap-3 p-4 sm:p-5">
				<div class="grid gap-2 sm:grid-cols-2">
					<label class="flex items-center gap-2">
						<span class="eyebrow shrink-0" style="color: var(--accent);">A</span>
						<input
							class="w-full rounded-[var(--r-2)] border border-line bg-surface px-2.5 py-1 font-serif text-[13.5px] text-ink italic outline-none focus-visible:shadow-[var(--focus-ring)]"
							bind:value={promptA}
							aria-label="First prompt"
						/>
					</label>
					<label class="flex items-center gap-2">
						<span class="eyebrow shrink-0" style="color: var(--cat-8);">B</span>
						<input
							class="w-full rounded-[var(--r-2)] border border-line bg-surface px-2.5 py-1 font-serif text-[13.5px] text-ink italic outline-none focus-visible:shadow-[var(--focus-ring)]"
							bind:value={promptB}
							aria-label="Second prompt"
						/>
					</label>
				</div>

				{#each [{ p: onlyA, l: 'A alone', c: 'var(--accent)' }, { p: onlyB, l: 'B alone', c: 'var(--cat-8)' }, { p: both, l: 'both at once', c: 'var(--good)' }] as row (row.l)}
					<div class="flex items-center gap-3">
						<span class="eyebrow w-20 shrink-0 text-[9.5px]" style="color: {row.c};">{row.l}</span>
						<Tiles
							pixels={row.p}
							count={SHOWN}
							columns={SHOWN}
							class="h-12 sm:h-16"
							label="Six emoji for {row.l}"
						/>
					</div>
				{/each}

				<div class="grid gap-x-8 gap-y-3 border-t border-line-soft pt-3 sm:grid-cols-2">
					<Slider label="how much A" bind:value={wA} min={0} max={8} step={0.5} tone="model" />
					<Slider label="how much B" bind:value={wB} min={0} max={8} step={0.5} tone="model-2" />
				</div>
				<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Drawing style">
					<span class="eyebrow mr-1">hand</span>
					<button
						class="chip"
						class:chip-on={style === null}
						aria-pressed={style === null}
						onclick={() => (style = null)}>any</button
					>
					{#each lab.info?.sets ?? [] as set, i (set.id)}
						<button
							class="chip"
							class:chip-on={style === i}
							aria-pressed={style === i}
							title={set.credit}
							onclick={() => (style = i)}>{set.label}</button
						>
					{/each}
				</span>
			</div>
		{/if}
	</div>
</Plate>
