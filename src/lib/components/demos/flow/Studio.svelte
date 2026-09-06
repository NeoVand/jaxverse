<script lang="ts">
	// The studio. Type something, choose a hand to draw it in, and the model
	// makes eight of them.
	//
	// A prompt is not free text to this model — it is a bag of tags out of a
	// 310-word vocabulary — so words it does not know are shown crossed out
	// rather than silently ignored. A demo that quietly does nothing is worse
	// than one that says no.
	import { Sparkles, Shuffle } from 'lucide-svelte';
	import Plate from '$lib/components/ui/Plate.svelte';
	import Btn from '$lib/components/ui/Btn.svelte';
	import Slider from '$lib/components/ui/Slider.svelte';
	import { inview } from '$lib/components/ui/inview';
	import Tiles from '$lib/components/demos/emoji/Tiles.svelte';
	import { makeVocab, parsePrompt } from '$lib/diffusion/corpus';
	import { coalesce } from '$lib/diffusion/lab.svelte';
	import { lab, SHOWN, SUGGESTIONS } from './lab.svelte';

	interface Props {
		title: string;
		caption: string;
	}
	let { title, caption }: Props = $props();

	let prompt = $state('smiling cat face');
	let style = $state<number | null>(null);
	let guidance = $state(2);
	let steps = $state(20);
	let seed = $state(2026);
	let pixels = $state<Float32Array | null>(null);
	let ms = $state(0);
	let busy = $state(false);
	let suggestion = 0;

	const vocab = $derived(makeVocab(lab.info?.tags ?? []));
	const parsed = $derived(parsePrompt(prompt, vocab));

	const draw = coalesce(async () => {
		if (lab.phase === 'no-webgpu' || !lab.info) return;
		busy = true;
		try {
			const r = await lab.sample(SHOWN, {
				steps,
				a: { tags: parsed.tags, style },
				guidanceA: guidance,
				seed,
				fromShipped: lab.hasShipped
			});
			pixels = r.pixels;
			ms = r.ms;
		} catch {
			// disposed mid-flight
		}
		busy = false;
	});

	// Redraw on every dial, but not on every keystroke — the prompt only takes
	// effect when the reader asks for it, so typing does not stutter the GPU.
	$effect(() => {
		void style;
		void guidance;
		void steps;
		void seed;
		if (lab.phase === 'ready') draw();
	});

	function surprise() {
		prompt = SUGGESTIONS[suggestion++ % SUGGESTIONS.length];
		seed = Math.floor(Math.random() * 100000);
	}
</script>

<Plate id="studio" live {title} {caption}>
	{#snippet status()}
		{#if lab.phase === 'no-webgpu'}
			<span>needs WebGPU</span>
		{:else if busy}
			<span>drawing…</span>
		{:else if pixels}
			<span>{SHOWN} pictures</span>
			<span aria-hidden="true">·</span>
			<span>{steps} steps</span>
			<span aria-hidden="true">·</span>
			<span>{ms.toFixed(0)} ms</span>
		{/if}
	{/snippet}
	{#snippet actions()}
		<Btn kind="primary" disabled={busy || lab.phase !== 'ready'} onclick={draw}>
			<Sparkles size={12} aria-hidden="true" /> Draw
		</Btn>
		<Btn
			disabled={busy || lab.phase !== 'ready'}
			onclick={surprise}
			title="A prompt and a new seed"
		>
			<Shuffle size={12} aria-hidden="true" /> Surprise me
		</Btn>
	{/snippet}

	<div class="flex flex-col" use:inview={() => void lab.boot()}>
		{#if lab.phase === 'no-webgpu'}
			<div class="px-6 py-10">
				<p
					class="mx-auto max-w-md text-center font-serif text-[15px] leading-relaxed text-ink-2 italic"
					style="font-variation-settings: 'opsz' 14;"
				>
					The studio draws on your own GPU, and that needs WebGPU — a current Chrome or Edge on a
					desktop will run it. Everything the chapter argues is in the prose and the captions.
				</p>
			</div>
		{:else if lab.phase === 'loading' || lab.phase === 'idle'}
			<div class="flex h-[320px] flex-col items-center justify-center gap-1">
				<span class="eyebrow">fetching the model (2.5 MB) and the emoji sheets…</span>
				<span class="text-[12.5px] text-ink-3">nothing leaves the page</span>
			</div>
		{:else}
			<div class="grid gap-x-6 gap-y-4 p-4 sm:p-5 lg:grid-cols-[1fr_20rem]">
				<div class="flex flex-col gap-2">
					<Tiles
						{pixels}
						count={SHOWN}
						columns={4}
						gap={4}
						class="h-40 sm:h-56"
						label="Eight emoji drawn for the current prompt"
					/>
					<span class="num text-[10px] text-ink-3">
						{parsed.tags.length === 0
							? 'no prompt — this is the model drawing whatever it likes'
							: `${parsed.matched.length} tag${parsed.matched.length === 1 ? '' : 's'} · guidance ${guidance.toFixed(1)}`}
					</span>
				</div>

				<div class="flex flex-col gap-3.5">
					<label class="flex flex-col gap-1.5">
						<span class="eyebrow">prompt</span>
						<input
							class="w-full rounded-[var(--r-2)] border border-line bg-surface px-2.5 py-1.5 font-serif text-[14px] text-ink italic outline-none focus-visible:shadow-[var(--focus-ring)]"
							bind:value={prompt}
							onkeydown={(e) => e.key === 'Enter' && draw()}
							placeholder="a smiling cat face"
							aria-label="What to draw"
						/>
					</label>

					<div class="flex flex-wrap items-baseline gap-1.5 text-[11px]">
						{#if parsed.matched.length}
							{#each parsed.matched as t (t)}
								<span
									class="rounded-[var(--r-1)] px-1.5 py-0.5"
									style="background: var(--accent-soft); color: var(--accent);">{t}</span
								>
							{/each}
						{/if}
						{#each parsed.ignored as t (t)}
							<span class="text-ink-3 line-through" title="not in the vocabulary">{t}</span>
						{/each}
						{#if !parsed.matched.length && !parsed.ignored.length}
							<span class="text-ink-3">the vocabulary has {lab.info?.tags.length ?? 0} words</span>
						{/if}
					</div>

					<span class="flex flex-wrap items-center gap-1" role="group" aria-label="Drawing style">
						<span class="eyebrow mr-1 w-full">hand</span>
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

					<Slider
						label="guidance w"
						bind:value={guidance}
						min={0}
						max={8}
						step={0.5}
						tone="knob"
						format={(v) => (v === 0 ? '0 · ignore the prompt' : v.toFixed(1))}
					/>
					<Slider
						label="steps"
						bind:value={steps}
						min={2}
						max={50}
						step={1}
						tone="op"
						format={(v) => String(v)}
					/>
				</div>
			</div>
		{/if}
	</div>
</Plate>
